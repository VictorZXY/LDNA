import copy
import os

import ogb.graphproppred
import pandas as pd
import torch
import torch_geometric.transforms as T
from ogb.graphproppred import PygGraphPropPredDataset
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch import nn
from torch_geometric.datasets import HeterophilousGraphDataset, MNISTSuperpixels, Planetoid, ZINC
from torch_geometric.loader import DataLoader

import models
from data_utils import add_eig_vecs, canonicalize_graphs, degree_histogram, node_rank, sort_graphs
from data_utils.code2 import encode_y_to_arr, get_vocab_mapping
from data_utils.exp import PlanarSATPairsDataset
from data_utils.transforms import AddDepthField, AddDepthToX, RemoveEdgeAttr, ToUndirectedNoAttr, UnsqueezeTargetDim
from utils.evaluator import (AccuracyEvaluator, Code2Evaluator, MicroF1Evaluator, MNISTEvaluator,
                             OGBNodeEvaluator, ROCAUCEvaluator, ZINCEvaluator)

# Datasets predicted per node rather than per graph. They take a different route through the
# resolver — no canonical sort, no graph readout, one graph per "batch" — and the readout guard
# below keys off this tuple to check that a config has not crossed the two levels.
NODE_DATASETS = ('Cora', 'Questions')

# Every model's readout chain is an `if`/`elif` with no `else`, so an unrecognized value does not
# raise: it silently skips pooling and the model returns one row per node. That is exactly the
# node-level behaviour, which makes the failure mode invisible at graph level until a metric comes
# out wrong. The recognized set is listed once here and checked before any model is built.
READOUT_CHOICES = (None, 'add', 'max', 'mean', 'gru', 'attention')


class FullGraphLoader:
    # Transductive node-level "loader": the whole graph is a single batch, so this is a one-element
    # iterable rather than a DataLoader. It carries `split` — the name of the boolean node mask that
    # selects this loader's nodes — which train.py reads to restrict the loss and the metrics to
    # that split. Wrapping the graph this way keeps the training loop a plain `for batch in loader`
    # shared with the graph-level path, and keeps `len(loader)` correct for its per-epoch loss mean.
    def __init__(self, data, split):
        self.data = data
        self.split = split

    def __iter__(self):
        yield self.data

    def __len__(self):
        return 1


def _planar_cv_folds(dataset, batch_size, num_folds=10):
    # 10-fold CV loaders for EXP/CEXP, transcribed from the GNN-RNI reference (GNNHyb.py). The
    # dataset is left UNSHUFFLED so each 1-WL-indistinguishable (2i, 2i+1) pair stays intact, and
    # every fold is a contiguous block. For fold i: the test set is the i-th block of
    # len(dataset)//num_folds graphs; the remaining graphs form the training pool; the validation
    # set is the i-th block of len(pool)//num_folds graphs carved from that pool; the rest is the
    # training set. The reference used uint8 masks, which PyG 2.7 no longer accepts as an index, so
    # the masks are torch.bool and are inverted with ~ — the resulting partition is identical.
    folds = []
    n = len(dataset) // num_folds
    for i in range(num_folds):
        test_mask = torch.zeros(len(dataset), dtype=torch.bool)
        test_mask[i * n:(i + 1) * n] = True
        test_dataset = dataset[test_mask]
        train_dataset = dataset[~test_mask]

        m = len(train_dataset) // num_folds
        val_mask = torch.zeros(len(train_dataset), dtype=torch.bool)
        val_mask[i * m:(i + 1) * m] = True
        val_dataset = train_dataset[val_mask]
        train_dataset = train_dataset[~val_mask]

        folds.append((
            DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
            DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
            DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
        ))
    return folds


def model_and_data_resolver(model_query, dataset_query, **kwargs):
    model_kwargs = kwargs.get('model_args', {})
    dataset_kwargs = kwargs.get('data_args', {})
    batch_size = dataset_kwargs.pop('batch_size', 1)

    model_choices = ['LDNA', 'DeeperGCN', 'DGN', 'EGC', 'GraphSAGE', 'GAT', 'GATv2', 'GCN', 'GIN', 'GINE', 'PNA', 'GNN-VPA']
    dataset_choices = ['MNISTSuperpixels', 'ZINC', 'ogbg-molhiv', 'ogbg-molpcba', 'ogbg-code2',
                       'EXP', 'CEXP'] + list(NODE_DATASETS)

    # Load the dataset
    if dataset_query == 'MNISTSuperpixels':
        transform = T.Compose([
            T.Cartesian(cat=False),
            RemoveEdgeAttr()
        ])
        train_dataset = MNISTSuperpixels(train=True, pre_transform=transform, **dataset_kwargs)
        val_dataset = test_dataset = MNISTSuperpixels(train=False, pre_transform=transform, **dataset_kwargs)

        train_dataset = sort_graphs(train_dataset, sort_y=False)
        val_dataset = test_dataset = sort_graphs(test_dataset, sort_y=False)

        # Only DGN reads the Laplacian eigenvector field, so it is attached only for DGN and its
        # presence is what routes it into the model. It is computed AFTER the canonical sort, which
        # permutes the nodes without realigning any extra per-node attribute, and (for the OGB
        # datasets below) BEFORE the splits, since an index view cannot be re-collated.
        if model_query == 'DGN':
            num_eig_vec = model_kwargs.get('num_eig_vec', 1)
            train_dataset = add_eig_vecs(train_dataset, k=num_eig_vec, cache_name='train')
            val_dataset = test_dataset = add_eig_vecs(test_dataset, k=num_eig_vec, cache_name='test')
    elif dataset_query == 'ZINC':
        transform = UnsqueezeTargetDim()
        subset = dataset_kwargs.pop('subset', False)
        train_dataset = ZINC(subset=subset, split='train', pre_transform=transform, **dataset_kwargs)
        val_dataset = ZINC(subset=subset, split='val', pre_transform=transform, **dataset_kwargs)
        test_dataset = ZINC(subset=subset, split='test', pre_transform=transform, **dataset_kwargs)
        
        train_dataset = sort_graphs(train_dataset, sort_y=False)
        val_dataset = sort_graphs(val_dataset, sort_y=False)
        test_dataset = sort_graphs(test_dataset, sort_y=False)

        if model_query == 'DGN':
            num_eig_vec = model_kwargs.get('num_eig_vec', 1)
            train_dataset = add_eig_vecs(train_dataset, k=num_eig_vec, cache_name='train')
            val_dataset = add_eig_vecs(val_dataset, k=num_eig_vec, cache_name='val')
            test_dataset = add_eig_vecs(test_dataset, k=num_eig_vec, cache_name='test')
    elif dataset_query in ['ogbg-molhiv', 'ogbg-molpcba']:
        dataset = PygGraphPropPredDataset(name=dataset_query, **dataset_kwargs)
        split_idx = dataset.get_idx_split()

        dataset = sort_graphs(dataset, sort_y=False)
        if model_query == 'DGN':
            dataset = add_eig_vecs(dataset, k=model_kwargs.get('num_eig_vec', 1), cache_name='all')

        train_dataset = dataset[split_idx['train']]
        val_dataset = dataset[split_idx['valid']]
        test_dataset = dataset[split_idx['test']]
    elif dataset_query == 'ogbg-code2':
        # Edge-less AST task: source-code method-name prediction. The raw `y` is a variable-length
        # list of subtoken strings, so it is turned into a fixed-length index array by a target
        # vocabulary built from the TRAIN split only (below).
        dataset = PygGraphPropPredDataset(name=dataset_query, **dataset_kwargs)
        split_idx = dataset.get_idx_split()

        # Build the vocabulary from the raw training targets BEFORE any transform is attached, so
        # `dataset[i].y` still yields the raw token lists.
        seq_list = [dataset[i].y for i in split_idx['train']]
        vocab2idx, idx2vocab = get_vocab_mapping(seq_list, 5000)

        # AST type/attribute vocab sizes are read from the dataset's mapping tables at runtime
        # rather than hardcoded, so they track the exact dataset build.
        num_nodetypes = len(pd.read_csv(os.path.join(dataset.root, 'mapping', 'typeidx2type.csv.gz'))['type'])
        num_nodeattributes = len(pd.read_csv(os.path.join(dataset.root, 'mapping', 'attridx2attr.csv.gz'))['attr'])

        # Applied lazily per access (a `transform`, not a `pre_transform`) so the on-disk cache is
        # never rewritten: fold node depth into x, symmetrize connectivity without edge features,
        # and attach the encoded target array `y_arr` (the raw `y` token list is kept for the F1
        # evaluator).
        transforms = [
            AddDepthToX(),
            ToUndirectedNoAttr(),
            lambda d: encode_y_to_arr(d, vocab2idx, 5),
        ]
        # Only DGN reads a directional field; here it is the AST depth, which the lazy chain can
        # carry because it costs a view and a cast (the eigenvector field the other datasets use
        # would need a sparse solve per graph). Depth is one-dimensional, so unlike `add_eig_vecs`
        # it cannot honour `num_eig_vec > 1` and the mismatch is rejected here rather than
        # surfacing as an empty-slice broadcast in the first forward.
        if model_query == 'DGN':
            if model_kwargs.get('num_eig_vec', 1) != 1:
                raise ValueError("Argument `num_eig_vec` must be 1 for DGN on ogbg-code2, whose "
                                 "directional field is the one-dimensional AST depth")
            transforms.append(AddDepthField())
        dataset.transform = T.Compose(transforms)

        # No sort_graphs: the AST DFS node order is already canonical, and sorting would desync
        # node_depth and the list-valued target from the nodes.
        train_dataset = dataset[split_idx['train']]
        val_dataset = dataset[split_idx['valid']]
        test_dataset = dataset[split_idx['test']]
    elif dataset_query in ['EXP', 'CEXP']:
        # Edge-less expressiveness datasets. The node features (a binary node-type column) cannot
        # impose an order, so the permutation-invariance device is a nauty canonical labelling
        # (canonicalize_graphs) instead of the feature lexsort; LDNA reads that order via
        # rank_mode='position'. As with the OGB branches, the DGN directional field is attached
        # AFTER the canonical order (which permutes the nodes) and BEFORE any index slicing (an
        # index view cannot be re-collated).
        dataset = PlanarSATPairsDataset(**dataset_kwargs)
        dataset = canonicalize_graphs(dataset)
        if model_query == 'DGN':
            dataset = add_eig_vecs(dataset, k=model_kwargs.get('num_eig_vec', 1), cache_name='all')

        # The paper's 10-fold CV: one (train, val, test) loader triple per fold. Fold 0 is exposed
        # as the normal return values and the full list rides on the returned train_loader, from
        # which train.py selects cv_folds[run] per run (folds map one-to-one to runs).
        cv_folds = _planar_cv_folds(dataset, batch_size)
        train_dataset = cv_folds[0][0].dataset
    elif dataset_query in NODE_DATASETS:
        # Transductive node classification: one graph, boolean node masks, no splitting of the
        # data itself. Deliberately NO sort_graphs — the tie-aware rank is permutation-equivariant,
        # so the model sees the same rank gaps whether or not the nodes are physically relabelled
        # (see data_utils.node_rank), and skipping the permutation keeps the masks and `y` aligned
        # with the nodes without an extra realignment pass.
        if dataset_query == 'Questions':
            dataset = HeterophilousGraphDataset(name=dataset_query, **dataset_kwargs)
            data = dataset[0]
            # Binary task scored on a single logit (BCEWithLogitsLoss + ROC-AUC), so the integer
            # class vector becomes the float [N, 1] target that loss path expects; Cora's long
            # class indices stay as-is for CrossEntropyLoss.
            data.y = data.y.view(-1, 1).float()
        else:
            dataset = Planetoid(name=dataset_query, **dataset_kwargs)
            data = dataset[0]

        # Only LDNA reads the canonical rank, so as with DGN's eigenvector field it is attached only
        # for LDNA and its presence on the batch is what routes it into the forward. Attached before
        # the per-split copies built below, which all share it.
        if model_query == 'LDNA':
            data.nrank = node_rank(data)

        # degree_histogram (PNA/DGN) iterates its argument, and iterating a bare `Data` yields
        # (key, value) pairs, so the single graph is wrapped in a one-element list.
        train_dataset = [data]
    else:
        raise ValueError(f"Could not resolve dataset '{dataset_query}' among choices {dataset_choices}")

    # Split the dataset into train/val/test dataloaders. EXP/CEXP instead reuse the 10-fold CV
    # loaders built above, returning fold 0 and carrying the full fold list on the train_loader.
    if dataset_query in ['EXP', 'CEXP']:
        train_loader, val_loader, test_loader = cv_folds[0]
        train_loader.cv_folds = cv_folds
    elif dataset_query in NODE_DATASETS:
        # All loaders yield the same full graph and differ only in the split mask they tag, which
        # is what transductive means here: every node is seen, only the scored ones change.
        # Questions ships its 10 official splits as [N, 10] mask matrices, mapped to runs the way
        # the EXP/CEXP folds are: one loader triple per split column, fold 0 returned as the plain
        # loaders and the full list riding on the train_loader. Each fold's `Data` is a SHALLOW
        # copy — x/edge_index/y/nrank stay the same tensors and only the 1-D mask columns differ —
        # so the 48.9k x 301 feature matrix is held once, not ten times.
        if data.train_mask.dim() == 2:
            cv_folds = []
            for i in range(data.train_mask.size(1)):
                fold_data = copy.copy(data)
                fold_data.train_mask = data.train_mask[:, i]
                fold_data.val_mask = data.val_mask[:, i]
                fold_data.test_mask = data.test_mask[:, i]
                cv_folds.append((FullGraphLoader(fold_data, 'train_mask'),
                                 FullGraphLoader(fold_data, 'val_mask'),
                                 FullGraphLoader(fold_data, 'test_mask')))
            train_loader, val_loader, test_loader = cv_folds[0]
            train_loader.cv_folds = cv_folds
        else:
            train_loader = FullGraphLoader(data, 'train_mask')
            val_loader = FullGraphLoader(data, 'val_mask')
            test_loader = FullGraphLoader(data, 'test_mask')
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # What the CONFIG asked for, captured before the node branch below overwrites it. The guard
    # further down has to judge this value, not the post-update one: the node branch sets
    # `readout=None` unconditionally, so checking afterwards would pass any node config, including
    # one that wrongly carries a graph readout.
    config_readout = model_kwargs.get('readout', None)

    # Update model kwargs. The node/edge encoders map to the model width, so every conv is
    # hidden -> hidden (uniform width across all models) and residual skips are identity.
    embedding_dim = model_kwargs['hidden_channels']
    if dataset_query == 'MNISTSuperpixels':
        model_kwargs.update({
            'in_channels': embedding_dim,
            'node_encoder': nn.Linear(1, embedding_dim),
            'num_pred_heads': train_dataset.num_classes
        })
    elif dataset_query == 'ZINC':
        model_kwargs.update({
            'in_channels': embedding_dim,
            'edge_dim': embedding_dim,
            'node_encoder': models.Encoder(28, embedding_dim=embedding_dim, num_features=1),
            'edge_encoder': nn.Embedding(4, embedding_dim=embedding_dim),
            'num_pred_heads': 1
        })
    elif dataset_query in ['ogbg-molhiv', 'ogbg-molpcba']:
        model_kwargs.update({
            'in_channels': embedding_dim,
            'edge_dim': embedding_dim,
            'node_encoder': AtomEncoder(emb_dim=embedding_dim),
            'edge_encoder': BondEncoder(emb_dim=embedding_dim),
            'num_pred_heads': dataset.num_tasks
        })
    elif dataset_query == 'ogbg-code2':
        # Edge-less: no edge_dim / edge_encoder. `num_pred_heads=None` makes the backbone return
        # the pooled graph embedding, which the Code2Head wrapper turns into the sequence head.
        model_kwargs.update({
            'in_channels': embedding_dim,
            'node_encoder': models.ASTNodeEncoder(embedding_dim, num_nodetypes, num_nodeattributes, max_depth=20),
            'num_pred_heads': None
        })
    elif dataset_query in ['EXP', 'CEXP']:
        # Edge-less: no edge_dim / edge_encoder. The single node-type column takes values {0, 1}, so
        # the encoder has two embeddings over one feature; the head has two classes (binary label).
        model_kwargs.update({
            'in_channels': embedding_dim,
            'node_encoder': models.Encoder(2, embedding_dim=embedding_dim, num_features=1),
            'num_pred_heads': 2
        })
    elif dataset_query in NODE_DATASETS:
        # Edge-less: no edge_dim / edge_encoder. `readout` is set here rather than left to the YAML
        # because configargparse stringifies the nested `model_args` mapping, so a `readout: null`
        # in a config arrives at the model as the STRING 'None' — which matches no branch of the
        # readout chain and would therefore look correct while being an accident. Setting it here
        # makes per-node output the resolver's decision, checked by the guard below.
        # Questions is binary and scored by ROC-AUC on a single logit (BCEWithLogitsLoss), so its
        # head is one output wide — `dataset.num_classes` (2) would build a two-logit CE head that
        # neither the loss nor the evaluator expects.
        model_kwargs.update({
            'in_channels': embedding_dim,
            'node_encoder': nn.Linear(dataset.num_features, embedding_dim),
            'num_pred_heads': 1 if dataset_query == 'Questions' else dataset.num_classes,
            'readout': None
        })

    # The one place the level contract is enforced, for every model and dataset. The readout chain
    # in each model has no `else`, so neither an unrecognized value nor a level mismatch raises on
    # its own: a graph-level config that loses its readout silently starts predicting per node, and
    # a node-level config that keeps one silently pools the whole graph into a single row. Both are
    # caught here, before any model is built. The two mismatches get separate messages because they
    # are opposite mistakes — one config says too little, the other too much.
    if config_readout not in READOUT_CHOICES:
        raise ValueError(f"Could not resolve readout '{config_readout}' among choices "
                         f"{READOUT_CHOICES}")
    if dataset_query in NODE_DATASETS and config_readout is not None:
        raise ValueError(f"Node-level dataset '{dataset_query}' predicts one row per node, so its "
                         f"config must not set a graph `readout`; got '{config_readout}'")
    if dataset_query not in NODE_DATASETS and config_readout is None:
        raise ValueError(f"Graph-level dataset '{dataset_query}' needs a graph `readout` to pool "
                         f"nodes into one row per graph; none was set (choices "
                         f"{READOUT_CHOICES[1:]})")

    # Load the model
    if model_query == 'LDNA':
        model = models.LDNA(**model_kwargs)
    elif model_query == 'DeeperGCN':
        model = models.DeeperGCN(**model_kwargs)
    elif model_query == 'EGC':
        model = models.EGC(**model_kwargs)
    elif model_query == 'GraphSAGE' or model_query == 'SAGE':
        model = models.GraphSAGE(**model_kwargs)
    elif model_query == 'GAT':
        model = models.GAT(**model_kwargs)
    elif model_query == 'GATv2':
        model = models.GATv2(**model_kwargs)
    elif model_query == 'GCN':
        model = models.GCN(**model_kwargs)
    elif model_query == 'GIN':
        model = models.GIN(**model_kwargs)
    elif model_query == 'GINE':
        model = models.GINE(**model_kwargs)
    elif model_query == 'PNA':
        model = models.PNA(deg=degree_histogram(train_dataset), **model_kwargs)
    elif model_query == 'DGN':
        model = models.DGN(deg=degree_histogram(train_dataset), **model_kwargs)
    elif model_query in ('GNN-VPA', 'VPA'):
        model = models.VPA(**model_kwargs)
    else:
        raise ValueError(f"Could not resolve dataset '{model_query}' among choices {model_choices}")

    # ogbg-code2: wrap any backbone with the per-position sequence head. The wrapper also carries
    # `idx2vocab`, which the training loop uses to detect the code2 task and decode predictions.
    if dataset_query == 'ogbg-code2':
        model = models.Code2Head(model, emb_dim=embedding_dim, max_seq_len=5,
                                  num_vocab=len(vocab2idx), idx2vocab=idx2vocab)

    return model, train_loader, val_loader, test_loader


def loss_resolver(query, **kwargs):
    if hasattr(nn, query):
        cls = getattr(nn, query)
        if callable(cls):
            return cls(**kwargs)
        else:
            raise ValueError(f"Could not resolve loss '{query}'")
    else:
        raise ValueError(f"Could not resolve loss '{query}'")


def evaluator_resolver(query, **kwargs):
    choices = {
        'OGBNodeEvaluator': OGBNodeEvaluator,
        'OGBGraphPropPredEvaluator': ogb.graphproppred.Evaluator,
        'ZINCEvaluator': ZINCEvaluator,
        'MNISTEvaluator': MNISTEvaluator,
        'AccuracyEvaluator': AccuracyEvaluator,
        'ROCAUCEvaluator': ROCAUCEvaluator,
        'MicroF1Evaluator': MicroF1Evaluator,
        'Code2Evaluator': Code2Evaluator,
    }

    if query not in choices:
        raise ValueError(f"Could not resolve evaluator '{query}' among choices {choices.keys()}")

    return choices[query](**kwargs)
