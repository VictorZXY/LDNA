import os

import ogb.graphproppred
import ogb.nodeproppred
import pandas as pd
import torch
import torch_geometric.transforms as T
from ogb.graphproppred import PygGraphPropPredDataset
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from torch import nn
from torch_geometric.datasets import MNISTSuperpixels, ZINC
from torch_geometric.loader import DataLoader

import models
from data_utils import add_eig_vecs, canonicalize_graphs, degree_histogram, sort_graphs
from data_utils.code2 import encode_y_to_arr, get_vocab_mapping
from data_utils.exp import PlanarSATPairsDataset
from data_utils.transforms import AddDepthField, AddDepthToX, RemoveEdgeAttr, ToUndirectedNoAttr, UnsqueezeTargetDim
from utils.evaluator import AccuracyEvaluator, Code2Evaluator, MNISTEvaluator, ZINCEvaluator


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
    dataset_choices = ['MNISTSuperpixels', 'ZINC', 'ogbg-molhiv', 'ogbg-molpcba', 'ogbg-code2', 'EXP', 'CEXP']

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
    else:
        raise ValueError(f"Could not resolve dataset '{dataset_query}' among choices {dataset_choices}")

    # Split the dataset into train/val/test dataloaders. EXP/CEXP instead reuse the 10-fold CV
    # loaders built above, returning fold 0 and carrying the full fold list on the train_loader.
    if dataset_query in ['EXP', 'CEXP']:
        train_loader, val_loader, test_loader = cv_folds[0]
        train_loader.cv_folds = cv_folds
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

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
        'OGBNodePropPredEvaluator': ogb.nodeproppred.Evaluator,
        'OGBGraphPropPredEvaluator': ogb.graphproppred.Evaluator,
        'ZINCEvaluator': ZINCEvaluator,
        'MNISTEvaluator': MNISTEvaluator,
        'AccuracyEvaluator': AccuracyEvaluator,
        'Code2Evaluator': Code2Evaluator,
    }

    if query not in choices:
        raise ValueError(f"Could not resolve evaluator '{query}' among choices {choices.keys()}")

    return choices[query](**kwargs)
