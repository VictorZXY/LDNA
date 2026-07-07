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
from torch_geometric.utils import degree

import models
from utils import sort_graphs
from utils.code2 import encode_y_to_arr, get_vocab_mapping
from utils.evaluator import Code2Evaluator, MNISTEvaluator, ZINCEvaluator
from utils.transforms import AddDepthToX, RemoveEdgeAttr, ToUndirectedNoAttr, UnsqueezeTargetDim


def model_and_data_resolver(model_query, dataset_query, **kwargs):
    model_kwargs = kwargs.get('model_args', {})
    dataset_kwargs = kwargs.get('data_args', {})
    batch_size = dataset_kwargs.pop('batch_size', 1)

    model_choices = ['LDNA', 'DeeperGCN', 'EGC', 'GraphSAGE', 'GAT', 'GATv2', 'GCN', 'GIN', 'GINE', 'PNA', 'GNN-VPA']
    dataset_choices = ['MNISTSuperpixels', 'ZINC', 'ogbg-molhiv', 'ogbg-molpcba', 'ogbg-code2']

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
    elif dataset_query == 'ZINC':
        transform = UnsqueezeTargetDim()
        subset = dataset_kwargs.pop('subset', False)
        train_dataset = ZINC(subset=subset, split='train', pre_transform=transform, **dataset_kwargs)
        val_dataset = ZINC(subset=subset, split='val', pre_transform=transform, **dataset_kwargs)
        test_dataset = ZINC(subset=subset, split='test', pre_transform=transform, **dataset_kwargs)
        
        train_dataset = sort_graphs(train_dataset, sort_y=False)
        val_dataset = sort_graphs(val_dataset, sort_y=False)
        test_dataset = sort_graphs(test_dataset, sort_y=False)
    elif dataset_query in ['ogbg-molhiv', 'ogbg-molpcba']:
        dataset = PygGraphPropPredDataset(name=dataset_query, **dataset_kwargs)
        split_idx = dataset.get_idx_split()

        dataset = sort_graphs(dataset, sort_y=False)
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
        dataset.transform = T.Compose([
            AddDepthToX(),
            ToUndirectedNoAttr(),
            lambda d: encode_y_to_arr(d, vocab2idx, 5),
        ])

        # No sort_graphs: the AST DFS node order is already canonical, and sorting would desync
        # node_depth and the list-valued target from the nodes.
        train_dataset = dataset[split_idx['train']]
        val_dataset = dataset[split_idx['valid']]
        test_dataset = dataset[split_idx['test']]
    else:
        raise ValueError(f"Could not resolve dataset '{dataset_query}' among choices {dataset_choices}")

    # Split the dataset into train/val/test dataloaders
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
        # the pooled graph embedding, which the Code2Model wrapper turns into the sequence head.
        model_kwargs.update({
            'in_channels': embedding_dim,
            'node_encoder': models.ASTNodeEncoder(embedding_dim, num_nodetypes, num_nodeattributes, max_depth=20),
            'num_pred_heads': None
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
        # Compute the maximum in-degree in the training data.
        max_degree = -1
        for data in train_dataset:
            d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
            max_degree = max(max_degree, int(d.max()))

        # Compute the in-degree histogram tensor
        deg = torch.zeros(max_degree + 1, dtype=torch.long)
        for data in train_dataset:
            d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
            deg += torch.bincount(d, minlength=deg.numel())

        model = models.PNA(deg=deg, **model_kwargs)
    elif model_query in ('GNN-VPA', 'VPA'):
        model = models.VPA(**model_kwargs)
    else:
        raise ValueError(f"Could not resolve dataset '{model_query}' among choices {model_choices}")

    # ogbg-code2: wrap any backbone with the per-position sequence head. The wrapper also carries
    # `idx2vocab`, which the training loop uses to detect the code2 task and decode predictions.
    if dataset_query == 'ogbg-code2':
        model = models.Code2Model(model, emb_dim=embedding_dim, max_seq_len=5,
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
        'Code2Evaluator': Code2Evaluator,
    }

    if query not in choices:
        raise ValueError(f"Could not resolve evaluator '{query}' among choices {choices.keys()}")

    return choices[query](**kwargs)
