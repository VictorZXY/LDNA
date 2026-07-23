import hashlib
import os

import numpy as np
import torch
from scipy.sparse.csgraph import connected_components
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import degree, sort_edge_index, to_dense_adj


def sort_graph(graph: Data, sort_y=False):
    x_arr = graph.x.numpy()
    sorted_idx = np.lexsort([x_arr[:, i] for i in range(x_arr.shape[1])])
    sorted_idx = torch.from_numpy(sorted_idx).long()
    inv_sorted_idx = torch.empty_like(sorted_idx)
    inv_sorted_idx[sorted_idx] = torch.arange(sorted_idx.shape[0])

    graph.new2old = sorted_idx
    graph.old2new = inv_sorted_idx

    # Sort the nodes
    graph.x = graph.x[graph.new2old]
    if sort_y:
        graph.y = graph.y[graph.new2old]
    graph.edge_index = graph.old2new[graph.edge_index]

    # Sort the edge indices and attributes
    if hasattr(graph, 'edge_attr'):
        graph.edge_index, graph.edge_attr = sort_edge_index(graph.edge_index, graph.edge_attr)
    else:
        graph.edge_index = sort_edge_index(graph.edge_index)

    return graph


def sort_graphs(dataset: InMemoryDataset, sort_y=False):
    all_graphs = [dataset.get(i) for i in range(len(dataset))]
    sorted_graphs = [sort_graph(g, sort_y) for g in all_graphs]

    dataset.data, dataset.slices = dataset.collate(sorted_graphs)
    return dataset


def add_eig_vec(graph: Data, k=1):
    # DGN's directional aggregators need the k lowest-frequency non-trivial eigenvectors of the
    # combinatorial Laplacian L = D - A. PyG's AddLaplacianEigenvectorPE is deliberately not used:
    # it hardcodes the symmetric-normalized Laplacian, and its random per-column sign flip both
    # makes the field irreproducible and draws from the global torch RNG stream that
    # `torch_geometric.seed_everything` seeds, which would change the model init of this baseline
    # relative to every other one.
    adj = to_dense_adj(graph.edge_index, max_num_nodes=graph.num_nodes)[0]
    eig_vec = torch.zeros(graph.num_nodes, k)

    # The Laplacian has one zero eigenvalue per connected component, so on a disconnected graph
    # its lowest non-trivial eigenvectors degenerate into component indicators and the field
    # vanishes on every edge. The eigenvectors are therefore taken per component, as the paper
    # prescribes (7.5% of ogbg-molhiv graphs are disconnected). `eigh` returns the eigenvalues in
    # ascending order, so column 0 is the trivial constant eigenvector and the k lowest-frequency
    # non-trivial ones follow it; components too small to supply k of them stay zero-padded.
    num_components, labels = connected_components(adj.numpy(), directed=False)
    for label in range(num_components):
        component = torch.from_numpy(labels == label)
        component_adj = adj[component][:, component]
        laplacian = torch.diag(component_adj.sum(dim=1)) - component_adj
        component_eig_vec = torch.linalg.eigh(laplacian)[1][:, 1:(k + 1)]
        eig_vec[component, :component_eig_vec.shape[1]] = component_eig_vec

    graph.eig_vec = eig_vec

    return graph


def add_eig_vecs(dataset: InMemoryDataset, k=1, cache_name=None):
    all_graphs = [dataset.get(i) for i in range(len(dataset))]
    num_nodes = [graph.num_nodes for graph in all_graphs]

    # The field is a deterministic function of the sorted connectivity, so it is computed once
    # and cached in the dataset's processed directory: a ranking run starts one process per
    # config and would otherwise repeat the eigendecomposition on every start. The cache is
    # keyed by a digest of that connectivity, so a revised canonical sort — which permutes the
    # nodes without changing any shape — invalidates the cache instead of silently misaligning
    # the field with the nodes.
    digest = hashlib.blake2b(digest_size=16)
    for graph in all_graphs:
        digest.update(graph.edge_index.numpy().tobytes())

    cache_path = None if cache_name is None else \
        os.path.join(dataset.processed_dir, f'{cache_name}_eig_vec_k{k}.pt')
    cache = torch.load(cache_path) if cache_path is not None and os.path.exists(cache_path) else None
    eig_vecs = cache['eig_vec'] if cache is not None and cache['digest'] == digest.hexdigest() else None

    if eig_vecs is None or eig_vecs.shape != (sum(num_nodes), k):
        # `eigh` on graph-sized matrices is dominated by BLAS thread-launch overhead, so the loop
        # runs single-threaded: it is far faster in wall clock than letting each tiny
        # decomposition fan out, and it leaves the other jobs sharing this machine their cores.
        num_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        eig_vecs = torch.cat([add_eig_vec(graph, k).eig_vec for graph in all_graphs])
        torch.set_num_threads(num_threads)

        if cache_path is not None:
            torch.save({'digest': digest.hexdigest(), 'eig_vec': eig_vecs}, cache_path)

    for graph, eig_vec in zip(all_graphs, eig_vecs.split(num_nodes)):
        graph.eig_vec = eig_vec

    dataset.data, dataset.slices = dataset.collate(all_graphs)
    return dataset


def degree_histogram(dataset: InMemoryDataset):
    # Compute the maximum in-degree in the training data.
    max_degree = -1
    for data in dataset:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        max_degree = max(max_degree, int(d.max()))

    # Compute the in-degree histogram tensor
    deg = torch.zeros(max_degree + 1, dtype=torch.long)
    for data in dataset:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())

    return deg
