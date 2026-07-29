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


def canonicalize_graph(graph: Data, sort_y=False):
    # Individualizing canonical node order for datasets whose features cannot impose one (EXP/CEXP,
    # BREC: 1-WL-hard, essentially featureless — sort_graph's lexicographic sort ties every node).
    # A nauty canonical labelling relabels the graph into a fixed, reproducible order that is
    # identical for isomorphic graphs (exact permutation invariance) and, unlike the feature
    # lexsort, individualizes ties so every node receives a distinct position. LDNA reads the order
    # via rank_mode='position'. Bookkeeping mirrors sort_graph, so the two are interchangeable.
    import pynauty  # optional heavy dependency; only imported where a canonical labelling is needed

    n = graph.num_nodes

    # nauty wants an undirected simple adjacency; PyG stores both directions, so symmetrize, drop
    # any self-loops, and dedupe into per-vertex neighbor lists.
    adjacency = {i: set() for i in range(n)}
    src, dst = graph.edge_index.tolist()
    for a, b in zip(src, dst):
        if a != b:
            adjacency[a].add(b)
            adjacency[b].add(a)
    adjacency = {i: sorted(neighbors) for i, neighbors in adjacency.items()}

    # Ordered color partition from the feature rows, so the labelling refines the features instead
    # of ignoring them. The partition order is significant to nauty, so the color classes are taken
    # in lexicographic feature-row order — itself permutation-invariant — which keeps the resulting
    # labelling isomorphism-invariant. No features (x is None) leaves the graph uncolored.
    vertex_coloring = []
    if graph.x is not None:
        rows = [tuple(row) for row in graph.x.tolist()]
        cells = {row: set() for row in rows}
        for node, row in enumerate(rows):
            cells[row].add(node)
        vertex_coloring = [cells[row] for row in sorted(cells)]

    # canon_label returns new2old (the old vertex placed at each canonical position; empirically
    # verified to be this direction, not its inverse), matching sort_graph's new2old.
    canon = pynauty.canon_label(pynauty.Graph(n, False, adjacency, vertex_coloring))
    sorted_idx = torch.as_tensor(canon).long()
    inv_sorted_idx = torch.empty_like(sorted_idx)
    inv_sorted_idx[sorted_idx] = torch.arange(sorted_idx.shape[0])

    graph.new2old = sorted_idx
    graph.old2new = inv_sorted_idx

    # Sort the nodes
    if graph.x is not None:
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


def canonicalize_graphs(dataset: InMemoryDataset, sort_y=False):
    all_graphs = [dataset.get(i) for i in range(len(dataset))]
    canon_graphs = [canonicalize_graph(g, sort_y) for g in all_graphs]

    dataset.data, dataset.slices = dataset.collate(canon_graphs)
    return dataset


def _wl_refine(graph: Data, colors, num_steps):
    # 1-WL color refinement of an initial integer node coloring, in original node order. Each step
    # replaces a node's color with an id of its signature: (own color, degree, the sorted multiset
    # of (neighbor color, edge label) over its incident edges), where the edge label is the
    # edge_attr row when the graph carries edge features (ZINC bond types) and omitted otherwise.
    # Because the signature contains the previous color, each step refines (never coarsens) the
    # input partition, so the final colors are themselves a refined dense rank of the initial one.
    #
    # New ids are assigned by LEXICOGRAPHIC SORT of the distinct signatures — a canonical
    # relabelling that depends only on signature content, never on hash values or first-appearance
    # order, which is what keeps the refinement permutation-invariant: nodes that remain
    # 1-WL-equivalent share a color at every step. Aggregation runs over incoming edges, the same
    # direction LDNAConv aggregates messages, which on the undirected (both-direction) datasets
    # used here is exactly the incident-edge multiset.
    n = graph.num_nodes
    src, dst = graph.edge_index.numpy()
    num_edges = src.shape[0]
    labels = graph.edge_attr.view(num_edges, -1).numpy() if graph.edge_attr is not None \
        else np.empty((num_edges, 0), dtype=np.int64)
    label_dim = labels.shape[1]

    # Signatures are laid out as fixed-width rows [own color, degree, pair_1, ..., pair_maxdeg],
    # padding unused pair slots with a sentinel so np.lexsort can order variable-length multisets.
    # The degree column already separates signatures of different length, so the sentinel value can
    # never conflate two distinct signatures.
    deg = np.bincount(dst, minlength=n)
    max_deg = int(deg.max()) if num_edges > 0 else 0
    ptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(deg, out=ptr[1:])
    sig_dtype = np.result_type(np.int64, labels.dtype)

    num_colors = int(colors.max()) + 1
    for _ in range(num_steps):
        if num_colors == n:  # discrete partition: nothing left to refine
            break

        # Group the edges by destination and sort each group by (neighbor color, edge label), so
        # every node's pair slots hold its sorted multiset. Ties within a group are content-equal
        # pairs, so how the sort breaks them cannot affect the signature.
        order = np.lexsort([labels[:, j] for j in reversed(range(label_dim))] + [colors[src], dst])
        slot = np.arange(num_edges) - ptr[dst[order]]

        sig = np.full((n, 2 + max_deg * (1 + label_dim)), -1, dtype=sig_dtype)
        sig[:, 0] = colors
        sig[:, 1] = deg
        col = 2 + slot * (1 + label_dim)
        sig[dst[order], col] = colors[src[order]]
        for j in range(label_dim):
            sig[dst[order], col + 1 + j] = labels[order, j]

        # Dense-rank the signature rows in lexicographic order — the same "compare adjacent rows +
        # cumsum" device as node_rank below, applied to signature rows instead of feature rows.
        row_order = np.lexsort(sig.T[::-1])
        sig = sig[row_order]
        changed = np.zeros(n, dtype=np.int64)
        changed[1:] = (sig[1:] != sig[:-1]).any(axis=1)
        new_colors = np.empty(n, dtype=np.int64)
        new_colors[row_order] = np.cumsum(changed)

        # A step can only split color classes, so an unchanged class count means the partition is
        # stable and further steps are no-ops.
        new_num_colors = int(new_colors.max()) + 1
        if new_num_colors == num_colors:
            break
        colors, num_colors = new_colors, new_num_colors

    return colors


def node_rank(graph: Data, rank_mode='feature', wl_steps=0):
    # Per-node canonical rank for node-level (transductive) data, returned as a float in [0, 1) of
    # shape [N, 1] — the same contract as LDNA._canonical_rank, which this reproduces for a single
    # graph. It is precomputed here and carried on the data because a node-level model sees the
    # whole graph (or a sampled subgraph) rather than a batch of complete graphs, so re-deriving the
    # rank inside the forward would rank a slice instead of the graph.
    #
    # Unlike sort_graph, this does NOT permute the graph. The tie-aware dense rank is permutation-
    # EQUIVARIANT: a node's rank depends only on its own feature row and the multiset of all feature
    # rows, so relabelling the nodes carries each rank along with its node. The gate reads only
    # delta = rank_j - rank_i on an edge, and aggregation is a scatter-sum, so both are invariant
    # under relabelling and the physical order is never observed. The sort in sort_graph exists only
    # to make _canonical_rank's O(N) "compare adjacent rows + cumsum" trick valid on a batch; here
    # the same trick runs in lexsort order and the result is scattered back to the original indices.
    if rank_mode != 'feature':
        raise ValueError(f"Argument `rank_mode` must be 'feature' for node-level data, got "
                         f"'{rank_mode}': a position rank presumes a canonical construction order, "
                         f"which a single transductive graph does not have")

    n = graph.num_nodes
    if n == 0:
        return graph.x.new_zeros((0, 1), dtype=torch.float)

    x_arr = graph.x.numpy()
    sorted_idx = np.lexsort([x_arr[:, i] for i in range(x_arr.shape[1])])
    sorted_idx = torch.from_numpy(sorted_idx).long()

    # Dense rank in lexicographic order: increment only where the ENTIRE feature row differs from
    # its predecessor, so nodes the sort cannot separate share a rank. That shared rank is what
    # makes the result independent of how the tie was broken, hence permutation-invariant.
    xf = graph.x[sorted_idx].view(n, -1)
    changed = torch.zeros(n, dtype=torch.bool)
    changed[1:] = (xf[1:] != xf[:-1]).any(dim=1)
    rank = torch.cumsum(changed.long(), dim=0).float()

    # Scatter back to the ORIGINAL node indices, then normalize by graph size to match
    # _canonical_rank's scale-free gap.
    nrank = torch.empty(n, dtype=torch.float)
    nrank[sorted_idx] = rank

    # Optional 1-WL refinement of the tie structure (wl_steps > 0): on a dataset whose single
    # feature column ties most nodes (ZINC atom types), most edges see a zero rank gap and the
    # gate reads no signal there. Refining the feature partition by neighborhood structure
    # recovers gaps between 1-WL-distinguishable nodes while keeping genuinely equivalent nodes
    # tied, so the permutation invariance argument above carries over (see _wl_refine).
    if wl_steps > 0:
        nrank = torch.from_numpy(_wl_refine(graph, nrank.numpy().astype(np.int64), wl_steps)).float()

    return (nrank / n).unsqueeze(-1)


def add_node_ranks(dataset: InMemoryDataset, rank_mode='feature', wl_steps=0):
    # Multi-graph node-level datasets (PPI): the rank is computed PER GRAPH, never once across the
    # collated store, since a rank shared across graphs would compare feature rows that no edge
    # connects. Follows add_eig_vecs' materialize -> mutate -> collate shape but keeps no disk
    # cache: a lexsort is cheaper to recompute than a cache key hashed over every feature row.
    all_graphs = [dataset.get(i) for i in range(len(dataset))]
    for graph in all_graphs:
        graph.nrank = node_rank(graph, rank_mode, wl_steps)

    dataset.data, dataset.slices = dataset.collate(all_graphs)
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
    # A cache is only usable if it is the current dict format with a matching digest; anything else
    # (a stale bare-tensor cache from an older format, a corrupt file) falls through to recompute
    # rather than raising, since the cache is a discardable accelerator, not a source of truth.
    valid_cache = isinstance(cache, dict) and cache.get('digest') == digest.hexdigest()
    eig_vecs = cache['eig_vec'] if valid_cache else None

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
