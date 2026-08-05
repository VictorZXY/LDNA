import hashlib
import os

import numpy as np
import torch
from scipy.sparse.csgraph import connected_components
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import degree, sort_edge_index, to_dense_adj

# Version of the canonical-labelling RULE, carried in the permutation cache's name and digest (see
# canonicalize_graphs). Bump it whenever canonicalize_graph's colouring construction changes or the
# pynauty version underneath it does.
CANON_RULE_VERSION = 'v1'


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
    # `pos` is a per-node attribute like `x`, so leaving it behind desynchronises it from the nodes
    # silently: it keeps its shape and every downstream check still passes while `pos[i]` describes
    # some other node. Nothing read `pos` until DGN's CIFAR10 directional field did, so this
    # permute is what makes that field legitimate; it also closes the same latent desync on
    # MNISTSuperpixels, which carries `pos` and never reads it.
    if graph.pos is not None:
        graph.pos = graph.pos[graph.new2old]
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


def canonicalize_graph(graph: Data, sort_y=False, new2old=None):
    # Individualizing canonical node order for datasets whose features cannot impose one (EXP/CEXP,
    # BREC: 1-WL-hard, essentially featureless — sort_graph's lexicographic sort ties every node).
    # A nauty canonical labelling relabels the graph into a fixed, reproducible order that is
    # identical for isomorphic graphs (exact permutation invariance) and, unlike the feature
    # lexsort, individualizes ties so every node receives a distinct position. LDNA reads the order
    # via rank_mode='position'. Bookkeeping mirrors sort_graph, so the two are interchangeable.
    #
    # A precomputed `new2old` skips the labelling itself, which is the expensive part (measured
    # 2.87 ms/graph on ogbg-ppa, against ~0.4 ms for the permute and re-sort below it); that is what
    # canonicalize_graphs' on-disk cache stores, and with one supplied pynauty is never imported.
    if new2old is None:
        import pynauty  # optional heavy dependency; only imported where a labelling is computed

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
        new2old = torch.as_tensor(canon).long()

    sorted_idx = new2old
    inv_sorted_idx = torch.empty_like(sorted_idx)
    inv_sorted_idx[sorted_idx] = torch.arange(sorted_idx.shape[0])

    graph.new2old = sorted_idx
    graph.old2new = inv_sorted_idx

    # Sort the nodes
    if graph.x is not None:
        graph.x = graph.x[graph.new2old]
    if sort_y:
        graph.y = graph.y[graph.new2old]
    # Same guarded permute sort_graph does, and for the same reason: `pos` is a per-node attribute,
    # so leaving it behind desynchronises it silently — the shape survives and every downstream
    # check passes while `pos[i]` describes some other node. Inert today, since no dataset that
    # takes this path carries `pos` (EXP/CEXP/BREC/ogbg-ppa), but the two orderings must not
    # disagree about what they realign.
    if graph.pos is not None:
        graph.pos = graph.pos[graph.new2old]
    graph.edge_index = graph.old2new[graph.edge_index]

    # Sort the edge indices and attributes
    if hasattr(graph, 'edge_attr'):
        graph.edge_index, graph.edge_attr = sort_edge_index(graph.edge_index, graph.edge_attr)
    else:
        graph.edge_index = sort_edge_index(graph.edge_index)

    return graph


def canonicalize_graphs(dataset: InMemoryDataset, sort_y=False, cache_name=None):
    all_graphs = [dataset.get(i) for i in range(len(dataset))]
    num_nodes = [graph.num_nodes for graph in all_graphs]

    # The labelling is a deterministic function of the graph, so it is computed once and cached in
    # the dataset's processed directory, following add_eig_vecs' sidecar contract below: a ranking
    # run starts one process per config and would otherwise repeat it on every start. Measured on
    # ogbg-ppa's 158,100 graphs: 2.87 ms/graph, so 7.6 minutes labelled against 2.6 minutes for the
    # digest, permute and re-collate a cache hit still pays — ~90 minutes saved over a twelve-config
    # ranking. What is cached is the PERMUTATION rather than the permuted graphs — 38.5M int64 is
    # 308 MB against the ~29 GiB the canonicalized store itself occupies, and old2new, edge_index
    # and edge_attr are all cheap functions of it.
    #
    # The digest is taken over the PRE-canonicalization graph, for two reasons. On a cache hit the
    # canonical order does not exist yet, so keying on it would mean doing the work the key exists
    # to avoid. And a canonical labelling is a function of the isomorphism class alone: two builds
    # of one dataset with different raw node orderings yield the IDENTICAL canonicalized edge_index
    # but need DIFFERENT permutations, so a post-canonical digest would validate a cache that no
    # longer matches the data on disk.
    #
    # Node count, connectivity AND features are hashed, because the labelling reads all three (x
    # becomes nauty's colour partition above). add_eig_vecs hashes connectivity alone, which is
    # correct there — its field ignores features — and would be a silent misalignment here.
    #
    # The rule version is hashed too, and named in the file. A pre-canonical digest is by
    # construction blind to a change in the labelling RULE (the colouring built above, or the
    # pynauty version underneath), so without it a stale cache would still validate. add_eig_vecs
    # needs no such tag because a revised sort changes the bytes it already hashes.
    digest = hashlib.blake2b(digest_size=16)
    digest.update(CANON_RULE_VERSION.encode())
    for graph in all_graphs:
        digest.update(str(graph.num_nodes).encode())
        digest.update(graph.edge_index.numpy().tobytes())
        if graph.x is not None:
            digest.update(graph.x.numpy().tobytes())

    cache_path = None if cache_name is None else \
        os.path.join(dataset.processed_dir, f'{cache_name}_canon_{CANON_RULE_VERSION}.pt')
    cache = torch.load(cache_path) if cache_path is not None and os.path.exists(cache_path) else None
    # As in add_eig_vecs: anything that is not the current dict format with a matching digest and
    # the right length falls through to recompute rather than raising, since the cache is a
    # discardable accelerator and not a source of truth.
    valid_cache = isinstance(cache, dict) and cache.get('digest') == digest.hexdigest() \
        and isinstance(cache.get('new2old'), torch.Tensor) \
        and cache['new2old'].shape == (sum(num_nodes),)
    perms = list(cache['new2old'].split(num_nodes)) if valid_cache else [None] * len(all_graphs)

    # Pinned to one thread for the same reason add_eig_vecs is, and the effect here is larger. Every
    # tensor operation in the loop is graph-sized — a gather over one edge_index, an argsort of a few
    # thousand entries — so letting torch fan out costs far more in thread-launch overhead than it
    # saves: measured on ogbg-ppa, unpinned the loop took 45 cores (4497% CPU over 191 threads) and
    # had not finished 158,100 graphs in 21 minutes, against 2 threads doing ~2.3 ms/graph. It also
    # keeps the loop from monopolizing a machine this project shares. A process pool is the
    # alternative and is deliberately not taken: it would need module-level globals and a
    # fork-safety guarantee this project has not established for pynauty.
    num_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    canon_graphs = [canonicalize_graph(graph, sort_y, perm)
                    for graph, perm in zip(all_graphs, perms)]
    torch.set_num_threads(num_threads)

    if not valid_cache and cache_path is not None:
        torch.save({'digest': digest.hexdigest(),
                    'new2old': torch.cat([graph.new2old for graph in canon_graphs])}, cache_path)

    # Release the source store BEFORE collating. Every per-node and per-edge tensor the loop above
    # produced is a fresh one, so the dataset itself is all that still references the old store —
    # including `_data_list`, whose cached per-graph views would otherwise pin it past the setter
    # that clears them. On EXP/CEXP that is bookkeeping; on ogbg-ppa the store is ~29 GiB and
    # holding it across the collate is the difference between a ~59 GiB and an ~88 GiB peak.
    dataset._data_list = None
    dataset._data = None

    dataset.data, dataset.slices = dataset.collate(canon_graphs)
    return dataset


def node_rank(graph: Data, rank_mode='feature'):
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

    return (nrank / n).unsqueeze(-1)


def add_node_ranks(dataset: InMemoryDataset, rank_mode='feature'):
    # Multi-graph node-level datasets (PPI): the rank is computed PER GRAPH, never once across the
    # collated store, since a rank shared across graphs would compare feature rows that no edge
    # connects. Follows add_eig_vecs' materialize -> mutate -> collate shape but keeps no disk
    # cache: a lexsort is cheaper to recompute than a cache key hashed over every feature row.
    all_graphs = [dataset.get(i) for i in range(len(dataset))]
    for graph in all_graphs:
        graph.nrank = node_rank(graph, rank_mode)

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

    # A directed edge_index makes `adj` asymmetric, and `eigh` assumes a Hermitian input and reads
    # only one triangle — so it returns the spectrum of a matrix it was never given, with no error
    # and no failed shape check. That cost this project a silently wrong DGN field on CIFAR10, whose
    # 8-NN graph leaves 20.05% of edges without a reverse; the fix there was to substitute the image
    # coordinates, the field the DGN paper itself uses on that dataset. Every remaining caller is
    # symmetric, so this check never fires today — which is exactly why it is worth having before
    # the next dataset arrives.
    if not bool((adj == adj.t()).all()):
        raise ValueError(f"The Laplacian eigenvector field needs an undirected graph: this one has "
                         f"{int((adj != adj.t()).sum()) // 2} edge(s) with no reverse. `eigh` would "
                         f"silently decompose the lower triangle mirrored onto itself, which is not "
                         f"this graph's Laplacian — supply a domain field instead, as CIFAR10 and "
                         f"ogbg-code2 do")

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
