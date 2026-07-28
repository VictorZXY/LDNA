import networkx as nx
import numpy as np
import torch
import torch_geometric
from torch.nn import CosineEmbeddingLoss
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils.convert import from_networkx

from utils import canonicalize_graph

# BREC RPC (Reliable Paired Comparison) protocol constants, kept verbatim from the reference
# `base/test_BREC.py` so this evaluator reproduces the published test exactly.
NUM_RELABEL = 32
P_NORM = 2
OUTPUT_DIM = 16
EPSILON_MATRIX = 1e-7
EPSILON_CMP = 1e-6
SAMPLE_NUM = 400
EPOCH = 20
MARGIN = 0.0
LEARNING_RATE = 1e-4
THRESHOLD = 72.34
BATCH_SIZE = 16
WEIGHT_DECAY = 1e-4
LOSS_THRESHOLD = 0.2
SEED = 2023

# The default 10-seed rerun used by the reference `base/test_BREC_search.py`.
SEEDS = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]

# part_dict: {graph generation type: pair-id range}, verbatim from the reference.
PART_DICT = {
    "Basic": (0, 60),
    "Regular": (60, 160),
    "Extension": (160, 260),
    "CFI": (260, 360),
    "4-Vertex_Condition": (360, 380),
    "Distance_Regular": (380, 400),
}


class BRECDataset(InMemoryDataset):
    """The BREC expressiveness benchmark as an ``InMemoryDataset``.

    ``brec_v3.npy`` holds 51,200 graph6 byte-strings = 800 blocks x 64 graphs. Blocks 0..399 are
    the comparison pairs (each block interleaves 32 permuted copies of G1 with 32 of G2,
    ``[G1, G2, G1, G2, ...]``); blocks 400..799 are the matching reliability blocks (64 permuted
    copies of one graph). The graphs are featureless, so ``x`` is set to a constant column that the
    ``nn.Linear(1, emb)`` node encoder maps to the model width, and every graph is put into a nauty
    canonical node order via ``canonicalize_graph`` — the repo's permutation-invariance device,
    applied identically for every model (LDNA reads the order via ``rank_mode='position'``).
    """

    def __init__(self, root='data/brec', transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['brec_v3.npy']

    @property
    def processed_file_names(self):
        return ['brec_v3.pt']

    def process(self):
        data_list = np.load(self.raw_paths[0], allow_pickle=True)

        graphs = []
        for graph6 in data_list:
            graph = from_networkx(nx.from_graph6_bytes(graph6))
            # Featureless graphs: a constant column matches the MNIST-style `nn.Linear(1, emb)`
            # encoder path. canonicalize_graph then imposes the individualizing canonical order.
            graph.x = torch.ones([graph.num_nodes, 1], dtype=torch.float)
            graphs.append(canonicalize_graph(graph))

        if self.pre_filter is not None:
            graphs = [g for g in graphs if self.pre_filter(g)]
        if self.pre_transform is not None:
            graphs = [self.pre_transform(g) for g in graphs]

        self.save(graphs, self.processed_paths[0])


def _t2_calculation(model, dataset, device, s_epsilon=True):
    # Hotelling T^2 on a 64-copy block: X = even-indexed embeddings, Y = odd-indexed, D = X - Y,
    # S = cov(D), T^2 = D_mean^T . pinv(S) . D_mean. Ported from the reference `T2_calculation`.
    with torch.no_grad():
        loader = DataLoader(dataset, batch_size=BATCH_SIZE)
        pred_0_list = []
        pred_1_list = []
        for data in loader:
            pred = model(data.to(device)).detach()
            pred_0_list.extend(pred[0::2])
            pred_1_list.extend(pred[1::2])
        X = torch.cat([x.reshape(1, -1) for x in pred_0_list], dim=0).T
        Y = torch.cat([x.reshape(1, -1) for x in pred_1_list], dim=0).T
        D = X - Y
        D_mean = torch.mean(D, dim=1).reshape(-1, 1)
        S = torch.cov(D)
        if s_epsilon:
            # The canonical node order collapses the permuted copies within each class onto the
            # identical labelled graph, so a deterministic model emits identical embeddings within
            # a class and the within-class covariance S is ~0 — the vanilla pinv(S) is then
            # numerically meaningless. The BREC authors left this S_epsilon variant commented in
            # their own code for exactly the "same graph without permutation" case: adding
            # EPSILON_MATRIX * I degenerates the test gracefully, so distinct class embeddings give a
            # huge T^2, identical ones give ~0, and the reliability block (one graph) gives ~0.
            S = S + EPSILON_MATRIX * torch.eye(S.size(0), device=S.device, dtype=S.dtype)
        inv_S = torch.linalg.pinv(S)
        return torch.mm(torch.mm(D_mean.T, inv_S), D_mean)


def evaluate(model, dataset, device, seed=SEED, s_epsilon=True, pair_limit=0):
    """Run the BREC RPC protocol for every pair and return per-category / total statistics.

    Faithful port of the reference `base/test_BREC.py` evaluation loop, adapted to a model whose
    `forward(data)` returns a `[batch, OUTPUT_DIM]` embedding and which exposes `reset_parameters()`:
    reset per pair -> contrastive train (CosineEmbeddingLoss target -1 on even/odd embeddings, Adam,
    ReduceLROnPlateau, <=20 epochs, early stop at loss < 0.2) -> Hotelling T^2 on the 64-copy
    traintest block and on the reliability block -> distinguished iff T^2_traintest > THRESHOLD and
    T^2_traintest not close to T^2_reliability; reliability failure iff T^2_reliability >= THRESHOLD.
    `pair_limit > 0` caps the evaluated pair ids (for smoke/debug); `0` runs all `SAMPLE_NUM`.
    """
    torch_geometric.seed_everything(seed)
    torch.backends.cudnn.deterministic = True

    model = model.to(device)
    loss_func = CosineEmbeddingLoss(margin=MARGIN)

    cnt = 0
    fail_in_reliability = 0
    correct_list = []
    per_category = {}

    for part_name, (part_start, part_end) in PART_DICT.items():
        if pair_limit:
            part_end = min(part_end, pair_limit)
        if part_end <= part_start:
            continue

        cnt_part = 0
        fail_part = 0
        for pair_id in range(part_start, part_end):
            model.reset_parameters()
            optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
            dataset_traintest = dataset[pair_id * NUM_RELABEL * 2:(pair_id + 1) * NUM_RELABEL * 2]
            dataset_reliability = dataset[(pair_id + SAMPLE_NUM) * NUM_RELABEL * 2:
                                          (pair_id + SAMPLE_NUM + 1) * NUM_RELABEL * 2]

            model.train()
            for _ in range(EPOCH):
                traintest_loader = DataLoader(dataset_traintest, batch_size=BATCH_SIZE)
                loss_all = 0
                for data in traintest_loader:
                    optimizer.zero_grad()
                    pred = model(data.to(device))
                    loss = loss_func(pred[0::2], pred[1::2],
                                     torch.tensor([-1] * (len(pred) // 2)).to(device))
                    loss.backward()
                    optimizer.step()
                    loss_all += len(pred) / 2 * loss.item()
                loss_all /= NUM_RELABEL
                if loss_all < LOSS_THRESHOLD:
                    break
                scheduler.step(loss_all)

            model.eval()
            T_square_traintest = _t2_calculation(model, dataset_traintest, device, s_epsilon)
            T_square_reliability = _t2_calculation(model, dataset_reliability, device, s_epsilon)

            isomorphic_flag = False
            reliability_flag = False
            if T_square_traintest > THRESHOLD and not torch.isclose(
                T_square_traintest, T_square_reliability, atol=EPSILON_CMP
            ):
                isomorphic_flag = True
            if T_square_reliability < THRESHOLD:
                reliability_flag = True

            if isomorphic_flag:
                cnt += 1
                cnt_part += 1
                correct_list.append(pair_id)
            if not reliability_flag:
                fail_in_reliability += 1
                fail_part += 1

        total_part = part_end - part_start
        per_category[part_name] = {'correct': cnt_part, 'total': total_part, 'fail_reliability': fail_part}
        print(f"{part_name}: Correct {cnt_part} / {total_part}, "
              f"Fail in reliability {fail_part} / {total_part}")

    total = sum(v['total'] for v in per_category.values())
    print(f"Seed {seed}: Correct {cnt} / {total}, "
          f"Reliable-correct {cnt - fail_in_reliability} / {total}, "
          f"Fail in reliability {fail_in_reliability} / {total}")

    return {
        'seed': seed,
        'per_category': per_category,
        'correct': cnt,
        'reliable_correct': cnt - fail_in_reliability,
        'fail_reliability': fail_in_reliability,
        'total': total,
        'correct_list': correct_list,
    }
