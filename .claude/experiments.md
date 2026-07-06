# Experiments & Tuning Playbook

Single source of truth for autonomously optimizing LDNA. The standing rules for
autonomous work live in `AGENTS.md` (§ Autonomous optimization); this file holds
the concrete objectives, datasets, baselines, search spaces, queues, and results.

Fill in the `TODO` markers. Do not invent targets or experiments not listed here.
Metrics/evaluators for not-yet-implemented datasets are best-effort and must be
confirmed against the OGB / PyG docs (see References) when implementing.

---

## Objective

Primary objective — **relative ranking, not an absolute score**:

- On **every** dataset, LDNA must rank in the **top-2** across all baselines.
- Ideally LDNA is **#1**, and when it is, the gap to #2 should be **non-trivial**
  (larger than run-to-run noise). Quantify "non-trivial" per metric: TODO
  (suggested default — margin exceeds the mean ± std overlap across the configured runs).

Secondary objective — **fairness by shared hyperparameters** (see § Tuning methodology).

---

## Datasets

Status legend: `done` = wired in `utils/resolver.py` with configs; `partial` = code
exists but not tuned/run; `todo` = not implemented.

### 1. Graph-level prediction

| Dataset | Task | Metric | Dir | Evaluator | Status | Notes |
|---|---|---|---|---|---|---|
| `ogbg-molhiv` | binary classification | ROC-AUC | max | `OGBGraphPropPredEvaluator` | done | baseline dataset |
| `ogbg-molpcba` | multi-task binary (128) | AP | max | `OGBGraphPropPredEvaluator` | partial | code + configs exist; not tuned/run |
| `ogbg-ppa` | 37-way classification | accuracy | max | `OGBGraphPropPredEvaluator` | todo | edge features; no informative node features (use constant/degree) |
| `ogbg-code2` | AST subtoken prediction | F1 | max | `OGBGraphPropPredEvaluator` | todo | sequence-style target; confirm head + evaluator wiring |

### 2. Node-level prediction

Node-level tasks are **not yet supported by the pipeline** (see § Implementation
queue, node-level infrastructure). Loaders are from `torch_geometric.datasets` /
`ogb.nodeproppred`.

| Dataset | Task | Metric | Dir | Evaluator | Status | Notes |
|---|---|---|---|---|---|---|
| `CitationFull` (Cora, PubMed only) | node classification | accuracy | max | custom (acc) | todo | transductive single graph; CitationFull has no standard split — define train/val/test masks |
| `Reddit` | node classification | accuracy / micro-F1 | max | custom (acc) | todo | large graph (~232k nodes) — needs neighbor sampling |
| `ogbn-products` | node classification | accuracy | max | `OGBNodePropPredEvaluator` | todo | large graph — needs neighbor sampling; OGB split |
| `ogbn-proteins` | 112-task binary | ROC-AUC | max | `OGBNodePropPredEvaluator` | todo | no node features (aggregate edge features to nodes); species split |

### 3. Synthetic expressiveness

| Dataset | Task | Metric | Dir | Evaluator | Status | Notes |
|---|---|---|---|---|---|---|
| `EXP` / `CEXP` | distinguish 1-WL-equivalent pairs | accuracy | max | custom (acc) | todo | from GNN-RNI; port dataset + splits from source repo |
| `BREC` | pairwise graph distinguishing | # pairs distinguished (RPC protocol) | max | custom (BREC protocol) | todo | non-standard eval — reuse BREC's own comparison procedure, not a plain train/val/test loop |

---

## Baselines

Every dataset is evaluated with LDNA plus all baselines. Status as above.

| Model | Status | Notes |
|---|---|---|
| `GCN`, `GIN`, `GINE`, `GAT`, `GATv2`, `PNA`, `EGC`, `DeeperGCN` | done | existing wrappers in `models/` |
| `GraphSAGE` | partial | code exists (`models/sage.py`), configs exist; not run yet |
| `GNN-VPA` | todo | variance-preserving aggregation; implement as a `models/` wrapper matching the shared `forward(x, edge_index, edge_attr, batch)` / `reset_parameters()` interface |

---

## Tuning methodology

The evaluation is designed for **fair comparison via shared hyperparameters**:

1. **Tune LDNA only.** Run the hyperparameter search on LDNA for each dataset.
2. **Reuse LDNA's tuned config for every baseline on that dataset**, verbatim for
   the shared knobs (`num_layers`, `batch_size`, `lr`, `weight_decay`, `dropout`,
   `hidden_channels`, epochs, optimizer/scheduler). Baselines are **not** separately tuned.
   Model-internal-only settings that have no LDNA counterpart use each model's defaults.
3. **Prefer similar hyperparameters across datasets.** Aim for one config that works
   everywhere; if that is not achievable, keep configs similar within a task family
   (graph-level / node-level / expressiveness). This is a preference, **not** a hard
   requirement — record deviations and why.

The point is that any LDNA win must come from the method, not from LDNA getting a
better-tuned config than the baselines.

---

## Search spaces

Baseline search space currently in `hyperparam_search.py::objective` (LDNA only;
`ogbg-molhiv`, `ogbg-molpcba`, `ZINC`):

| Hyperparameter | Range / choices | Notes |
|---|---|---|
| `hidden_channels` | {128, 256, 512, 1024} | categorical |
| `num_layers` | [2, 10] | int |
| `dropout` | [0.0, 0.7] | float |
| `lr` | [1e-5, 1e-2] | log-uniform |
| `weight_decay` | [1e-6, 1e-3] | log-uniform |
| `batch_size` | 512 | fixed in code |
| `epochs` | 50 | CLI default |
| `n_trials` | 50 | CLI default; TPESampler, seed 42 |

LDNA-specific knobs not yet in the search space (candidates to add): `num_pre_layers`,
`num_post_layers`, `num_pred_layers`, `readout` (`add`/`max`/`mean`/`gru`), `batch_norm`, `act`.

`hyperparam_search.py` must be **extended to accept all datasets in this playbook**
(currently only the three graph-level ones). Node-level and expressiveness search
depends on the pipeline changes in § Implementation queue. Per-dataset overrides go here: TODO.

---

## Evaluation protocol

- **Search:** `hyperparam_search.py` scores a trial by the median validation metric
  over the last 5 epochs (single run). Use for finding LDNA's config.
- **Final ranking:** `train.py` runs `train_args['runs']` independent runs and reports
  per-run best train/val/test via `utils/logger.py` (mean ± std). Rank LDNA vs. all
  baselines using the mean test metric (respecting each metric's direction).
- **Node-level & expressiveness** need their own evaluation paths (single-graph
  transductive accuracy; BREC's pairwise protocol). Define when implementing.
- "Non-trivial margin": TODO (see § Objective).

---

## Compute & parallelism

- The machine has **4 GPUs** (indices `0`–`3`). Training pins to one GPU via the
  config `cuda` field (`train.py`: `cuda: <idx>`, `-1` = CPU). Each run is one process.
- **Run many experiments concurrently** — one process per config, each on a distinct
  GPU — to saturate the available share. Grid over {datasets × models} maps naturally
  onto separate configs.
- **GPU etiquette:** before launching, check `nvidia-smi`; only use GPUs that are
  idle. If others are using some GPUs, use the free ones and leave theirs alone.
- Scaling is done **through configs only** — no code changes needed to parallelize.

---

## Guardrails

Mirror of `AGENTS.md` (§ Autonomous optimization); concrete limits here.

- Compute budget per optimization session: TODO (e.g. max Optuna trials, max GPU-hours, wall-clock).
- Off-limits **while tuning**: the canonical sort in `utils/_utils.py`; model
  architecture contracts; dataset splits/preprocessing. Tune via configs / search spaces.
- **Implementing** new datasets, baselines, or node-level task support IS a sanctioned
  code change (distinct from tuning-time edits) — but keep the shared model interface
  and the canonical-sort design intact, and prefer minimal diffs.
- GPU etiquette above is a hard rule: never preempt another user's GPU.
- Stop-and-report conditions: TODO (e.g. target ranking reached on a dataset; no
  improvement over N trials; budget exhausted; unexpected failure).

---

## Implementation queue

Code to write, ordered easiest-first (graph-level reuses the existing pipeline;
node-level needs new infrastructure; expressiveness needs custom evaluation).

- [ ] Run GraphSAGE on existing datasets (no new code; configs exist).
- [ ] Tune LDNA + run all baselines on `ogbg-molpcba` (code done; needs tuning + runs).
- [ ] Implement `GNN-VPA` baseline wrapper in `models/` (available to all datasets).
- [ ] Add `ogbg-ppa` (graph-level; constant/degree node features; OGB evaluator).
- [ ] Add `ogbg-code2` (graph-level; confirm target head + F1 evaluator).
- [ ] **Node-level infrastructure**: resolver branches for node datasets; a node-mode
      LDNA/baseline path (no graph pooling — per-node prediction head); transductive
      training loop with train/val/test masks; neighbor sampling for large graphs.
- [ ] Add `CitationFull` (Cora, PubMed) — smallest node datasets; define splits.
- [ ] Add `Reddit` (neighbor sampling).
- [ ] Add `ogbn-products` (neighbor sampling; OGB evaluator).
- [ ] Add `ogbn-proteins` (edge-feature aggregation to nodes; ROC-AUC).
- [ ] Add `EXP` / `CEXP` (port dataset + splits; accuracy).
- [ ] Add `BREC` (dataset + BREC pairwise-distinguishing evaluation).
- [ ] Extend `hyperparam_search.py` to every dataset above as its pipeline lands.

---

## Experiment queue

Runs to do once the code is in place. Keep status current; move finished items to
the results log. (Populate as implementation lands.)

- [ ] TODO

---

## Results log

Append one row per completed run/trial group. For each dataset, record LDNA's rank
vs. baselines and the margin to #2. Keep § Objective status in sync.

| Date | Dataset | Model | Config summary | Metric (val / test) | LDNA rank / margin | Notes |
|---|---|---|---|---|---|---|
| TODO | | | | | | |

---

## References

- OGB: https://ogb.stanford.edu
- torch_geometric: https://pytorch-geometric.readthedocs.io/en/latest/
- EXP / CEXP (GNN-RNI): https://arxiv.org/pdf/2010.01179 · https://github.com/ralphabb/GNN-RNI
- BREC: https://arxiv.org/pdf/2304.07702 · https://github.com/GraphPKU/BREC
- GNN-VPA: https://arxiv.org/abs/2403.04747 · https://github.com/ml-jku/GNN-VPA
