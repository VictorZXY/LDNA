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
  (larger than run-to-run noise). **Non-trivial margin** = LDNA's mean beats #2's
  mean by more than the combined **standard error** (`SEM = std / sqrt(runs)`), i.e.
  `mean_LDNA - SEM_LDNA > mean_2nd + SEM_2nd` (respecting each metric's direction).
  Use SEM, not raw std: std measures run-to-run spread, SEM measures uncertainty of
  the *mean*, which is what a ranking claim rests on. Increase `runs` (e.g. 10 on the
  high-variance `ogbg-molhiv`) so the estimate is stable. As a **disclosed** fallback
  when spread is large, best-of-N may be used — but applied **symmetrically** to LDNA
  and every baseline, never LDNA-best vs. baseline-mean.

Secondary objective — **fairness by shared hyperparameters** (see § Tuning methodology).

Soft objective (preferred, **not required**) — **hyperparameter coherence across similar
tasks**: LDNA's tuned settings should stay close for datasets in the same task family
(e.g. the graph-level molecular datasets `ogbg-molhiv` / `ogbg-molpcba` / `ZINC`) rather
than diverging wildly per dataset. Nice to have and worth reporting; do not sacrifice the
primary ranking objective for it. See § Tuning methodology (3).

---

## Datasets

Status legend: `done` = wired in `utils/resolver.py` with configs; `partial` = code
exists but not tuned/run; `todo` = not implemented.

### 1. Graph-level prediction

| Dataset | Task | Metric | Dir | Evaluator | Status | Notes |
|---|---|---|---|---|---|---|
| `ogbg-molhiv` | binary classification | ROC-AUC | max | `OGBGraphPropPredEvaluator` | done | wired + configs; **re-run needed** (see note) |
| `ogbg-molpcba` | multi-task binary (128) | AP | max | `OGBGraphPropPredEvaluator` | partial | code + configs exist; not tuned/run |
| `ZINC` | graph regression | MAE | min | `ZINCEvaluator` | done | wired + configs; search uses `subset=True` (~10k), final training uses full ZINC; **re-run needed** |
| `MNISTSuperpixels` | 10-way classification | accuracy | max | `MNISTEvaluator` | done | wired + configs; not in `hyperparam_search.py` yet; **re-run needed** |
| `ogbg-code2` | AST subtoken prediction | F1 | max | `Code2Evaluator` (OGB F1) | done | **edge-less** (GIN family, no `edge_attr`); dedicated seq-head (5×5002) + sum-over-position CE loss + decode→F1 eval + train-split vocab pre-pass; skip lexsort (AST DFS order is canonical). `num_nodetypes=98`, `num_nodeattributes=10030` |

> **`ogbg-ppa` removed from scope** (2026-07-07): ppa has no node features, so the feature-based
> canonical sort cannot impose a **permutation-invariant** node order (identical features → the
> sort falls back to input order). LDNA *requires* a canonical order so its MLP aggregator is
> permutation invariant (the sort is a correctness requirement, **not** a performance booster —
> LDNA's gain comes from the aggregator). Giving ppa a well-defined order needs a bespoke rule;
> not worth it given the four graph-level datasets already cover binary / multi-task / regression /
> multi-class. `ogbg-code2` is kept: its AST nodes have a natural DFS order (permutation invariant),
> so LDNA works there by using that order directly (skip the feature lexsort).

> **Prior graph-level results are invalidated** by the recent shared architecture change
> (`readout: attention` + `residual` + unified encoder width). Every wired graph-level
> dataset (`ogbg-molhiv`, `ogbg-molpcba`, `ZINC`, `MNISTSuperpixels`) must be **(re-)run**
> under the current architecture; treat any earlier numbers as stale.

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
| `GNN-VPA` | done | `models/vpa.py` (`VPA`): GIN/GINE backbone with PyG built-in `VariancePreservingAggregation` (`sum/√N`) via `aggr=`; dual-path (edge datasets→GINEConv, edge-less→GINConv); shared interface. Resolver query `GNN-VPA` |

---

## Tuning methodology

The evaluation is designed for **fair comparison via shared hyperparameters**:

1. **Tune LDNA only.** Run the hyperparameter search on LDNA for each dataset.
2. **Reuse LDNA's tuned config for every baseline on that dataset**, verbatim for
   the shared knobs (`num_layers`, `batch_size`, `lr`, `weight_decay`, `dropout`,
   `hidden_channels`, `readout`, `residual`, epochs, optimizer/scheduler). Baselines are
   **not** separately tuned. Model-internal-only settings that have no LDNA counterpart
   use each model's defaults.

   `readout` and `residual` are **shared** knobs — every model family supports them, so
   a value like `readout: attention` / `residual: true` is set identically for LDNA and
   all baselines. `readout` includes `attention` (`AttentionalAggregation`); because the
   canonical sort is applied on the data side, even the sort-exploiting `gru` readout is
   available to every model. `residual` is implemented with **projection skips** — a
   learnable `Linear` where a layer changes width, identity where it does not. The
   resolver maps every model's node encoder to the model width
   (`embedding_dim = hidden_channels`, unified across all models), so the conv stack is
   uniform (`hidden -> hidden`) and the skips are plain identities; the projection is only
   a safety net that engages if a config sets `out_channels ≠ hidden_channels`. Baselines
   whose own paper defines a residual scheme (e.g. `DeeperGCN`'s `res+`, which already
   relied on this unified width) keep it and ignore the shared `residual` flag.
3. **Prefer similar hyperparameters across datasets.** Aim for one config that works
   everywhere; if that is not achievable, keep configs similar within a task family
   (graph-level / node-level / expressiveness). This is a preference, **not** a hard
   requirement — record deviations and why.

The point is that any LDNA win must come from the method, not from LDNA getting a
better-tuned config than the baselines.

---

## Escalation: when tuning is not enough

"Iteratively optimize LDNA" means **escalate**, not just re-search. The loop per
dataset:

1. **Tune.** Run the Optuna search; take the best config.
2. **Rank.** Run `train.py` (multi-run) for LDNA + all baselines on that config;
   compare by the § Objective margin.
3. **If LDNA is not top-2**, the cause is almost certainly *not* the search space —
   change the **LDNA architecture non-trivially**, then go back to step 1.

Note first: `readout` and `residual` are **shared** knobs (§ Tuning methodology), not
LDNA-only levers — changing them changes every model, so they cannot by themselves
explain an LDNA win. A genuine LDNA advantage must come from LDNA-specific structure:

- **`LDNAConv` internals** — message-MLP depth/width, the learnable linear map, the
  per-node sum, the post-MLP, normalization placement.
- **Sort-exploiting mechanisms unique to LDNA** — structure that uses the canonical
  node order more directly than a shared readout does.
- **Depth/width knobs already exposed** — `num_pre_layers`, `num_post_layers`,
  `num_pred_layers`.

Invariants that must hold through every escalation (else the comparison is unfair or
the method is no longer LDNA):

- The **canonical-sort premise** stays (`utils/_utils.py`).
- The **shared interface** `forward(x, edge_index, edge_attr, batch)` /
  `reset_parameters()` stays, so the resolver and the baseline comparison keep working.
- **Dataset splits/preprocessing stay**, and **baselines are never modified** to
  flatter LDNA.

Stop the escalation loop for a dataset after **3 consecutive architecture changes**
that still fail to reach top-2 — stop and report instead of looping.

---

## Search spaces

LDNA search space in `hyperparam_search.py::objective` (all graph-level datasets —
`ogbg-molhiv`, `ogbg-molpcba`, `ZINC`, `MNISTSuperpixels`, `ogbg-code2`):

| Hyperparameter | Range / choices | Notes |
|---|---|---|
| `hidden_channels` | {128, 256, 512, 1024} | categorical |
| `num_layers` | [2, 8] | int (capped for stability — deep configs diverge sensitive baselines, see § stability below) |
| `dropout` | [0.1, 0.7] | float (sub-0.1 dropout is not meaningful) |
| `lr` | [1e-5, 1e-2] | log-uniform |
| `weight_decay` | [1e-6, 1e-3] | log-uniform |
| `batch_size` | not searched — per-dataset, power of 2 | implemented (see batch-size policy below) |
| `epochs` | search cap **50**; final **150** | search early-stops at plateau (`patience=20`, `min_delta=1e-3`); final runs the full 150 |
| `n_trials` | **100** (pruning enabled) | CLI default; TPESampler, seed 42 |

**batch-size policy.** `batch_size` is **not searched** — it entangles with `lr`
(linear-scaling rule), so searching both wastes budget and muddies lr comparability. It is
a fixed **per-dataset** value, **shared across all models** on that dataset (fairness
holds). Prefer a **power of 2** sized so `steps/epoch = train_size / batch_size` stays
~100–300: a universal 512 starves small datasets (ZINC `subset` ≈ 10k → ~20 steps/epoch)
while being fine for large ones. Values now in `hyperparam_search.py` (search) and the
configs (final):

| Dataset | ~train graphs | batch_size | steps/epoch |
|---|---|---|---|
| `ogbg-molhiv` | 33k | 256 | ~128 |
| `ogbg-molpcba` | 350k | 512 | ~684 |
| `ZINC` (subset, search) | 10k | 128 | ~78 |
| `ZINC` (full, final) | 220k | 512 | ~430 |
| `MNISTSuperpixels` | 60k | 256 | ~234 |

**Epochs.** Final (ranking) runs all use the **same `epochs` (150)** — kept uniform so
models are comparable. The search uses a much lower **cap of 50** to bound tuning time,
plus **early stopping on plateau**: a trial ends once the validation metric has not
improved by at least **`min_delta` (default `1e-3`)** for **`patience=20`** epochs. The
`min_delta` guard is essential — without it, a slow monotonic climb of sub-noise upticks
would reset the patience counter every epoch and the trial would run to the cap; with it,
only improvements that clear `1e-3` over the running best reset the counter, so true
saturation is caught. `patience=20` is **twice** the `ReduceLROnPlateau` patience (10):
the scheduler drops the LR at the first plateau and the resulting post-drop improvement is
captured before the trial stops, so the search scores configs on their **LR-annealed**
performance — the same regime the 150-epoch final runs use. (At `patience=10` the trial
would stop at the first plateau, *before* any LR drop, and could mis-rank configs that
only shine after annealing.) Post-saturation epochs don't change the trial score (median
of the last 5), so this is free savings on top of pruning. (`min_delta` is an absolute
threshold suited to the AUC/AP/MAE scales; raise it if trials still run to the cap.)

**Pruning is enabled.** `MedianPruner` (`n_warmup_steps=10`) + `trial.report` /
`should_prune` are active in `hyperparam_search.py`. Pruning and early stopping are
complementary: pruning kills *bad* trials early relative to others; early stopping ends
*any* trial once it saturates. Together, **`n_trials` defaults to 100** at roughly the
wall-clock of 50 un-pruned full-length trials.

**Gradient clipping & stability.** All training loops — search and final, in both
`train.py` and `hyperparam_search.py` (including the `ogbg-code2` forks) — clip gradients
to **`max_norm=1.0`** before `optimizer.step()`. This is a global stabilizer applied to
every model. Even so, some baselines (notably **`EGC`**) diverge to NaN at **deep** configs
(`num_layers ≳ 9`): the NaN originates in the forward/loss, so clipping — which only
rescales *finite* gradients — cannot rescue it. Hence `num_layers` is capped at **8**, and
`num_layers` is the primary instability driver (prefer the shallower of two otherwise
comparable configs). If a baseline **still** diverges at LDNA's tuned shared config, that
config is treated as **invalid** (§ Tuning methodology): re-tune, or fall back to the best
config within the settled ranges that keeps every model finite. This keeps the shared-config
comparison fair — a config that only LDNA can survive is not a valid basis for a ranking claim.

Knobs not yet in the search space (candidates to add). Shared across all models (a
searched value would apply to LDNA and every baseline): `readout`
(`add`/`max`/`mean`/`gru`/`attention`), `residual`, `batch_norm`, `act`. LDNA-specific:
`num_pre_layers`, `num_post_layers`, `num_pred_layers`.

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
- "Non-trivial margin": SEM-based; defined in § Objective.
- **Run logs:** every `train.py` run tees stdout+stderr to `logs/<experiment_name>.txt`
  (built into `train.py`); the final numbers are also pickled to
  `logs/<experiment_name>_logger.pickle`.

---

## Compute & parallelism

- The machine has **4 GPUs** (indices `0`–`3`, RTX A6000 49GB, no MIG). Training pins
  to one GPU via the config `cuda` field (`train.py`: `cuda: <idx>`, `-1` = CPU). Each
  run is one process.
- **Run many experiments concurrently** — one process per config — to saturate the
  GPUs. Grid over {datasets × models} maps naturally onto separate configs.
- **GPU policy** (before launching, always check `nvidia-smi`):
  - **Prefer idle GPUs first**, then **aim to keep all 4 GPUs busy** with our work.
  - Sharing is allowed: you may run **multiple of our own jobs on one GPU**, and you may
    **co-locate on a GPU another user is using** (compute is time-sliced — both sides
    slow down, but that is acceptable here).
  - **The one hard constraint: never OOM anyone** (ours or theirs). Keep the combined
    VRAM on each card within its 49GB — size batch/hidden so each process's peak fits the
    free headroom, release cache between search trials (`torch.cuda.empty_cache()`), make
    searches OOM-tolerant (`study.optimize(catch=(RuntimeError,))`), and watch per-card
    free memory (a safety monitor alerts if any card's free VRAM drops into the danger
    zone). If a card gets tight, back our jobs off rather than risk a collision.
- Scaling is done **through configs only** — no code changes needed to parallelize.
- **Second machine (`ee-tiamat`, H100 NVL 95GB).** A remote H100 is available for the
  heaviest datasets (`ogbg-molpcba`, `ogbg-code2`). Drive it via `ssh ee-tiamat`; the repo
  (`~/Projects/LDNA`) and conda env (`LDNA`) mirror this host, and code syncs through GitHub
  (push here → `git pull` there). Results (`out/logs/`, `logs/`) are gitignored, so copy them
  back with `scp` and record numbers in § Results log here. Exact SSH/job commands are in
  the two-machine setup memory (`.claude/project_state.md` § Next step has the summary).

---

## Guardrails

Mirror of `AGENTS.md` (§ Autonomous optimization); concrete limits here.

- Compute budget per optimization session: **100 Optuna trials/dataset** (the default).
  Three compute savers in the search: a low **epoch cap of 50** (vs 150 for final runs),
  `MedianPruner` (`n_warmup_steps=10`) that kills *bad* trials early relative to others,
  and **early stopping** (`patience=20`, `min_delta=1e-3`) that ends *any* trial once its
  validation metric saturates. `batch_size` is a fixed per-dataset power of 2, not
  searched (see § Search spaces). Set a wall-clock ceiling per search and stop if exceeded.
- **Always off-limits** (every phase): the **canonical-sort premise**
  (`utils/_utils.py`); the **shared model interface**
  `forward(x, edge_index, edge_attr, batch)` / `reset_parameters()`; **dataset
  splits/preprocessing**; **baseline model code**.
- **LDNA internal architecture** is off-limits **during tuning**, but IS in-scope as
  an **escalation** step once tuning cannot reach top-2 on a dataset — see
  § Escalation for the levers and invariants.
- **Implementing** new datasets, baselines, or node-level task support IS a sanctioned
  code change (distinct from tuning-time edits) — but keep the shared model interface
  and the canonical-sort design intact, and prefer minimal diffs.
- GPU policy above (§ Compute & parallelism): prefer idle GPUs, aim to keep all 4 busy,
  sharing (multiple of our jobs per GPU, or co-locating with other users) is allowed —
  the one hard rule is **never OOM anyone**.
- Stop-and-report conditions (any one triggers stop + report):
  1. **Target reached** — LDNA is top-2 on the dataset (ideally #1 with a non-trivial
     SEM margin per § Objective).
  2. **Search stalled** — Optuna best value not improved for ~15–20 trials.
  3. **Escalation exhausted** — 3 consecutive LDNA architecture changes still fail to
     reach top-2 (see § Escalation).
  4. **Budget exhausted** — trial or wall-clock ceiling hit.
  5. **Failure** — OOM / NaN / divergence.

---

## Implementation queue

Code to write, ordered easiest-first (graph-level reuses the existing pipeline;
node-level needs new infrastructure; expressiveness needs custom evaluation).

**Graph-level (do these first, in order):**

- [x] Implement `GNN-VPA` baseline wrapper in `models/` (`models/vpa.py`; shared
      `forward(x, edge_index, edge_attr, batch)` / `reset_parameters()` interface; available to all datasets).
- [x] Add `ogbg-code2` to the pipeline (graph-level; **edge-less/GIN**, no `edge_attr`): dedicated
      seq-head (5×5002) + sum-over-position CE loss + decode→F1 eval + train-split vocab pre-pass +
      `ASTNodeEncoder` (type/attr/depth); **skip lexsort** (AST DFS order). Merged `aade777`
      (`utils/code2.py`, `models/{ast_encoder,code2_head}.py`, guarded `train.py`/`hyperparam_search.py` fork).
- [x] Extend `hyperparam_search.py` to all graph-level datasets — `MNISTSuperpixels` **done**
      (search also arch-aligned to `readout: attention` + `residual: true`); `ogbg-code2` **done**
      (F1 evaluator, maximize, code2 loss/eval loop, batch_size 128).
- [ ] **Run the graph-level experiment protocol** (see § Experiment queue) on every
      graph-level dataset — do this after GNN-VPA and the items above are in place.

**Node-level (needs new infrastructure):**

- [ ] **Node-level infrastructure**: resolver branches for node datasets; a node-mode
      LDNA/baseline path (no graph pooling — per-node prediction head); transductive
      training loop with train/val/test masks; neighbor sampling for large graphs.
- [ ] Add `CitationFull` (Cora, PubMed) — smallest node datasets; define splits.
- [ ] Add `Reddit` (neighbor sampling).
- [ ] Add `ogbn-products` (neighbor sampling; OGB evaluator).
- [ ] Add `ogbn-proteins` (edge-feature aggregation to nodes; ROC-AUC).

**Expressiveness (custom evaluation):**

- [ ] Add `EXP` / `CEXP` (port dataset + splits; accuracy).
- [ ] Add `BREC` (dataset + BREC pairwise-distinguishing evaluation).
- [ ] Extend `hyperparam_search.py` and the experiment protocol to node-level and
      expressiveness datasets as their pipelines land.

---

## Experiment queue

Runs to do once the code is in place. Keep status current; move finished items to
the results log.

**Graph-level protocol.** For **each** dataset in § Datasets → Graph-level prediction,
run these steps independently — **do not reuse any past hyperparameters** (they are
invalid after the architecture change):

1. **Tune LDNA from scratch** on the dataset with `hyperparam_search.py`.
2. **Broadcast** the tuned shared config verbatim to all baselines via `broadcast.py`
   (`python broadcast.py --dataset <D> --hidden_channels .. --num_layers .. --dropout ..
   --lr .. --weight_decay .. --gpus 0,1,2,3`). It overrides only the shared knobs in each
   model's config (model-internal settings preserved), creates the `GNN-VPA` config, and
   round-robins `cuda`. Baseline set per § Tuning methodology + the GIN-xor-GINE rule:
   GCN, **{GINE if edge else GIN}**, GraphSAGE, GAT, GATv2, PNA, EGC, DeeperGCN
   (softmax+powermean), GNN-VPA — plus LDNA.
3. **Run** LDNA + all baselines on the dataset with `train.py` (multi-run).
4. **Rank** LDNA vs. baselines by the § Objective margin and append to § Results log.

- [ ] `ogbg-molhiv`
- [ ] `ogbg-molpcba` — **search capped** at best-value plateau or ~25-30 trials (whichever
      first; user 2026-07-07), then broadcast best-so-far. **Ranking stays full** (11 models
      × 3 runs × 150 epochs — no reduction). 350k graphs → the compute outlier.
- [ ] `ZINC`
- [ ] `MNISTSuperpixels`
- [ ] `ogbg-code2` (edge-less pipeline landed; run after a GPU frees)

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
- PyTorch Geometric: https://pytorch-geometric.readthedocs.io/en/stable/
- EXP / CEXP (GNN-RNI): https://arxiv.org/pdf/2010.01179 · https://github.com/ralphabb/GNN-RNI
- BREC: https://arxiv.org/pdf/2304.07702 · https://github.com/GraphPKU/BREC
- GNN-VPA: https://arxiv.org/abs/2403.04747 · https://github.com/ml-jku/GNN-VPA
