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
| `ogbg-molhiv` | binary classification | ROC-AUC | max | `OGBGraphPropPredEvaluator` | done | wired; configs; high run-to-run variance — use more runs (e.g. 10) |
| `ogbg-molpcba` | multi-task binary (128) | AP | max | `OGBGraphPropPredEvaluator` | done | wired; configs; largest (350k graphs) — search capped, ranking full (see § Experiment queue) |
| `ZINC` | graph regression | MAE | min | `ZINCEvaluator` | done | wired; configs; search uses `subset=True` (~10k), final training uses full ZINC |
| `MNISTSuperpixels` | 10-way classification | accuracy | max | `MNISTEvaluator` | done | wired; configs |
| `ogbg-code2` | AST subtoken prediction | F1 | max | `Code2Evaluator` (OGB F1) | done | **edge-less** (GIN family, no `edge_attr`); dedicated seq-head (5×5002) + sum-over-position CE loss + decode→F1 eval + train-split vocab pre-pass; skip lexsort (AST DFS order is canonical). `num_nodetypes=98`, `num_nodeattributes=10030` |

> **`ogbg-ppa` removed from scope** (2026-07-07): ppa has no node features, so the feature-based
> canonical sort cannot impose a **permutation-invariant** node order (identical features → the
> sort falls back to input order). LDNA *requires* a canonical order so its MLP aggregator is
> permutation invariant (the sort is a correctness requirement, **not** a performance booster —
> LDNA's gain comes from the aggregator). Giving ppa a well-defined order needs a bespoke rule;
> not worth it given the four graph-level datasets already cover binary / multi-task / regression /
> multi-class. `ogbg-code2` is kept: its AST nodes have a natural DFS order (permutation invariant),
> so LDNA works there by using that order directly (skip the feature lexsort).

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
| `EXP` / `CEXP` | distinguish 1-WL-equivalent pairs | accuracy | max | `AccuracyEvaluator` | done | ported from GNN-RNI (GPL v3 — note on redistribution); 600 pairs each, binary node-type feature, **edge-less** (GIN family); paper's contiguous 10-fold CV, folds map to `runs`; raw pickles under `data/EXP|CEXP/raw/` (gitignored, fetch from the GNN-RNI repo) |
| `BREC` | pairwise graph distinguishing | # pairs distinguished (RPC protocol) | max | RPC core in `utils/brec.py` | done | dedicated entry point `brec_eval.py` (the train/val/test loop does not fit the pairwise protocol); 400 pairs × 10 seeds; featureless (`x = ones`); `data/brec/raw/brec_v3.npy` from the BREC repo |

> **Batch-3 protocol decisions (2026-07-27).**
> - **Canonical order**: these graphs are featureless/uniform, so the feature lexsort cannot impose
>   an order. The permutation-invariance device is a **nauty canonical labelling**
>   (`utils/_utils.py: canonicalize_graph(s)`, pynauty, data-side, replaces `sort_graphs` on these
>   datasets); LDNA reads it via `rank_mode: position`. Isomorphic graphs get the identical
>   labelled form, so invariance is exact. Write-up caveat: nauty canonization is
>   isomorphism-complete — LDNA's beyond-1-WL signal on these benchmarks is the **combination** of
>   canonical individualization (the source) and the learned rank-gate aggregation (the mechanism).
>   Feature-mode LDNA is provably exactly 1-WL here and is not run.
> - **Model set: LDNA + GIN + DGN only.** Every other baseline is provably ≤ 1-WL (none claims
>   otherwise) and is recorded at the theoretical ceiling (EXP 50%, BREC 0/400) without burning
>   compute; GIN is the empirical 1-WL control; DGN is the one baseline whose paper claims beyond-
>   1-WL structure (spectral directional field). DGN keeps its default **Laplacian field** — the
>   synthetic graphs provide no domain field to substitute, and inventing one would inflate the
>   baseline. Its spectrum is degenerate on the regular BREC families (recorded as DGN's own
>   limitation); the canonical order incidentally makes the arbitrary eigenbasis deterministic.
> - **No hyperparameter search.** Hyperparameters are protocol constants — EXP: the GNN-RNI
>   protocol (Adam, constant lr 1e-3 via `min_lr == lr`, 500 epochs, batch 20, 8×64); BREC: the
>   benchmark's fixed constants (20 epochs/pair, lr 1e-4, batch 16, 16-dim embedding — the T²
>   threshold 72.34 is calibrated for them; backbone 4×64 per the reference). Tuning could not
>   change the structural result (1-WL models sit at the ceiling regardless) and would invite a
>   fairness objection. `hyperparam_search.py` is deliberately **not** extended to these datasets.
> - **BREC specifics**: canonicalization collapses each block's 32 permuted copies onto one
>   labelled graph, so every deterministic model has within-class covariance ≈ 0 and the vanilla
>   Hotelling T² is numerically meaningless — the **S + εI variant** (the BREC authors' own
>   commented-out alternative) is the default (`s_epsilon`). `batch_norm: true` (explicit in the
>   three `*_brec.yaml`, shared): adjudicated empirically after conflicting evidence. BN's
>   batch-mean subtraction does erase the between-graph mean signal on regular constant-feature
>   families (measured: BN-off distinguishes 1/4 SimpleReg pairs from untrained embeddings
>   alone), but with BN off LDNA's contrastive loss stays pinned at 1.0 in **every** family — it
>   never trains inside the protocol's 80-step budget. Category-weighted totals ≈124 (on) vs
>   ≈12 (off) / 400 → BN on by ~10×. GIN is 0 both ways; the reliability negative control is ≈0
>   throughout, so BN-on distinctions are not artifacts. Backbone is **64 hidden × 8 layers**
>   (shared), chosen by a pre-committed probe sweep over {64×4, 128×4, 64×8, 128×8} on 20 pairs
>   × 3 seeds (per-seed totals [3,3,4] / [3,2,2] / [5,5,5] / [3,4,3]; depth helps, extra width
>   hurts inside the step budget); DGN runs and GIN stays 0 at that size. Results go to
>   `out/logs/<name>_results.json` (the Logger's train/val/test shape does not fit).
> - **EXP deviations from the source protocol** (recorded, all shared across models): `attention`
>   readout per the repo-wide shared protocol (paper: max-pool); scheduler steps on val accuracy
>   with `mode: max` (paper: val loss / min — numerically inert since the LR is constant);
>   learnable `Encoder` embedding instead of one-hot; CrossEntropy ≡ NLL+log_softmax; the repo's
>   best-epoch reporting vs the paper's final-epoch (negligible on a task that saturates).

---

## Baselines

Every dataset is evaluated with LDNA plus all baselines. Status as above.

| Model | Status | Notes |
|---|---|---|
| `GCN`, `GIN`, `GINE`, `GraphSAGE`, `GAT`, `GATv2`, `PNA`, `EGC`, `DeeperGCN` | done | existing wrappers in `models/` (`GraphSAGE` = `models/sage.py`, resolver query `GraphSAGE`/`SAGE`) |
| `GNN-VPA` | done | `models/vpa.py` (`VPA`): GIN/GINE backbone with PyG built-in `VariancePreservingAggregation` (`sum/√N`) via `aggr=`; dual-path (edge datasets→GINEConv, edge-less→GINConv); shared interface. Resolver query `GNN-VPA` |
| `DGN` | done | `models/dgn.py` (`DGN`, `DGNConv`): PNA-shaped layer whose aggregator set adds the directional `dirK-av` / `dirK-dx`, weighted by the gradient of a per-node scalar field. No PyG built-in exists (`DirGNNConv` is an unrelated paper), so `DGNConv` is a custom `MessagePassing(aggr=None)` port of the official DGL implementation. Resolver query `DGN`. The field is the low-frequency Laplacian eigenvectors on the four molecular/image datasets and the AST depth on `ogbg-code2` — see the two notes below |

> **DGN's configs use the paper's ZINC aggregator set, not its MolHIV/PCBA one.** All four
> `dgn_*.yaml` set `aggregators: [mean, dir1-dx, dir1-av]` — the set the official repo ships for
> ZINC, and the best row on ZINC-simple / MolHIV-simple in the paper's fixed-budget ablation.
> The reason is comparison hygiene: DGN is PNA plus directionality, and the alternative set
> (`[mean, max, min, dir1-dx, dir1-av]`, which the code keeps as its default) overlaps this
> repo's `PNA` baseline (`[mean, min, max, std]`) in three of five aggregators, so the two
> baselines would largely re-measure each other. With `[mean, dir1-dx, dir1-av]` the overlap is
> `mean` alone — unavoidable, since the directional weights are a partition of unity (`av`) or
> sum to ~0 (`dx`), so a purely directional layer would drop the isotropic signal, a
> configuration the paper never runs. The degree `scalers` stay at all three: they are a
> per-node gain applied uniformly to whatever aggregators exist, orthogonal to which aggregators
> those are, and keeping them reproduces the official ZINC config exactly. Side effect: the
> narrower `post_nn` input (`(3*3+1)*hidden` instead of `(5*3+1)*hidden`) puts DGN at ~0.73x
> `PNA`'s parameter count and ~3.0x LDNA's, rather than ~1.05x PNA's.

> **DGN needs a precomputed eigenvector field.** `utils/_utils.py: add_eig_vecs` attaches a
> per-node `eig_vec` to the data, and `train.py` forwards it only when present — the resolver
> attaches it only for `DGN`, so its presence is the routing switch and the other baselines'
> call path is untouched. Two ordering constraints are load-bearing: the field is computed
> **after** `sort_graphs` (the canonical sort permutes nodes without realigning extra per-node
> attributes, so an earlier attachment desynchronises silently) and, for the OGB datasets,
> **before** the split slices (an index view cannot be re-collated). Eigenvectors are taken per
> connected component, as the paper prescribes: on a disconnected graph the whole-graph
> Laplacian's lowest non-trivial eigenvectors degenerate into component indicators and the
> directional field vanishes on every edge (7.5% of `ogbg-molhiv` graphs are disconnected, vs
> 0.2% of `ogbg-molpcba` and none of `ZINC`/`MNISTSuperpixels`). The field costs ~0.9 ms per graph
> to build and is **cached on disk** as a sidecar `<split>_eig_vec_k<k>.pt` in the dataset's
> `processed_dir`, so only the first run pays for it. The cache is keyed by a digest of the
> sorted connectivity, so a revised canonical sort invalidates it automatically — without that
> guard the field is silently reused against permuted nodes (verified: it survives the shape
> check and leaves the per-edge field an order of magnitude wrong, with no error). The sidecar
> deliberately does not rewrite the dataset's own processed cache, which every other model
> shares. The build loop is pinned to one thread: `eigh` on graph-sized matrices is
> dominated by BLAS thread-launch overhead, and letting it fan out was ~30x slower in wall clock
> while starving the machine's other jobs. This whole path applies to the four
> molecular/image datasets only; `ogbg-code2` uses the depth field below instead.

> **DGN on `ogbg-code2` uses the AST depth, not eigenvectors.** The paper substitutes a
> domain-provided field when eigenvectors are unusable (it used image coordinates for CIFAR10,
> whose grid symmetry makes λ₁ degenerate). Every code2 graph is a tree (measured: `num_edges ==
> num_nodes - 1` for all 452,741), so on the symmetrized edge set `|depth_j − depth_i| = 1`
> everywhere and the field's gradient is exactly the parent/child orientation — after
> `ToUndirectedNoAttr`, `dir1-dx` is the only aggregator in the code2 lineup that can tell a
> message from the parent apart from one from a child. `utils/transforms.py: AddDepthField`
> attaches it inside the existing lazy chain (a view and a cast, so no precompute and no cache),
> gated on `model_query == 'DGN'`; `Code2Head.forward` and the two code2 call sites in `train.py`
> forward it. The eigenvector alternative was measured and rejected: it is *feasible* (sparse
> shift-invert, ~13 min one-time) but needs four settings whose failure modes are all silent
> (`which='SA'` converges to the wrong eigenpair at the tolerances the official DGN code uses;
> `sigma=0` exactly makes `splu` fail; an unfixed ARPACK `v0` makes the field depend on how many
> solves preceded it; float32 loses precision above ~800 nodes), plus a full materialization of
> the one dataset the resolver never materializes — and it is not more faithful anyway, since
> λ₁ is degenerate on 0.8–1.3% of code2 graphs versus zero on the datasets DGN validated on
> (99.76% of ASTs have a nontrivial subtree-swap automorphism). Config: `[mean, dir1-dx]` —
> `dir1-av` is dropped because with a ±1 field it is provably the `mean` block (measured
> agreement to ~3 float32 ulps). `node_dfs_order` was rejected as a second field: it equals
> `arange(n)`, i.e. the positional signal only `LDNA` consumes via `rank_mode: position`, and no
> other baseline can see it. One direction is the right count for a tree — an AST has a single
> canonical axis, where CIFAR10's two came from an image having two coordinates.
>
> Three consequences of the substitution, recorded as decisions rather than left implicit.
> (1) **`dir-dx` keeps its `.abs()`**, although the reason for that absolute value — folding away
> an eigenvector's arbitrary sign — does not apply to depth, whose sign is canonical (it grows
> away from the root). Measured cost: 50.1% of the pre-`abs` entries are negative and
> `corr(|signed|, signed) = −0.007`, so the fold is not close to a no-op; child-dominant and
> parent-dominant nodes become indistinguishable. It is kept because `DGNConv` then stays the
> aggregator we verified against the reference to 0.0 error, and because the paper's own
> `dirK-dx-no-abs` variant was never benchmarked ("only B_dx and B_av are tested empirically") —
> a second deviation stacked on the field substitution would be harder to defend than a slightly
> weaker baseline. (2) **DGN runs 2 aggregators on code2 and 3 elsewhere**, so it carries
> 12.05M parameters against PNA's 14.34M. That is *less* of a gap than the other datasets, where
> dropping `min`/`max`/`std` puts DGN at 24–27% below PNA; code2 is 16% below. (3) **`hyperparam_search.py` gets no
> `_eig_kwargs` guard at any of its four model call sites** — not the two in its code2 fork and
> not the two in its standard path. This is a property of the whole script, not a code2
> asymmetry: it builds the hardcoded string `'LDNA'` at both resolver calls and has no `--model`
> flag, so DGN is unreachable there and the resolver would not attach `eig_vec` for a non-DGN
> query anyway. If the search is ever parameterized by model, it fails loudly at the first
> forward (`ValueError: Argument 'eig_vec' must be given ...`), not silently.

> **GIN xor GINE per dataset.** `GIN` and `GINE` are separate baselines, but only **one**
> runs per dataset — edge datasets use `GINE`, edge-less use `GIN` (never fabricate
> `edge_attr`). The rule is realized by the **config inventory**: each dataset ships only
> `gine_<suffix>.yaml` (edge) *or* `gin_<suffix>.yaml` (edge-less), never both — currently
> `gine_{hiv,molpcba,zinc}` and `gin_mnist` (code2 → `gin`). `broadcast.py` globs
> `*_<suffix>.yaml`, so it and the ranking pick up whichever exists; its `HAS_EDGE` map
> selects the matching GINE/GIN template for the `GNN-VPA` config the same way.

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
- **Run logs:** every `train.py` run tees its stdout to `out/logs/<experiment_name>.txt`
  (built into `train.py`); when launched via `run_jobs.sh` the same output (plus stderr) is
  also captured to `out/logs/<experiment_name>.run.log` — a near-duplicate of the `.txt`
  (only extra content is a `pkg_resources` deprecation warning). The final numbers are also
  pickled to `out/logs/<experiment_name>_logger.pickle`.
- **Log/artifact housekeeping (`out/logs/` is gitignored — local only):** once a run finishes,
  its redundant `.run.log` is deleted and both its `.txt` log and its `_logger.pickle` are filed
  together under `out/logs/ranking/<dataset>/` (one folder per dataset holds each config's log +
  pickle); hyperparameter-search logs live under `out/logs/search/`. This is a manual cleanup
  applied to **completed** runs only; live runs still write flat to `out/logs/` and are archived
  when they finish. Ranking analysis reads pickles from `out/logs/ranking/*/` (with a flat
  `out/logs/` fallback for a just-finished, not-yet-archived config).

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
  - **`--mem_fraction` policy (user, 2026-07-08): do NOT cap a search that owns its card.**
    `hyperparam_search.py --mem_fraction` calls `set_per_process_memory_fraction`, a **hard**
    per-process cap: a large sampled config (hidden 1024 × layers 8) that exceeds it OOMs, is
    swallowed by `catch=(RuntimeError,)`, and is **silently dropped** — biasing the search away
    from big models. So run **one uncapped job per dedicated card** (`--mem_fraction 1.0`); it
    uses only what it needs. Re-introduce a fraction **only** when co-locating multiple jobs on
    one card, and even then watch `nvidia-smi` so a big config can't OOM a co-tenant.
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

- [x] Graph-level pipeline infrastructure complete — `GNN-VPA` baseline (`models/vpa.py`),
      the `ogbg-code2` edge-less pipeline (`utils/code2.py`, `models/code2.py`,
      guarded `train.py`/`hyperparam_search.py` fork), and `hyperparam_search.py` extended to all
      five graph-level datasets. Details in `.claude/project_state.md` § Current progress.
- [ ] **Run the graph-level experiment protocol** (see § Experiment queue) on every
      graph-level dataset.

**Node-level (needs new infrastructure):**

- [ ] **Node-level infrastructure**: resolver branches for node datasets; a node-mode
      LDNA/baseline path (no graph pooling — per-node prediction head); transductive
      training loop with train/val/test masks; neighbor sampling for large graphs.
- [ ] Add `CitationFull` (Cora, PubMed) — smallest node datasets; define splits.
- [ ] Add `Reddit` (neighbor sampling).
- [ ] Add `ogbn-products` (neighbor sampling; OGB evaluator).
- [ ] Add `ogbn-proteins` (edge-feature aggregation to nodes; ROC-AUC).

**Expressiveness (custom evaluation):**

- [x] Add `EXP` / `CEXP` (ported dataset + the paper's 10-fold CV; `AccuracyEvaluator`).
- [x] Add `BREC` (dataset + ported RPC pairwise evaluation; `utils/brec.py`, `brec_eval.py`).
- [x] ~~Extend `hyperparam_search.py` to expressiveness datasets~~ — resolved as **not needed**:
      batch-3 hyperparameters are protocol constants (see § Datasets → batch-3 notes). Node-level
      search support still lands with the node-level pipeline.

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

**Batch 3 — synthetic expressiveness** (protocol in § Datasets → batch-3 notes; no search,
no broadcast — the three configs per dataset are the protocol). **Complete 2026-07-28**;
results in § Results log, logs archived under `out/logs/ranking/{exp,cexp,brec}/`:

- [x] `EXP`: LDNA 1.0000 / GIN 0.5000 / DGN 1.0000
- [x] `CEXP`: LDNA 1.0000 / GIN 0.7492 / DGN 1.0000
- [x] `BREC`: LDNA 122.8±9.2 / GIN 0.0 / DGN 271.4±1.0 (of 400)

**Batch 3 — follow-up: DGN BREC canonicalization ablation** (D1 running on cuda:1, 2026-07-28;
D2 designed, not implemented). Code committed: `BRECDataset(canonical=False)` +
`dgn/gin_brec_noncanon.yaml`. (`exp_fold_rerun.py`, added alongside, is the reusable
single-fold EXP/CEXP driver behind the finished fold-10 rerun — see § Results log.)

- **D1 has a GIN control** on the same non-canonical build (`gin_brec_noncanon.yaml`): dropping
  canonicalization changes the RPC statistical regime for *every* model (within-class copies stay
  genuinely permuted → the near-zero within-class covariance behind the S+εI guard becomes a real
  covariance), not just DGN's field. GIN is exactly permutation-invariant, so it isolates the
  regime change from the field effect. Expected signature of a real invariance effect: DGN's drop
  concentrated in the degenerate families (Regular/CFI/DR) — both directional aggregators are
  sign-invariant (`dir1-av` weights by `|field|`; `dir1-dx` takes `.abs()` of its output), so
  simple-λ₁ graphs should not move; a uniform drop across all categories would indicate the
  regime confound instead.

- [ ] **DGN BREC ablation (D1 running; D2 designed, not implemented): how much of 271.4 is the shared canonicalization?** DGN's Laplacian
      directional field is computed from `edge_index` by `torch.linalg.eigh`. On the regular BREC
      families the low-frequency eigenvalues are **degenerate** (repeated), so within the degenerate
      eigenspace "which vector is the eigenvector" is mathematically arbitrary — this is exactly the
      pathology the DGN paper flags when it calls the spectral field "unusable" on symmetric graphs
      (it substitutes image coordinates on CIFAR10 for this reason: grid symmetry makes λ₁
      degenerate). Two effects can rescue it in our pipeline: **(A)** `eigh` is deterministic, so the
      same matrix always yields the same basis; **(B)** the shared canonicalization gives isomorphic
      copies the *same* labelled matrix, so the field becomes isomorphism-invariant (which the RPC
      reliability control rewards). The committed DGN 271.4 (≈ published 3-WL/PPGN level) benefits
      from both — i.e. our shared preprocessing turns the paper's "unusable" field into a legitimate
      invariant feature, which must be disclosed (it already is, in the BREC results-log row). Two
      ablations to quantify the split, DGN-only, BREC-only, everything else at the committed
      64×8/BN-on/S+εI/10-seed setting:
      - **D1 (remove B, keep A):** build BREC **without** canonicalization (raw graph6 order), keep
        deterministic `eigh`. The field stays learnable per graph but is no longer consistent across
        the 32 permuted copies → expect reliability to start failing and the score to drop. Isolates
        how much came from the invariance (B).
      - **D2 (remove A, approach the paper's pathology):** keep canonicalization but **randomize
        inside the degenerate eigenspace** (random orthogonal rotation + sign flip on the near-equal
        eigenvalue groups), restoring the arbitrariness the paper assumes → expect the regular
        families to collapse toward DGN's true unaided level. Isolates the deterministic solver (A).
      Implementation sketch (reverted last time): a `canonical` flag on `BRECDataset` (distinct
      processed-cache filename + a separate `data/brec_noncanon` root so the main cache is untouched)
      for D1; a degenerate-subspace randomizer in `add_eig_vec` gated behind a flag for D2. Intent is
      **honest attribution, not to depress DGN** — if DGN stays strong, report that. It also
      strengthens the LDNA story: LDNA's 122.8 comes purely from its own rank-gate, not from this
      incidental spectral boost.

---

## Results log

Append one row per completed run/trial group. For each dataset, record LDNA's rank
vs. baselines and the margin to #2. Keep § Objective status in sync.

| Date | Dataset | Model | Config summary | Metric (val / test) | LDNA rank / margin | Notes |
|---|---|---|---|---|---|---|
| 2026-07-10 | ogbg-molhiv | LDNA + 10 baselines | tuned h256/l6/do0.143/lr6.30e-5/wd4.52e-6, attention, residual, runs=10, 150ep | test ROC-AUC: LDNA 0.7867±0.0118 (SEM 0.0037) | **#2 / tie** | #1 GAT 0.7875±0.0175 (SEM 0.0055); gap 0.0008 ≪ combined SEM 0.0092 → statistical tie. Field 0.7772–0.7875 all within noise. **Top-2 met; no meaningful #1** (molhiv is high-variance). LDNA search val AUC 0.8244. |
| 2026-07-10 | ZINC (full) | LDNA + 10 baselines | tuned h256/l7/do0.144/lr1.19e-4/wd2.23e-6, attention, residual, runs=3, 150ep | test MAE: LDNA 0.0903±0.0037 (SEM 0.0021) | **#2** | #1 PNA 0.0807±0.0119 (SEM 0.0069); PNA lead 0.0096, PNA_upper 0.0876 < LDNA_lower 0.0882 → marginally significant. **Top-2 met, NOT #1** — PNA edges LDNA (PNA high-variance n=3). Field 0.0807–0.8513. LDNA search val MAE 0.342 (subset). |
| 2026-07-13 | MNISTSuperpixels | LDNA + 10 baselines | tuned h256/l6/do0.128/lr6.65e-4/wd3.25e-6, attention, residual, runs=3, 150ep | test acc: LDNA 0.9357±0.0019 (SEM 0.0011) | **#2** | #1 PNA 0.9532±0.0004 (SEM 0.0002); PNA lead 0.0175, PNA_lower 0.9530 ≫ LDNA_upper 0.9368 → **highly significant**. Top-2 met, NOT #1. egc #3 (0.9288). sage (0.170) + deepergcn_powermean (0.351, huge var) diverge under shared config. Search val acc 0.9319. |
| 2026-07-17 | ogbg-molpcba | LDNA + 10 baselines | tuned h1024/l7/do0.103/lr3.44e-5/wd4.40e-6, attention, residual, 150ep; runs=3 (LDNA+7 baselines), runs=2 (PNA/vpa/deepergcn_powermean, stopped after run 2 per user to save time) | test AP: LDNA 0.2834±0.0016 (SEM 0.0009, n=3) | **#2 / tie** | #1 PNA 0.2838±0.0043 (SEM 0.0030, n=2); PNA lead only 0.0004, PNA_lower 0.2808 < LDNA_upper 0.2843 → **statistical TIE** (no meaningful #1). LDNA's best result vs PNA — matches it here (PNA clearly beat LDNA on ZINC/MNIST) at ~1/3.5 the params (PNA 134M vs LDNA 38.7M). Next sage 0.2715. Search val AP 0.2766. |
| 2026-07-08 | ZINC (full) | LDNA (rank-gated) rerun | same tuned shared config as 07-10 ZINC row, runs=3, 150ep; baselines frozen (LDNA-internal change only) | test MAE: LDNA 0.0856±0.0021 (SEM 0.0012) | **#2** | Rank-gated aggregation vs old LDNA 0.0903±0.0037: −0.0047 (clear improvement). Still behind #1 PNA 0.0807±0.0119 (SEM 0.0069): gap 0.0049 < combined SEM 0.0070 → now within noise of PNA (was marginally significant). Per user: keep tuned params, rerun rank-gated LDNA on all datasets. |
| 2026-07-22 | ogbg-molhiv | LDNA (rank-gated) rerun | same tuned shared config as 07-10 molhiv row, runs=10, 150ep; baselines frozen | test ROC-AUC: LDNA 0.7923±0.0089 (SEM 0.0028) | **#1 (nominal) / tie** | vs old LDNA 0.7867±0.0118: +0.0056 (> combined SEM 0.0047, modest real gain). vs old #1 GAT 0.7875±0.0175 (SEM 0.0055): LDNA leads by 0.0048 < combined SEM 0.0062 → statistical tie, but LDNA takes the nominal top spot (was #2). |
| 2026-07-23 | MNISTSuperpixels | LDNA (rank-gated) rerun | same tuned shared config as 07-13 MNIST row, runs=3, 150ep; baselines frozen | test acc: LDNA 0.9410±0.0023 (SEM 0.0013) | **#2** | vs old LDNA 0.9357±0.0019: +0.0053 (> combined SEM 0.0017, real gain). Still behind #1 PNA 0.9532±0.0004 (SEM 0.0002): gap 0.0122 ≫ combined SEM 0.0013 → still highly significant. Closes ~30% of the old LDNA→PNA gap but PNA's per-node degree-scaler statistics remain out of reach on this dataset. |
| 2026-07-28 | EXP | LDNA + GIN + DGN (batch 3) | canonical order (nauty) + rank_mode position; shared h64/l8, attention, residual, 10-fold CV, 500ep, constant lr 1e-3; per-fold best-of-N selected on **val**, applied symmetrically — three full 10-fold sweeps (seeds 42/43/44) plus one extra fold-10 seed (`exp_fold_rerun.py`) | test acc: LDNA **1.0000** (all 10 folds) | **#1 (tie with DGN)**; 1-WL ceiling broken | DGN 1.0000±0.0000 (seed 42; best-of-N monotone → no extra seeds needed); GIN 0.5000±0.0000 at **all three seeds** — the exact 1-WL ceiling. Fold 10 was the one fold still short of 1.0 under the three sweep seeds; its extra seed reached train/val/test 1.0 (logs `ranking/exp/ldna_exp_fold10.txt` + `_results.json`). |
| 2026-07-28 | CEXP | LDNA + GIN + DGN (batch 3) | same shared config, seed 42, 10-fold CV | test acc: LDNA **1.0000±0.0000** | **#1 (tie with DGN)** | DGN 1.0000±0.0000; GIN 0.7492±0.0026 ≈ the theoretical 75% (corrupt half 1-WL-learnable, EXP-hard half pinned at 50%). No stuck folds for LDNA on CEXP — the corrupt half's easy gradient reliably bootstraps the rank gate. |
| 2026-07-28 | BREC | LDNA + GIN + DGN (batch 3) | RPC protocol, shared 64×8, BN on, S+εI, 10 seeds × 400 pairs | pairs distinguished: LDNA **122.8±9.2 / 400** | **#2** of the 3 run; ≫ every 1-WL baseline (provably and empirically 0) | GIN 0.0±0.0 (exact, all seeds); DGN **271.4±1.0** (#1, ≈ the published 3-WL/PPGN level ~270). LDNA by category: Basic 43.3/60, Extension 61.8/100, 4-Vertex 10.7/20, Regular 6.0/100, DR 0.9/20, CFI 0.1/100 — the rank gate cracks the non-regular families, not CFI. Reliability failures 0 for every model/seed. **Recorded caveat:** DGN's Regular/DR sweep (99/100, 20/20) is partly a by-product of the shared canonicalization — a deterministic node order fixes the otherwise arbitrary degenerate eigenbasis, turning it into an isomorphism-invariant feature; this is shared preprocessing (identical for all models), not DGN-specific tuning, but it materially helps DGN beyond its own paper's setting and is disclosed as such. |

### Parameter-count analysis (recorded 2026-07-13, no action taken yet)

Under the shared-hyperparameter protocol (same `hidden_channels`/`num_layers` for all
models) the **param counts still differ a lot** because each architecture has a different
per-layer structure. From `# Params` in the ranking logs, ratios **relative to LDNA (=1.00×)**:

| model | ZINC (h256/l7) | molpcba (h1024/l7) | MNIST (h256/l6) |
|---|---|---|---|
| **pna** | **3.47×** | **3.47×** | **3.79×** |
| gatv2 | 0.62× | 0.62× | 0.54× |
| gine | 0.43× | 0.43× | (edge-less → gin) |
| sage | 0.43× | 0.43× | 0.53× |
| gat | 0.43× | 0.43× | 0.30× |
| gcn / deepergcn_softmax | 0.24× | 0.24× | 0.30× |
| egc | 0.22× | 0.16× | 0.27× |

(LDNA absolute: ZINC 2.42M, MNIST 1.69M, molpcba 38.7M.)

- **Invariant to `hidden_channels`:** ZINC (h256) and molpcba (h1024) give **identical** ratios
  per model — param count ≈ `c_arch · L · H²` (+ encoder/head), so at fixed `L` the ratio
  cancels `H²`. Ratios **do shift** with `num_layers` and edge-vs-edge-less (MNIST l6/edge-less
  differs from the l7 edge datasets).
- **Fairness caveat:** the shared protocol equalizes hidden/layers, **not** param count. **PNA
  carries ~3.5–3.8× LDNA's params at every setting** — so PNA's #1 on ZINC/MNIST comes with a
  large capacity advantage, and LDNA reaching top-2 at ≈1/3.5 of PNA's params is an efficiency
  point in LDNA's favour. A param-normalised comparison (tune each model's width so params ≈
  LDNA's) is a possible **future** supplementary experiment — not run yet, per user.

---

## References

- OGB: https://ogb.stanford.edu
- PyTorch Geometric: https://pytorch-geometric.readthedocs.io/en/stable/
- EXP / CEXP (GNN-RNI): https://arxiv.org/pdf/2010.01179 · https://github.com/ralphabb/GNN-RNI
- BREC: https://arxiv.org/pdf/2304.07702 · https://github.com/GraphPKU/BREC
- GNN-VPA: https://arxiv.org/abs/2403.04747 · https://github.com/ml-jku/GNN-VPA
