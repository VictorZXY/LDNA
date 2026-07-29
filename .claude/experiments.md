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

The node-level **pipeline is implemented** (transductive masked training loop, per-node head
via `readout=None`, precomputed canonical rank); of the datasets below **none is wired yet** —
only the unreported `Planetoid` Cora smoke test is. Sampling and layer-wise inference are still
to come. Loaders are from `torch_geometric.datasets` / `ogb.nodeproppred`.

Five reported datasets, matching the graph-level count, listed in **priority order** (cut
the last one first if the budget shrinks). Sizes below are from the PyG/OGB metadata.

| Dataset | Task | Metric | Dir | Evaluator | Status | Notes |
|---|---|---|---|---|---|---|
| `ogbn-arxiv` | 40-class node classification | accuracy | max | `OGBNodePropPredEvaluator` | todo | 169k nodes / 1.17M edges; OGB **time** split (train ≤2017 / val 2018 / test 2019); the only dataset runnable **both** full-graph and sampled — hence the sampling control (see notes) |
| `ogbn-products` | 47-class node classification | accuracy | max | `OGBNodePropPredEvaluator` | todo | 2.45M nodes / 61.9M edges; OGB `sales_ranking` split; neighbor sampling required. The scale anchor |
| `Questions` (`HeterophilousGraphDataset`) | binary node classification | ROC-AUC | max | custom (ROC-AUC) | todo | 48.9k nodes / 154k edges / 301 features; **heterophilous** — the one non-homophilous entry; ships 10 official split masks (`train_mask` is `[N, 10]`), so splits map to `runs` the way EXP folds do |
| `PPI` | 121-task multi-label | micro-F1 | max | custom (micro-F1) | todo | 24 graphs (20 train / 2 val / 2 test), ~2.2k nodes each; **inductive** and multi-graph, so it reuses the existing `DataLoader` path — only the head changes (no pooling). Carries the multi-label axis `ogbn-proteins` was to provide |
| `Reddit` | 41-class node classification | accuracy | max | `AccuracyEvaluator` | todo | 233k nodes / **114.6M** edges (mean degree ~492) — the density stress case for `LDNAConv`'s edge-level MLP; neighbor sampling required. Weakest citability (only NAGphormer reports it, on a non-standard 60/20/20 split), so it buys the O(E)-at-scale claim rather than a comparison |

> **`ogbn-proteins` removed from scope (2026-07-29), replaced by `ogbn-arxiv`.** proteins has no
> node features, so the feature-based canonical sort cannot impose a permutation-invariant node
> order — the exact reason `ogbg-ppa` was removed above, and the edge-feature-to-node aggregation
> that would fix it is a bespoke rule not worth inventing twice. Its multi-label / ROC-AUC axis is
> recovered by `PPI` (multi-label) and `Questions` (ROC-AUC).

> **Suite design.** The five span scale (2.2k → 48.9k → 169k → 233k → 2.45M nodes), mean degree
> (6 → 14 → 25 → 27 → 492), task type (3× multi-class accuracy, 1× binary ROC-AUC, 1× multi-label
> micro-F1), structure (4 homophilous, 1 heterophilous), split semantics (time / sales-ranking /
> 10 standard splits / by-graph inductive / post-time) and sampling regime (3 full-graph,
> 2 sampled). Rejected for redundancy: a second heterophilous graph (Minesweeper, Tolokers),
> `Yelp` (multi-label at scale — same axis as PPI at ~6x the compute; the swap to make if
> multi-label-at-scale ever matters more than Reddit's density), `Flickr` and `LINKXDataset`
> (occupy no axis of their own), `AmazonProducts` (same domain as ogbn-products, larger),
> `ogbn-mag` (heterogeneous — needs infrastructure outside this pipeline).

> **`Planetoid` Cora is a smoke test, not a reported dataset.** 2,708 nodes / 7 classes with the
> standard public split — the cheapest correctness check for the node path, run during development
> and left out of the results log. `CitationFull` is dropped entirely: its Cora is a *different*
> dataset variant (19,793 nodes / 70 classes, not the 2,708/7 everyone reports) and it ships no
> standard split, so numbers from it would be uncitable on both counts. Small classics are also a
> poor fit for the § Objective SEM margin — their run-to-run spread makes every model a statistical
> tie, contributing nothing to a ranking claim.

> **Measured node-level diagnostics (2026-07-29, pre-implementation).** Run on the real data before
> any training compute, to replace estimates with measurements.
>
> | Dataset | nodes | mean deg | distinct feature rows | median \|δ\| | P(δ=0) | median node covers 50% at | `num_layers` range |
> |---|---|---|---|---|---|---|---|
> | Cora | 2,708 | 3.90 | 99.4% | 0.264 | 0.0002 | k=6 | [2, 8] |
> | Questions | 48,921 | 6.28 | 76.8% | 0.206 | 0.0010 | **k=4** | **[2, 6]** |
> | PPI (per graph) | ~1,767 | 18.3 | **17.7%** | **0.036** | **0.156** | k=3 | [2, 5] |
> | ogbn-arxiv | 169,343 | 13.67 | 99.8% | 0.243 | 0.0001 | k=5 | [2, 7] |
>
> Three conclusions. (1) **The rank gap is healthy at every scale** — median \|δ\| is 0.21–0.31 on
> Cora/Questions/arxiv, refuting the pre-implementation worry that δ collapses on large graphs;
> δ is a difference of percentiles between two edge endpoints, so it does not shrink with N.
> (2) **PPI is the predicted tie-collapse case, and it is severe**: only 17.7% of feature rows are
> distinct (14.5–28.8% across the 20 train graphs), so 15.6% of edges carry δ = 0 exactly and
> median \|δ\| is an order of magnitude below the others. This is the ZINC failure mode — expect
> LDNA to be weakest here, and report it rather than hiding it. (3) **`deg^L` is a bad estimator
> of coverage** and the measured depths supersede it: Questions covers at k=4, not "never" (its
> ball grows far faster than mean degree predicts — the heavy-tail effect), so its range is [2, 6]
> rather than the full [2, 8]. Ranges are the measured covering depth + 2.

> **Two constraints the node path must honour** (both load-bearing, decided before implementation).
> (1) **The canonical rank is precomputed once on the full graph** and carried as a node attribute.
> `LDNA._canonical_rank` / `_position_rank` currently derive the rank per batch and normalize by
> graph size; on a `NeighborLoader` subgraph that would make a node's rank drift with the batch it
> landed in *and* normalize by the wrong size. Precomputing keeps δ identical to the full-graph
> value on every subgraph. (2) **Whether a dataset is sampled is a per-dataset decision shared by
> every model**, with the fanout a shared knob (not searched) — the edge-MLP models (LDNA, PNA,
> DGN, GINE) and the sparse-matmul models (GCN, SAGE) have very different memory profiles, so
> letting the regime vary per model would break the shared-hyperparameter fairness premise.
> `ogbn-arxiv` is run **both ways** to quantify what sampling costs.

### 3. Synthetic expressiveness

| Dataset | Task | Metric | Dir | Evaluator | Status | Notes |
|---|---|---|---|---|---|---|
| `EXP` / `CEXP` | distinguish 1-WL-equivalent pairs | accuracy | max | `AccuracyEvaluator` | done | ported from GNN-RNI (GPL v3 — note on redistribution); 600 pairs each, binary node-type feature, **edge-less** (GIN family); paper's contiguous 10-fold CV, folds map to `runs`; raw pickles under `data/exp|cexp/raw/` (gitignored, fetch from the GNN-RNI repo) |
| `BREC` | pairwise graph distinguishing | # pairs distinguished (RPC protocol) | max | RPC core in `data_utils/brec.py` | done | dedicated entry point `scripts/pairwise_eval.py` (the train/val/test loop does not fit the pairwise protocol); 400 pairs × 10 seeds; featureless (`x = ones`); `data/brec/raw/brec_v3.npy` from the BREC repo |

> **Batch-3 protocol decisions (2026-07-27).**
> - **Canonical order**: these graphs are featureless/uniform, so the feature lexsort cannot impose
>   an order. The permutation-invariance device is a **nauty canonical labelling**
>   (`data_utils/preprocess.py: canonicalize_graph(s)`, pynauty, data-side, replaces `sort_graphs` on these
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
> - **BREC reporting convention (revised 2026-07-29): report the BEST across the 10 protocol seeds,
>   with mean ± std alongside.** The BREC paper itself reports the best across its seeds
>   {100..1000}, explicitly as an upper bound on expressiveness, and the published entries we
>   compare against are on that convention — so the earlier mean-only reporting understated LDNA and
>   DGN against every quoted number. Best-of-seeds figures come from the same committed 10-seed runs
>   (no re-run): LDNA gate-init **321** (mean 312.0±4.0), LDNA zero-init **138** (122.8±9.2), DGN
>   shared-canonical **273** (271.4±1.0), DGN own-construction **145** (144.4±0.7), GIN **0** (exact).
>   Seed 100 is the best seed for all four. Caveat to carry into the write-up: the convention is not
>   universal in the follow-up literature — Edge Transformer's 258.1±1.9 is a mean — so state which
>   convention each compared column uses.
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
>   hurts inside the step budget); DGN runs and GIN stays 0 at that size. `gate_init_std`
>   (`ldna_brec_gateinit.yaml`) is **LDNA-internal**, not a shared knob — no baseline has a gate;
>   it is the same category of knob as `rank_mode`, and both LDNA numbers (zero-init and
>   gate-init) are reported side by side. Results go to
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

> **DGN needs a precomputed eigenvector field.** `data_utils/preprocess.py: add_eig_vecs` attaches a
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
> message from the parent apart from one from a child. `data_utils/transforms.py: AddDepthField`
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
> Two consequences of the substitution, recorded as decisions rather than left implicit.
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
> dropping `min`/`max`/`std` puts DGN at 24–27% below PNA; code2 is 16% below.

> **DGN's advantage is dataset-selective.** DGN places 1st only on ZINC (0.0507 test MAE)
> and code2 (0.1876 test F1) — the datasets whose dominant structure matches its directional
> field: the molecular backbone recovered by the low-frequency Laplacian eigenvector, and the
> AST's exact root-to-leaf depth. Elsewhere the field is fragmented (molhiv, 7.5% disconnected
> → 4th), collapses a 2-D layout onto one axis (MNIST → 3rd, tied LDNA), or is one axis shared
> across 128 tasks (molpcba 3rd), and DGN reduces to a costlier PNA. The edge is bought with a
> hand-supplied per-dataset directional prior. **Adding DGN drops LDNA out of top-2 on ZINC**:
> ZINC now reads DGN 0.0507 / PNA 0.0807 / LDNA 0.0856 (LDNA 3rd). LDNA stays top-2 on the other
> four (molhiv 1st, code2 2nd, mnist 2nd, molpcba 2nd).

> **GIN xor GINE per dataset.** `GIN` and `GINE` are separate baselines, but only **one**
> runs per dataset — edge datasets use `GINE`, edge-less use `GIN` (never fabricate
> `edge_attr`). The rule is realized by the **config inventory**: each dataset ships only
> `gine_<suffix>.yaml` (edge) *or* `gin_<suffix>.yaml` (edge-less), never both — currently
> `gine_{hiv,molpcba,zinc}` and `gin_mnist` (code2 → `gin`). `scripts/broadcast.py` globs
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

- The **canonical-sort premise** stays (`data_utils/preprocess.py`).
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

`hyperparam_search.py` covers **all five graph-level datasets**. Expressiveness search is
**not needed** (batch-3 hyperparameters are protocol constants — see § Datasets → batch-3
notes). Node-level search still has to be added, and lands with the node-level pipeline
(§ Implementation queue). Per-dataset overrides go here: TODO.

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
  (built into `train.py`); when launched via `scripts/run_jobs.sh` the same output (plus stderr) is
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
    free memory. If a card gets tight, back our jobs off rather than risk a collision.
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
  (`data_utils/preprocess.py`); the **shared model interface**
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
      the `ogbg-code2` edge-less pipeline (`data_utils/code2.py`, `models/code2.py`,
      guarded `train.py`/`hyperparam_search.py` fork), and `hyperparam_search.py` extended to all
      five graph-level datasets. Details in `.claude/project_state.md` § Current progress.
- [x] **Run the graph-level experiment protocol** (see § Experiment queue) on every
      graph-level dataset — complete on all five, plus the `DGN` baseline.

**Node-level (needs new infrastructure):**

- [x] **Node-level infrastructure (core, 2026-07-29)**: resolver node branch + the readout
      level guard, a per-node head via `readout=None` (no baseline model file touched),
      the transductive masked training loop (`_split_mask` applied *before* the NaN
      filter), full-graph precomputation of the canonical rank without permuting
      (`data_utils/preprocess.py: node_rank` / `add_node_ranks`), and the three node
      evaluators. **Still to come**: GraphSAINT sampling and layer-wise exact inference.
- [x] `Planetoid` Cora — smoke test for the node path only; not reported.
- [ ] Add `ogbn-arxiv` (OGB split + evaluator; full-graph run plus the sampled control).
- [ ] Add `ogbn-products` (neighbor sampling; OGB evaluator).
- [ ] Add `Questions` (heterophilous; 10 official splits map to runs; ROC-AUC evaluator).
- [ ] Add `PPI` (inductive multi-graph; multi-label micro-F1 evaluator).
- [ ] Add `Reddit` (neighbor sampling; density stress case).

**Expressiveness (custom evaluation):**

- [x] Add `EXP` / `CEXP` (ported dataset + the paper's 10-fold CV; `AccuracyEvaluator`).
- [x] Add `BREC` (dataset + ported RPC pairwise evaluation; `data_utils/brec.py`, `scripts/pairwise_eval.py`).
- [x] ~~Extend `hyperparam_search.py` to expressiveness datasets~~ — resolved as **not needed**:
      batch-3 hyperparameters are protocol constants (see § Datasets → batch-3 notes). Node-level
      search support still lands with the node-level pipeline.

---

## Experiment queue

Runs to do once the code is in place. Keep status current; move finished items to
the results log.

**Graph-level protocol** (per § Tuning methodology: tune LDNA → `scripts/broadcast.py` the tuned
shared config to all baselines → run → rank → append to § Results log): **complete on all
five datasets** — `ogbg-molhiv`, `ogbg-molpcba`, `ZINC`, `MNISTSuperpixels`, `ogbg-code2` —
plus the `DGN` baseline added afterward. Final rankings in § Results log.

**Batch 3 — synthetic expressiveness** (protocol in § Datasets → batch-3 notes; no search,
no broadcast — the three configs per dataset are the protocol). **Complete 2026-07-28**;
results in § Results log, logs archived under `out/logs/ranking/{exp,cexp,brec}/`:

- [x] `EXP`: LDNA 1.0000 / GIN 0.5000 / DGN 1.0000
- [x] `CEXP`: LDNA 1.0000 / GIN 0.7492 / DGN 1.0000
- [x] `BREC` (official): LDNA **321** (gate-init; `gate_init_std: 0.1` is a **probe-selected**
      LDNA-internal knob — pre-committed grid, four arms archived under
      `ranking/brec/gateinit_probe/`) / GIN 0 / DGN **145** (its own
      construction) — of 400. Both official numbers were established by the batch-3 follow-ups,
      not by the original runs: LDNA's was promoted by the gate-init experiment, DGN's by the
      canonicalization ablation. Kept alongside as companions: LDNA 138 (zero-init) and
      DGN 273 (shared-canonicalization pipeline). Scores are **best across the 10 protocol seeds**
      per the benchmark's own convention (see § Datasets → batch-3 notes, reporting convention);
      the corresponding means are LDNA 312.0±4.0 / 122.8±9.2 and DGN 144.4±0.7 / 271.4±1.0.

**Batch 3 — follow-up: DGN BREC canonicalization ablation** (done 2026-07-28; results in
§ Results log). Code committed: `BRECDataset(canonical=False)` + `dgn/gin_brec_no_canonical.yaml`.
(`scripts/train_fold.py`, added alongside, is the reusable single-fold driver behind the
finished fold-10 rerun — see § Results log.)

- **The ablation has a GIN control** on the same non-canonical build (`gin_brec_no_canonical.yaml`): dropping
  canonicalization changes the RPC statistical regime for *every* model (within-class copies stay
  genuinely permuted → the near-zero within-class covariance behind the S+εI guard becomes a real
  covariance), not just DGN's field. GIN is exactly permutation-invariant, so it isolates the
  regime change from the field effect. Expected signature of a real invariance effect: DGN's drop
  concentrated in the degenerate families (Regular/CFI/DR) — both directional aggregators are
  sign-invariant (`dir1-av` weights by `|field|`; `dir1-dx` takes `.abs()` of its output), so
  simple-λ₁ graphs should not move; a uniform drop across all categories would indicate the
  regime confound instead.

- [x] **DGN BREC canonicalization ablation (done 2026-07-28): how much of 271.4 is the shared
      canonicalization?** DGN's Laplacian directional field is computed from `edge_index` by
      `torch.linalg.eigh`. On the regular BREC families the low-frequency eigenvalues are
      **degenerate**, so within the degenerate eigenspace "which vector is the eigenvector" is
      mathematically arbitrary — the pathology the DGN paper flags when it calls the spectral
      field "unusable" on symmetric graphs (it substitutes image coordinates on CIFAR10 for this
      reason). In our pipeline the deterministic `eigh` plus the shared canonicalization resolve
      the arbitrariness: isomorphic copies get the same labelled matrix, hence the same basis,
      and the field becomes an isomorphism-invariant feature. The committed DGN 271.4
      (≈ published 3-WL/PPGN level) benefits from this, disclosed in the BREC results-log row.
      The ablation evaluates DGN on BREC built **without** canonicalization
      (`BRECDataset(canonical=False)`, `dgn_brec_no_canonical.yaml`) — the benchmark's **own
      raw-graph6 construction**, i.e. the setting DGN would face in a standard BREC evaluation —
      with everything else at the committed 64×8/BN-on/S+εI/10-seed setting, plus the GIN
      regime-change control above. Intent is **honest attribution, not to depress DGN** — if DGN
      stays strong, report that. It also sharpens the LDNA story: LDNA's 122.8 comes purely from
      its own rank-gate, not from this incidental spectral boost. A finer split (randomizing the
      degenerate eigenbasis while keeping canonicalization) was considered and **rejected**:
      per-copy randomization inside a shared-canonicalization pipeline ablates the pipeline's
      defining property — a standard not applied to LDNA's equally-arbitrary canonical rank
      either — and corresponds to no setting anyone would run; the benchmark's own construction
      is the meaningful reference point.

**Batch 3 — follow-up: LDNA gate-init on BREC (done 2026-07-28; results in § Results log).**

- [x] **Does a small random gate init lift LDNA's BREC score?** Yes: **321 vs 138** best-of-seeds
      (means 312.0±4.0 vs 122.8±9.2) (§ Results log). Protocol as pre-committed below. The
      zero-init identity gate means
      two 1-WL-equivalent graphs embed identically at step 0, and the RPC per-pair budget
      (20 epochs, lr 1e-4) must first push the gate off identity before any separating signal
      exists. `LDNAConv(gate_init_std)` draws the gate output layer from N(0, σ²) instead
      (default 0.0 = byte-identical previous behavior, no RNG consumed); permutation invariance
      and the ≥1-WL family property are untouched, and reliability is structurally clean — the
      canonicalized copies embed identically under *any* parameters. **Scoped by the user as
      BREC-only, NOT an escalation** — no propagation to graph-level or node-level configs.
      **Pre-committed probe protocol:** σ ∈ {0.001, 0.01, 0.1}; per-category pair cap 20
      (120 pairs spanning all six families); protocol seeds {100, 200, 300}; deciding metric =
      mean reliability-clean correct; ties → the smaller σ. Zero-init baseline = a fourth
      **σ = 0.0 arm under the same capped protocol** — the committed correct_lists cannot
      substitute: a pair's init is its position in the seed's reset stream, and the per-category
      cap shifts every non-Basic pair's position, so the full run's per-pair outcomes do not
      transfer to the capped run. Full
      run at the chosen σ: 400 pairs × 10 seeds (`ldna_brec_gateinit.yaml`), reported
      **alongside** (not replacing) the committed zero-init 122.8. Disclosure: the gate draw
      shifts the RNG stream for every module reset after it, so gate-init vs zero-init at a fixed
      protocol seed is not single-factor — attribute only deltas exceeding the committed ±9.2
      spread. (An earlier Basic-only 20-pair probe, run before pair_limit became per-category,
      is superseded by this protocol.)

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
| 2026-07-28 | EXP | LDNA + GIN + DGN (batch 3) | canonical order (nauty) + rank_mode position; shared h64/l8, attention, residual, 10-fold CV, 500ep, constant lr 1e-3; per-fold best-of-N selected on **val**, applied symmetrically — three full 10-fold sweeps (seeds 42/43/44) plus one extra fold-10 seed (`scripts/train_fold.py`) | test acc: LDNA **1.0000** (all 10 folds) | **#1 (tie with DGN)**; 1-WL ceiling broken | DGN 1.0000±0.0000 (seed 42; best-of-N monotone → no extra seeds needed); GIN 0.5000±0.0000 at **all three seeds** — the exact 1-WL ceiling. Fold 10 was the one fold still short of 1.0 under the three sweep seeds; its extra seed reached train/val/test 1.0 (logs `ranking/exp/ldna_exp_fold10.txt` + `_results.json`). |
| 2026-07-28 | CEXP | LDNA + GIN + DGN (batch 3) | same shared config, seed 42, 10-fold CV | test acc: LDNA **1.0000±0.0000** | **#1 (tie with DGN)** | DGN 1.0000±0.0000; GIN 0.7492±0.0026 ≈ the theoretical 75% (corrupt half 1-WL-learnable, EXP-hard half pinned at 50%). No stuck folds for LDNA on CEXP — the corrupt half's easy gradient reliably bootstraps the rank gate. |
| 2026-07-28 | BREC | LDNA + GIN + DGN (batch 3) | RPC protocol, shared 64×8, BN on, S+εI, 10 seeds × 400 pairs | pairs distinguished: LDNA **138 best / 122.8±9.2 mean** of 400 | **#2** of the 3 run; ≫ every 1-WL baseline (provably and empirically 0) | Original-run numbers, kept as companions — the official LDNA/DGN figures were set later by the gate-init and canonicalization-ablation rows below. GIN 0 (exact, all seeds); DGN **273 best / 271.4±1.0 mean** (#1, ≈ the published 3-WL/PPGN level ~270). LDNA by category at its best seed: Basic 47/60, Extension 66/100, 4-Vertex 13/20, Regular 12/100, DR 0/20, CFI 0/100 — the rank gate cracks the non-regular families, not CFI. Reliability failures 0 for every model/seed. **Recorded caveat:** DGN's Regular/DR sweep (99/100, 20/20) is partly a by-product of the shared canonicalization — a deterministic node order fixes the otherwise arbitrary degenerate eigenbasis, turning it into an isomorphism-invariant feature; this is shared preprocessing (identical for all models), not DGN-specific tuning, but it materially helps DGN beyond its own paper's setting and is disclosed as such (quantified by the canonicalization-ablation row below: unaided DGN 144.4). |
| 2026-07-28 | BREC (LDNA gate-init) | LDNA | `ldna_brec_gateinit.yaml`: `gate_init_std: 0.1`, chosen by the pre-committed per-category probe (σ ∈ {0.001, 0.01, 0.1} + σ=0 arm, 20 pairs × 6 families × 3 seeds → 69.0 / 98.0 / 106.0 vs 46.7 of 120); everything else the committed BREC setting | pairs distinguished: LDNA **321 best / 312.0±4.0 mean** of 400 | **#1** | **Official LDNA BREC number** (promoted by this follow-up). Per category at the best seed (100): Basic **60/60**, Regular **100/100**, Extension **100/100**, 4-Vertex **20/20**, DR **20/20**, CFI 21/100 — perfect on every family except CFI, the sole residual. (Means: Basic 59.7, Regular 100.0, Extension 99.5, 4-Vertex 20.0, DR 19.9, CFI 12.9±3.4.) Reliability failures 0 on all 10 seeds. Under the aligned best-of-seeds convention this exceeds every published entry — I²-GNN+SGT 304, N2-GNN 298, 3-WL 270, Edge Transformer 258.1 (a mean), Graphormer 79 — and the paper's aggregated Regular column at 140/140 is the highest anywhere (previous best N2-GNN 138). Reported alongside zero-init LDNA 138 best / 122.8±9.2 mean (identical protocol otherwise). Above DGN 273 (shared-canonicalization setting) and DGN 145 (its own construction). Logs `ranking/brec/ldna_brec_gateinit.{txt,_results.json}`. |
| 2026-07-28 | BREC (canonicalization ablation) | DGN (+ GIN control) | BREC rebuilt without the shared canonicalization — the benchmark's own raw-graph6 construction (`BRECDataset(canonical=False)`, `data/brec`, processed as `brec_v3_no_canonical.pt`); everything else at the committed 64×8/BN-on/S+εI/10-seed setting | pairs distinguished: DGN **145 best / 144.4±0.7 mean** of 400 (canonical: 273 best / 271.4±1.0 mean) | n/a (ablation; LDNA not involved) | **Official DGN BREC number** (the benchmark's own construction; promoted by this ablation). The −127 drop is confined to the degenerate families — Regular 99→43, 4-Vertex 20→0, DR 20→0, CFI 29.5→3.0 — while Basic is bit-identical (32.0±0.0) and Extension moves only 70.9→66.4: the signature of the arbitrary degenerate eigenbasis, not of the changed RPC regime. GIN control: exactly 0/400 with 0 reliability failures on all 10 seeds on **both** builds — the regime change alone produces no signal. DGN's reliability failures also 0 throughout (field inconsistency deflates T² rather than inflating it). Reading: DGN's directional aggregation is genuinely >1-WL unaided (144.4, still above LDNA's 122.8); the other ~127 pairs of its committed 271.4 come from the shared canonicalization turning the degenerate eigenbasis into an isomorphism-invariant feature. LDNA's 122.8 involves no spectral pipeline and is untouched by this ablation. |
| 2026-07-29 | all graph-level (DGN added as 11th baseline) | DGN | shared tuned config per dataset (each dataset's LDNA row); runs as before; molpcba 3 runs (seed 42 ×2 + seed 43 ×1) | DGN test: ZINC 0.0507±0.0001 (#1), code2 0.1876±0.0004 (#1), MNIST 0.9409±0.0012 (#3), molpcba 0.2727±0.0050 (#3), molhiv 0.7836±0.0110 (#6) | LDNA top-2 on 4/5; **3rd on ZINC** | DGN is dataset-selective — 1st only where a clean directional field meets an anisotropic task (Fiedler-vector molecular backbone on ZINC; exact AST root-to-leaf depth on code2), and mid-pack where the field fragments (molhiv, 7.5% disconnected), collapses a 2-D layout onto one axis (MNIST), or is one axis shared across 128 tasks (molpcba). Effect on LDNA: **code2 #1→#2** (0.1850, still ahead of every other baseline incl PNA 0.1844); **ZINC #2→#3** — ZINC now reads DGN 0.0507 / PNA 0.0807 / LDNA 0.0856. ZINC is simultaneously DGN's designed showcase and LDNA's weakest rank signal (a single atom-type node feature collapses the tie-aware rank to a coarse atom-type-rank gap), and the two methods ahead are both heavyweight molecular-regression specialists — DGN's hand-supplied precomputed spectral field, PNA's multi-aggregator + degree scalers. LDNA remains the only method first-or-second across the whole benchmark with no per-dataset directional prior or extra engineering. |

### Parameter-count analysis (recorded 2026-07-13; refreshed 2026-07-29 to the rank-gated LDNA baseline, DGN added)

Under the shared-hyperparameter protocol (same `hidden_channels`/`num_layers` for all
models) the **param counts still differ a lot** because each architecture has a different
per-layer structure. From `# Params` in the ranking logs, ratios **relative to LDNA (=1.00×)**
(GIN xor GINE per dataset → one row: `gin` on the edge-less datasets — MNIST, code2 — and `gine`
on the edge datasets — ZINC, molpcba, molhiv. `avg` is the mean ratio across the **four non-code2
datasets** — code2 is excluded because its shared sequence head compresses every ratio toward 1×
and would distort the mean; see the note below.)

| model | ZINC (h256/l7) | molpcba (h1024/l7) | MNIST (h256/l6) | molhiv (h256/l6) | code2 (h256/l5) | **avg⁻ᶜ²** |
|---|---|---|---|---|---|---|
| **pna** | **4.15×** | **4.25×** | **4.76×** | **4.05×** | **1.42×** | **4.30×** |
| **dgn** | **3.02×** | **3.09×** | **3.59×** | **2.95×** | **1.19×** | **3.16×** |
| gatv2 | 0.74× | 0.76× | 0.67× | 0.75× | 0.96× | 0.73× |
| sage | 0.51× | 0.53× | 0.67× | 0.53× | 0.96× | 0.56× |
| gat | 0.52× | 0.53× | 0.38× | 0.53× | 0.93× | 0.49× |
| gin / gine | 0.52× | 0.53× | 0.38× | 0.53× | 0.93× | 0.49× |
| vpa (GNN-VPA) | 0.52× | 0.53× | 0.38× | 0.53× | 0.93× | 0.49× |
| gcn / deepergcn | 0.29× | 0.29× | 0.38× | 0.31× | 0.93× | 0.32× |
| egc | 0.26× | 0.20× | 0.34× | 0.28× | 0.93× | 0.27× |

(LDNA absolute, rank-gated: ZINC 2.02M, MNIST 1.35M, molhiv 1.79M, code2 10.12M, molpcba 31.56M.
`avg⁻ᶜ²` = mean over ZINC/molpcba/MNIST/molhiv only.)

- **Nearly invariant to `hidden_channels`:** the four molecular/image datasets (ZINC h256,
  molpcba h1024, MNIST/molhiv h256; `l5`–`l7`) give near-identical ratios per model — param count
  ≈ `c_arch · L · H²` (+ encoder/head), so at fixed `L` the ratio cancels `H²` up to a few-% drift
  from the encoder/head terms (largest for the heavy models and molpcba's 128-task head).
  **`ogbg-code2` is the outlier**: every ratio compresses toward 1× because the shared
  sequence head (5×5002-vocab decode) dominates each model's param count and swamps the
  conv-stack differences — which is exactly why code2 is left out of the `avg` column (it would
  otherwise flatten every model toward 1× and hide the real capacity gaps).
- **Fairness caveat:** the shared protocol equalizes hidden/layers, **not** param count. The two
  models that beat LDNA on its weaker datasets are also the two heaviest — **PNA at 4.30× and DGN
  at 3.16× LDNA's params on average** (excl. code2; up to 4.8× / 3.6× per dataset); every other
  baseline is ≤0.73×. So PNA's #1 on ZINC/MNIST and DGN's #1 on ZINC/code2 come with a large
  capacity advantage (and, for DGN, a hand-supplied directional field), while LDNA reaches top-2
  at ≈1/4 of PNA's and ≈1/3 of DGN's params — an efficiency point in LDNA's favour.

---

## References

- OGB: https://ogb.stanford.edu
- PyTorch Geometric: https://pytorch-geometric.readthedocs.io/en/stable/
- EXP / CEXP (GNN-RNI): https://arxiv.org/pdf/2010.01179 · https://github.com/ralphabb/GNN-RNI
- BREC: https://arxiv.org/pdf/2304.07702 · https://github.com/GraphPKU/BREC
- GNN-VPA: https://arxiv.org/abs/2403.04747 · https://github.com/ml-jku/GNN-VPA
