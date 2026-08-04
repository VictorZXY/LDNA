# Experiments & Tuning Playbook

Single source of truth for autonomously optimizing LDNA. The standing rules for
autonomous work live in `AGENTS.md` (§ Autonomous optimization); this file holds
the concrete objectives, datasets, baselines, search spaces, queues, and results.

Do not invent targets or experiments not listed here.
Metrics/evaluators for not-yet-implemented datasets are best-effort and must be
confirmed against the OGB / PyG docs (see References) when implementing.

---

## Objective

**Status (2026-08-04): the primary objective is met on three of the five graph-level datasets and
not met on two.** LDNA is #1 on molhiv (a statistical tie with GAT), #2 on molpcba and MNIST and
code2, and **#3 on ZINC** (behind DGN and PNA, both of which carry a hand-supplied per-dataset
prior). On the one node-level dataset ranked, `Questions`, LDNA is **#3 of 11** and the gap to #1
is significant. Details in § Results log.

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
exists but not tuned/run; `todo` = not implemented; `dropped` = deliberately not implemented,
with the reason recorded at the section that owns it.

### 1. Graph-level prediction

| Dataset | Task | Metric | Dir | Evaluator | Status | Notes |
|---|---|---|---|---|---|---|
| `ogbg-molhiv` | binary classification | ROC-AUC | max | `OGBGraphPropPredEvaluator` | done | wired; configs; high run-to-run variance — use more runs (e.g. 10) |
| `ogbg-molpcba` | multi-task binary (128) | AP | max | `OGBGraphPropPredEvaluator` | done | wired; configs; largest (350k graphs) — **search capped at ~25–30 trials** (or an earlier plateau) by standing instruction, while the ranking stays full; the retained search log shows 30 finished trials against the 100 the general budget states |
| `ZINC` | graph regression | MAE | min | `ZINCEvaluator` | done | wired; configs; search uses `subset=True` (~10k), final training uses full ZINC |
| `MNISTSuperpixels` | 10-way classification | accuracy | max | `MNISTEvaluator` | done | wired; configs. **Its validation split IS its test split** — `utils/resolver.py` assigns `val_dataset = test_dataset = MNISTSuperpixels(train=False, …)`, since the dataset ships no separate validation set. Every MNIST log therefore has val ≡ test to the last digit for every model, run and epoch. Harmless for the ranking (identical for all models) but it means MNIST has no held-out selection signal, so its best-over-epochs figure is selected on the same split it reports |
| `ogbg-code2` | AST subtoken prediction | F1 | max | `Code2Evaluator` (OGB F1) | done | **edge-less** (GIN family, no `edge_attr`); dedicated seq-head (5×5002) + sum-over-position CE loss + decode→F1 eval + train-split vocab pre-pass; skip lexsort (AST DFS order is canonical). `num_nodetypes=98`, `num_nodeattributes=10030` |
| `Peptides-func` | 10-task multi-label | AP | max | `OGBGraphPropPredEvaluator`-style AP (new) | todo | **PLANNED.** `LRGBDataset(name='Peptides-func')`. ~15.5k graphs, ~150 nodes, atom-level features and bond `edge_attr` — so the feature lexsort applies unchanged and this is a **GINE** dataset. The point is the **long-range axis**, which none of the current five has: peptide chains have large diameter and the label depends on dependencies many hops apart. LRGB is the standard long-range benchmark since 2022 (GraphGPS, Exphormer, GRIT all report on it), so the numbers are citable |
| `Peptides-struct` | 11-target regression | MAE | min | `ZINCEvaluator`-style MAE (new, multi-target) | todo | **PLANNED.** Same loader and same graphs as `Peptides-func`, different targets. Adds **multi-target regression**, which the suite also lacks — `ZINCEvaluator` handles one target, so this needs a multi-target MAE variant (mean over the 11 normalized targets, per the LRGB protocol) |
| `CIFAR10` (superpixels) | 10-way classification | accuracy | max | `MNISTEvaluator` | todo | **PLANNED.** `GNNBenchmarkDataset(name='CIFAR10')`, the direct sibling of `MNISTSuperpixels` — same superpixel construction, same pipeline (`T.Cartesian` + `RemoveEdgeAttr`), harder task. It adds no new structural axis; it is here because the benchmarking-gnns suite is expected complete and a MNIST row without a CIFAR10 row is conspicuous. **Check whether it ships a real validation split** — MNIST's does not (see its row), and if CIFAR10's does not either, that has to be disclosed the same way |
| `ogbg-ppa` | 37-way classification | accuracy | max | `OGBGraphPropPredEvaluator` | todo | **PLANNED — readmitted via nauty.** 158,100 graphs, ~243 nodes / ~2,266 edges each, **no node features**, 7-dim `edge_attr` (so **GINE**). It was removed in 2026-07-07 (note below) because the feature lexsort cannot impose a permutation-invariant order without node features. A **nauty canonical labelling** removes exactly that blocker — the same device already used for EXP/CEXP/BREC — via `canonicalize_graphs` + `rank_mode: position`. Measured feasibility: ~0.022 s per graph at this size and density, i.e. **~1 CPU-hour once** for the full set, and cacheable. See the readmission note below for what this does and does not buy |

> **`ogbg-ppa`: removed 2026-07-07, readmitted 2026-08-04 via a nauty canonical labelling.**
> The original reason for removing it stands and is worth restating, because it is a *correctness*
> reason and not a performance one: ppa has no node features, so the feature-based canonical sort
> cannot impose a **permutation-invariant** node order (identical features → the sort falls back to
> input order), and LDNA *requires* a canonical order for its MLP aggregator to be permutation
> invariant. The sort is a correctness requirement, **not** a performance booster.
>
> What changed is that the project has since built and validated the device that supplies such an
> order without features: `data_utils/preprocess.py: canonicalize_graph(s)` (pynauty), used for
> EXP/CEXP/BREC, read by LDNA through `rank_mode: position`. Isomorphic graphs get the identical
> labelled form, so invariance is exact. Measured cost at ppa's size and density (~243 nodes, mean
> degree ~18.6): **0.022 s per graph, ≈1 CPU-hour for all 158,100**, dominated by the Python-side
> adjacency build rather than by nauty itself, and cacheable in the dataset's `processed_dir` the
> way `add_eig_vecs` already caches the DGN field.
>
> **What this buys, honestly.** It restores a dataset excluded on correctness grounds and adds the
> suite's only **featureless** entry, which demonstrates that LDNA's canonical-order premise does
> not depend on node features existing. It is **not** a prediction of a performance gain, and the
> project's own measurements cut both ways. In favour: ppa has no node features at all, so the
> signal is structural to begin with and a structure-derived order is not displacing a semantically
> primary quantity — unlike ZINC, where 1-WL refinement lost 0.0048 precisely because it replaced
> atom type (correlating +0.82 with the endpoint chemistry) with a chemically arbitrary ordering.
> Against: a canonical labelling still imposes an arbitrary total order among structurally
> distinguished nodes, and BREC — the one place nauty demonstrably helps — has a task that *is*
> isomorphism distinguishing, whereas ppa's is classification. Neutral but real: δ also carries
> graph size, which attention pooling normalises away (measured `R²(δ ~ 1/graph_size) = 0.545` on
> ZINC), and ppa's size distribution is wide.
>
> **De-risking option before committing the ~1 CPU-hour and the full ranking**: `MalNetTiny`
> (5k function-call graphs, also featureless, used by GraphGPS and Exphormer) exercises the same
> nauty + `rank_mode: position` path at a fraction of the cost. If the path misbehaves there it
> will misbehave on ppa.

### 2. Node-level prediction

The node-level pipeline is implemented (transductive masked training loop, per-node head via
`readout=None`, precomputed canonical rank) and **the programme is closed**. `Questions` is tuned
and ranked against all ten baselines, `ogbn-arxiv` is tuned and carries the decisive gate
ablation, and `Cora` remains an unreported smoke test. The other three datasets, GraphSAINT
sampling and layer-wise inference are **dropped** — the finding is structural rather than
dataset-specific (LDNA's rank is a deterministic function of a node's own feature row here, so
the gate has nothing to condition on; § Results log), and reconfirming it would cost 50–200 GPU-h
per dataset. Loaders are from `torch_geometric.datasets` / `ogb.nodeproppred`.

The suite was originally five datasets in priority order; the sizes and axes below are kept
because they document what the dropped entries would have covered and why the two that ran are
enough. Sizes are from the PyG/OGB metadata.

| Dataset | Task | Metric | Dir | Evaluator | Status | Notes |
|---|---|---|---|---|---|---|
| `ogbn-arxiv` | 40-class node classification | accuracy | max | `OGBNodeEvaluator` | done | 169k nodes / 1.17M directed edges, symmetrized by `T.ToUndirected()` to 2,315,598 entries (a preprocessing choice shared by all models, since every model here aggregates over an undirected neighbourhood); OGB **time** split (train ≤2017 / val 2018 / test 2019) shipped as index tensors, converted to boolean masks by `_idx_to_masks` (90,941 / 29,799 / 48,603, a partition verified at load); the only dataset that could have run **both** full-graph and sampled, but the sampled control was dropped with the rest of the sampled programme. Tuned by a 100-trial search (best val 0.7040) and used for the decisive gate ablation; the 11-model ranking was deliberately not run |
| `ogbn-products` | 47-class node classification | accuracy | max | `OGBNodePropPredEvaluator` | dropped | 2.45M nodes / 61.9M edges; OGB `sales_ranking` split; neighbor sampling required. The scale anchor |
| `Questions` (`HeterophilousGraphDataset`) | binary node classification | ROC-AUC | max | `ROCAUCEvaluator` | done | 48.9k nodes / 154k edges / 301 features; **heterophilous** — the one non-homophilous entry; ships 10 official split masks (`train_mask` is `[N, 10]`), so splits map to `runs` the way EXP folds do (per-split shallow `Data` copies share x/edge_index/y/nrank; the fold list rides the train_loader as `cv_folds`); single BCE logit (`num_pred_heads=1`, float `[N, 1]` y). Tuned, ranked against all ten baselines (LDNA #3 — § Results log), and the site of the gate ablation (`configs/ldna_questions_{gated,ungated}.yaml`) |
| `PPI` | 121-task multi-label | micro-F1 | max | custom (micro-F1) | dropped | 24 graphs (20 train / 2 val / 2 test), ~2.2k nodes each; **inductive** and multi-graph, so it reuses the existing `DataLoader` path — only the head changes (no pooling). Carries the multi-label axis `ogbn-proteins` was to provide |
| `Reddit` | 41-class node classification | accuracy | max | `AccuracyEvaluator` | dropped | 233k nodes / **114.6M** edges (mean degree ~492) — the density stress case for `LDNAConv`'s edge-level MLP; neighbor sampling required. Weakest citability (only NAGphormer reports it, on a non-standard 60/20/20 split), so it buys the O(E)-at-scale claim rather than a comparison |

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
> Three conclusions. (1) **The rank gap is healthy at every scale** — median \|δ\| is 0.21–0.27 on
> Cora/Questions/arxiv, refuting the pre-implementation worry that δ collapses on large graphs;
> δ is a difference of percentiles between two edge endpoints, so it does not shrink with N.
> (2) **PPI's tie structure is not a red flag** (user ruling, 2026-07-29): the operative collapse
> metric is **P(δ=0) on edges** — the fraction of edges whose gate input is exactly zero — not the
> distinct-feature-row share. On P(δ=0), ZINC is the highest of all datasets (which is what
> motivates the ZINC WL-refinement check), and PPI's 15.6% sits *below* MNIST/molpcba/molhiv,
> graph-level datasets where LDNA already places top-2. PPI therefore runs through the normal
> protocol with no WL refinement or other special handling. (3) **`deg^L` is a bad estimator
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
> `ogbn-arxiv` was to be run **both ways** to quantify what sampling costs; the sampled arm was dropped with the sampled programme.

> **Measured full-graph memory on `ogbn-arxiv` (2026-07-29).** LDNA, 3 real training steps,
> `torch.cuda.max_memory_allocated` on an RTX A6000 (49 GB):
>
> | h | L | peak allocated | peak reserved |
> |---|---|---|---|
> | 128 | 3 | 18.33 GiB | 21.02 GiB |
> | 256 | 3 | 35.58 GiB | 36.27 GiB |
>
> Scaling is linear in width and **affine** in depth — a purely per-layer term misses the fixed
> encoder/feature cost and reads low, worst at shallow depth. Fit on the h=128 points (13.36 GiB at
> L=2, 18.33 at L=3): **allocated ≈ (3.42 + 4.97·L) · (h/128) · (edges / 2.32M)**, reproducing both
> exactly and staying ~3% conservative at h=256/L=3. The cost is dominated by `LDNAConv`'s per-edge
> activations over 2.3M edges rather than by the 0.2–0.7M parameters, which is why it is carried per
> edge entry and rescales to any node dataset. `mem_fraction` caps RESERVED memory, and the allocator
> overhead is **additive** — 0.42–2.69 GiB across the six measured configs, and *anti*-correlated with
> size (the 35.58 GiB run reserved 0.69 GiB extra, the 18.33 GiB one 2.69) because it is allocator
> fragmentation rather than a share of the tensor budget. So add the worst observed gap, **2.7 GiB**,
> rather than a percentage: a multiplicative margin over-predicts where it matters (a 1.2× factor gives
> 44 GiB at h=256/L=3 and would prune a config measured to reserve 36.27), while any additive value
> below 2.69 under-predicts the h=128/L=3 point above. The real card is 47.40 GiB, not the 49140 MiB
> nvidia-smi reports, so the 0.85 cap is a 40.29 GiB budget; that admits **L ≤ 7 at h=64, L ≤ 6 at
> h=128, L ≤ 3 at h=256** on arxiv — 6 + 5 + 2 = **13 of its 18 (h, L) points** — and all 15 of
> Questions'. The deeper and wider corner needs the sampled regime. Depth past ~L=4 is extrapolated
> from the L=2/L=3 fit, so h=128/L=7 lands just outside the budget and is pruned rather than left to
> OOM.

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

> **DGN's configs use the paper's ZINC aggregator set, not its MolHIV/PCBA one.** The four molecular/image
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
> → 6th), collapses a 2-D layout onto one axis (MNIST → 3rd, tied LDNA), or is one axis shared
> across 128 tasks (molpcba 3rd), and DGN reduces to a costlier PNA. The edge is bought with a
> hand-supplied per-dataset directional prior. **Adding DGN drops LDNA out of top-2 on ZINC**:
> ZINC now reads DGN 0.0507 / PNA 0.0807 / LDNA 0.0856 (LDNA 3rd). LDNA stays top-2 on the other
> four (molhiv 1st, code2 2nd, mnist 2nd, molpcba 2nd).

> **GIN xor GINE per dataset.** `GIN` and `GINE` are separate baselines, but only **one**
> runs per dataset — edge datasets use `GINE`, edge-less use `GIN` (never fabricate
> `edge_attr`). The rule is realized by the **config inventory**: each dataset ships only
> `gine_<suffix>.yaml` (edge) *or* `gin_<suffix>.yaml` (edge-less), never both — currently
> `gine_{hiv,molpcba,zinc}` and `gin_{mnist,questions}` (code2 → `gin`). `scripts/broadcast.py`
> globs `*_<suffix>.yaml`, so it and the ranking pick up whichever exists; its `HAS_EDGE` map
> selects the matching GINE/GIN template for the `GNN-VPA` config the same way.

> **`DGN` is excluded at node level — an IMPLEMENTATION-COST exclusion, NOT a structural one.**
> It is a weaker exclusion than `ogbg-ppa` and `ogbn-proteins` in § Datasets and must not be
> described as the same thing: those two are excluded because a canonical node order is
> *undefinable* without node features, whereas DGN's field is perfectly well defined here and it is
> only *our* implementation that takes a dense path. DGN is the one baseline needing a precomputed
> input this implementation cannot produce at scale: `data_utils/preprocess.py: add_eig_vec` materializes a dense `N × N`
> adjacency (`to_dense_adj`) and runs `eigh` on the Laplacian of every connected component. That
> is fine for graphs of tens of nodes and infeasible for one graph of 48,921 (a float32 `N × N`
> is 9.6 GB before the decomposition, and `eigh` on it is O(N³)); the resolver's `NODE_DATASETS`
> branch accordingly never calls `add_eig_vecs`, so a `dgn_<node>.yaml` would build a model whose
> directional aggregators receive no field at all. The exclusion is recorded the same way the
> GIN/GINE rule is — by the **config inventory**: node-level datasets ship no `dgn_*` config, so
> `scripts/broadcast.py` and the ranking simply do not see one. Node level therefore ranks LDNA
> plus **ten** baseline configs (`gcn`, `gin`, `sage`, `gat`, `gatv2`, `pna`, `egc`,
> `deepergcn_softmax`, `deepergcn_powermean`, `vpa`) rather than the eleven used at graph level.
> A sparse Lanczos field would be the fix if DGN is ever wanted here; § Baselines' DGN note above
> records why that path was measured and rejected once already on `ogbg-code2` (four settings whose
> failure modes are all silent). **Whether it is practical at 48,921 nodes is untested** — a sparse
> shift-invert `eigsh` was started during review and killed after ~6 minutes still factorizing, so
> "DGN cannot run here" is a statement about this codebase, not about DGN. Anyone asking why the one
> heterophilous dataset has no DGN row deserves that answer rather than the stronger-sounding one.

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
| `lr` | [1e-5, 1e-2] | log-uniform (node level: **[1e-4, 5e-2]** — see § Search spaces → node-level) |
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

**Pruning is enabled.** `MedianPruner` (`n_warmup_steps=10` at graph level; `round(0.2 × epochs)` at node level) + `trial.report` /
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
notes). Per-dataset overrides are documented in place: the batch-size table above and the node-level table below.

**Node-level search space** (`Questions`, `ogbn-arxiv` — the two wired node datasets). Same
`dropout` / `lr` / `weight_decay` ranges as above; `batch_size` does not exist (full-graph).
Everything else differs because one epoch is **one optimizer step**, not 130–680:

| Hyperparameter | Range / choices | Notes |
|---|---|---|
| `hidden_channels` | `Questions` {32, 64, 128, 256}, `ogbn-arxiv` {64, 128, 256} | per dataset because the floor is a **model** constraint: `LDNA` rejects `num_pred_heads > out_channels`, so arxiv's 40 classes rule out 32 while Questions' single BCE logit does not. Questions gained 32 after its first search chose the smallest width and depth on offer (64, 2) — a value on a bound cannot be distinguished from an optimum. 256 is the widest that fits arxiv full-graph |
| `num_layers` | `Questions` [2, 6], `ogbn-arxiv` [2, 7] | the measured k-hop covering depth + 2 (§ Datasets → node-level) |
| LR schedule | **none** | constant LR, per § Evaluation protocol; `ReduceLROnPlateau(patience=10)` is ten *steps* here |
| early stopping | **off** | `patience=20` is 20 steps (2% of the budget), and its calibration (2× the scheduler patience) is void with no scheduler |
| `epochs` | **no search cap** — a trial runs the full budget (`Questions` 1000, `ogbn-arxiv` 2000) | the cap only ever bounded wall-clock, and a full Questions trial is ~79 s. It also costs exactness: measured over six configs, rank correlation with the reported metric is 0.60 at a 333-step cap and **1.00 uncapped**, since a cap under-rates any config whose optimum lands past it and the measured optima span step 117 to step 559 |
| trial score | **best rolling median** (window 5) over the run, not the last-five median | the node runs report `Logger`'s best-over-epochs value, and with a constant LR the curve is not monotone. Measured on Questions (split 0, 1000 steps): h=128/L=3 at lr 3e-3 peaks at **0.742** near step 150 and decays to 0.609 by step 1000; h=256/L=5 at lr 1e-3 peaks at 0.730, also near step 150; h=128/L=3 at lr 3e-4 climbs slowly to 0.735 at step 800. The last-five rule bundles two assumptions — a median resists single-epoch noise (still true) and the last epochs are where the config settled (false here) — so the median is kept and only *where* it is taken moves. Measured rank correlation with the reported metric: last-five median **−0.20**, and still negative at windows of 25/50/100 (a decayed tail says nothing about the peak, and a slow config's tail is *higher*); rolling median **+1.00** uncapped, matching a raw max on ranking while a lone spike cannot move a five-point median |
| memory prune | `optuna.TrialPruned()` when the predicted full-graph peak exceeds `mem_fraction ×` the card | reserved peak ≈ (3.42 + 4.97·L) · (h/128) · (edges / 2.32M) + 2.7 GiB, affine in depth because a purely per-layer term missed the fixed encoder cost and read low everywhere. The allocator overhead is **additive** and anti-correlated with size (measured 0.42–2.69 GiB; the largest gap is on the *smaller* arxiv config), so it is carried at the worst observed value — a multiplicative margin would over-predict where it matters (a 1.2× factor gives 44 GiB at h=256/L=3, pruning a config measured to reserve 36.27) and anything under 2.69 under-predicts the recorded h=128/L=3 point. The estimate errs **conservative**: a prune and an OOM are not equivalent to the sampler, since TPE scores a pruned trial as worst and so steers away, while a `FAIL` trial is filtered out and teaches it nothing (measured: 55/100 vs 9/100 trials reaching the feasible point). Depth past ~L=4 is an extrapolation from the L=2/L=3 fit |

---

## Evaluation protocol

- **Search:** `hyperparam_search.py` scores a trial on a single run — at graph level by the
  median validation metric over the last 5 epochs; at node level by the best rolling median
  (window 5) over the run, because one epoch is one optimizer step there and the curve peaks
  early rather than settling (§ Search spaces → node-level). Use for finding LDNA's config.
- **Final ranking:** `train.py` runs `train_args['runs']` independent runs and reports
  per-run best train/val/test via `utils/logger.py` (mean ± std). Rank LDNA vs. all
  baselines using the mean test metric (respecting each metric's direction).
- **The reported metric is `utils/logger.py`'s `best_test` — the best test value over
  epochs**, not test at the best-validation epoch (that variant is present but commented
  out). Settled policy (user, 2026-07-29); **node-level reports the same way**. It is
  applied identically to every model on every dataset, which is what § Objective's
  relative-ranking criterion rests on. Do not re-derive committed results under another
  convention and do not re-open this.
- **Node-level training regime: constant learning rate, no LR schedule.** Full-graph
  transductive training runs **one optimizer step per epoch**, where a graph-level epoch runs
  130–680, so an epoch-counted schedule shared across the two levels means two different things.
  Inherited unchanged, `ReduceLROnPlateau` at PyTorch's defaults (`factor=0.1`, `patience=10`)
  collapses the LR from 1e-3 to 1e-7 in ~44 steps and training dies before it starts — measured,
  and it produced a Questions run at 0.59 ROC-AUC, *below* the graph-agnostic ResNet baseline
  (70.34) in the dataset's own paper. Node configs therefore **omit** `scheduler` (they must omit
  it, not set `scheduler: null` — configargparse delivers a YAML null as the string `'None'`, the
  same trap `readout` carries; `train.py: _lr_scheduler` rejects that string rather than honouring
  it by accident). This matches what the reference full-graph implementations actually do, verified
  by reading their source: OGB's `examples/nodeproppred/` has no scheduler anywhere; tunedGNN has
  none; DGL's arxiv GAT warms up then holds constant; and Platonov et al. construct a `LambdaLR`
  whose multiplier is identically 1 at the default `warmup_proportion=0`. Since the baselines'
  published numbers were produced in that regime, matching it is what makes comparison to them
  meaningful — and it removes a shared knob rather than adding three. Disclosed trade-off: a fixed
  constant LR is not per-model-neutral, but neither is a plateau scheduler, which fires on
  whichever model plateaus first and penalizes slow starters invisibly; genuine per-model
  adaptivity would require per-model LR tuning, which § Tuning methodology forbids.
  - Evidence base: **22 full-graph node-level implementations surveyed, 21 with no LR decay** —
    OGB's five node examples, Platonov, tunedGNN, Polynormer, SGFormer, NodeFormer, DIFFormer,
    LINKX, GCNII, SGC, APPNP, DeeperGCN, Kipf's GCN, GAT, PyG's examples, DGL's arxiv GAT
    (`grep -c 'scheduler|Cosine|Plateau'` is literally 0 in Polynormer's, SGFormer's,
    NodeFormer's and DIFFormer's `main.py`). The lone exception, the benchmarking-gnns arxiv
    port, carries `min_lr` and **breaks** the loop when the LR reaches it — a stopping criterion
    rather than an anneal, i.e. it had to neutralize this same failure mode.
  - **The rule is NOT "full-graph doesn't anneal, graph-level does" — that framing is false and
    should not be written down.** OGB's *graph-level* examples (mol, code2, ppa) also have zero
    schedulers, and SAN/benchmarking-gnns apply the same `ReduceLROnPlateau` to graph-level ZINC
    and to node-level SBM_PATTERN. The split is codebase lineage (OGB's minimal reference code /
    node specialists / the NLP-transformer line / benchmarking-gnns), and underneath it the real
    mechanism is **the unit the schedule counts in**: a schedule counted in *epochs* becomes
    meaningless when an epoch is one optimizer step, while one counted in *updates* is unaffected
    at any batch size (Graphormer's 60,000 warmup updates is well defined; `ReduceLROnPlateau`'s
    patience of 10 epochs is 10 steps at node level and 1300–6800 at graph level). Our bug was
    exactly an epoch-counted schedule meeting a one-step epoch.
  - Sharpest available test, and it is unanimous: **every graph transformer that runs full-graph
    node-level uses a constant LR** (Polynormer, SGFormer, NodeFormer, DIFFormer, tunedGNN's GT
    re-runs). The transformers that do anneal are precisely the ones not doing full-graph node
    work — GraphGPS ships 89 configs and **zero** node-level ones; Graphormer's examples are
    pcqm4m/zinc/hiv/oc20; NAGphormer is minibatch tokenized (batch 1000) and counts its decay in
    updates.
  - **Warmup is a separate mechanism from decay, and only warmup is attested here.** Two of the
    22 use it, both *without* any decay: DGL's arxiv GAT scales `lr × epoch/50` for 50 epochs then
    holds, and Platonov's LambdaLR is warmup-capable with the feature off by default. So
    warmup-then-constant is a live convention in full-graph node training and warmup-then-decay is
    not. If LDNA ever shows early-step instability, linear warmup over ~50 steps is the citable
    fix (DGL's arxiv GAT); it cannot reintroduce the failure mode, since it ends early and never
    drops the LR below base. Decay stays unsupported.
  - **Step budget: a cap, not a target.** No source justifies its budget or reports an
    accuracy-vs-steps curve for full-graph node classification, and the numbers do not scale
    with graph size (GCNII gives 2.7k-node Cora 1500 steps; OGB gives 169k-node arxiv 500).
    Full-graph budgets cluster at **500–2500, with 1000 the single most common value**. The only
    real correlate is that `lr × steps` is tighter within a dataset than either factor alone —
    evidence that each work tuned the pair jointly on validation, not a law (it breaks across
    datasets). So the budget only has to be long enough to contain the optimum, and the check
    worth reporting is that the optimum falls well inside it rather than at the boundary.
  - Reference protocols, for the record: **Platonov et al.** (`Questions`' own paper) — AdamW,
    lr 3e-5, weight decay 0, **1000 steps**, no schedule, no early stopping, 10 runs one per
    official split, 225 reproduction invocations with zero hyperparameter overrides.
    **OGB `ogbn-arxiv`** — Adam, lr 0.01, weight decay 0, 500 epochs, no scheduler, hidden 256,
    3 layers, dropout 0.5, BatchNorm. **tunedGNN** — Questions 1500, arxiv 2000.
    **Polynormer** — Questions 1700, arxiv 2000. **SGFormer** — 1000 with early-stop patience 200.
  - **`ogbn-arxiv` needs `runs: 10`.** The OGB rules require the mean and unbiased std to be
    taken over 10 random seeds (https://ogb.stanford.edu/docs/leader_overview/); fewer is not
    comparable to any published arxiv number. Config updated from 2.
  - **Our LR is validation-selected, not inherited — say so precisely.** Three official repos
    (Platonov, tunedGNN, Polynormer) all use **3e-5** on Questions, and tunedGNN's paper table
    saying 1e-3 is a lone outlier contradicted by its own script. Our probe selects 3e-3 on
    validation (val 0.7556 / test 0.7585 at 1000 steps, single split), which lands inside the
    published 76–79 band but gives an `lr × steps` product ~3 against their ~0.04. Their 3e-5 is
    entangled with their architecture (hidden 512, pre-norm residual blocks). The accurate claim
    is therefore "we follow their budget and selection rule, with the LR tuned on validation" —
    **not** "we follow Platonov's protocol".
  - **The shared `epochs` knob stops meaning one thing once sampling starts.** On `PPI`
    (inductive, 20 train graphs), `Reddit` and `ogbn-products` (both sampled), one epoch is many
    optimizer steps, so a single `epochs` value fixed across the node suite would fix wildly
    different step counts per dataset. Decide before those land: specify the node protocol in
    **optimizer steps** and derive `epochs` per dataset, or state plainly that full-graph and
    sampled datasets carry separate budgets. Reference budgets there: products Cluster-GCN
    50 epochs / 15000 partitions / batch 32; Reddit 10 epochs at batch 512–1024 (GraphSAGE, PyG)
    or 40 sampled epochs (GraphSAINT); PPI 100 epochs × 20 graphs ≈ 2000 steps (PyG).
- **Node-level & expressiveness evaluation paths are implemented**: the node evaluators in
  `utils/evaluator.py` (`OGBNodeEvaluator`, `ROCAUCEvaluator`, `MicroF1Evaluator`) and BREC's
  pairwise RPC protocol in `scripts/pairwise_eval.py` + `data_utils/brec.py`.
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
  **`out/logs/` has four folders, by what the artifact IS rather than by when it was produced.**
  `ranking/<dataset>/` — reported model-vs-model comparisons under the shared config, one folder
  per dataset. `search/` — hyperparameter searches; a permanent record, since the tuned config it
  produces is committed and cited by every ranking row. `ablation/<mechanism>/` — reported
  method-internal comparisons, where the arms are the same model with one component varied
  (currently `ablation/gate/`, the four gate-ablation runs across both node datasets; they belong
  together as one experiment rather than split by dataset). `explore/<topic>/` — probes,
  candidate comparisons and abandoned variants, i.e. work that may be discarded outright.
  Nothing is filed flat at the root once its run has finished.

  Two pre-existing exceptions are left as they are rather than moved, because their paths are
  cited in § Results log rows: `ranking/brec/gateinit_probe/` (a probe) and the two
  `*_no_canonical` BREC arms (an ablation), both under `ranking/brec/`.

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
  the two-machine setup memory (summarised in `.claude/project_state.md` § Graph-level experiment protocol).

---

## Guardrails

Mirror of `AGENTS.md` (§ Autonomous optimization); concrete limits here.

- Compute budget per optimization session: **100 Optuna trials/dataset** (the default).
  **Node-level wall-clock ceilings** (§ Guardrails requires one per search): `Questions` **~3 GPU-h** for 100 trials at its 1000-step budget (measured end-to-end from the
  retained search log; the earlier ~1 GPU-h estimate extrapolated from per-step cost and ignored
  per-trial setup, and only 9 of 100 trials pruned there); **`ogbn-arxiv` ~26 GPU-h** at its 2000-step
  budget (~0.87 s/step measured, ~29 min per full trial, with `MedianPruner` culling ~50% of total
  steps; 48 GPU-h unpruned). arxiv is now the most expensive single job in the project and needs a
  card held exclusively — `mem_fraction` is a self-cap, not a reservation, so 0.85 on a card already
  holding another user's work breaks the never-OOM-anyone rule.
  Three compute savers in the search. Two are **graph-level only** — the node path runs its full
  budget with no early stopping (§ Search spaces → node-level) — while `MedianPruner` is active on
  both levels — it culled ~50% of total steps on arxiv but only 9 of 100 trials on Questions: a low
  **epoch cap of 50** (vs 150 for final runs),
  `MedianPruner` (`n_warmup_steps=10` at graph level; `round(0.2 × epochs)` at node level) that kills *bad* trials early relative to others,
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
      evaluators. GraphSAINT sampling and layer-wise exact inference were dropped with the rest of the sampled programme (see the entry below).
- [x] `Planetoid` Cora — smoke test for the node path only; not reported.
- [x] Add `ogbn-arxiv` (OGB split + `OGBNodeEvaluator`; full-graph). Wired, tuned (100 trials)
      and used for the decisive gate ablation. **The sampled control was dropped** with the rest
      of the sampled programme below.
- [x] Add `Questions` (heterophilous; 10 official splits map to runs; `ROCAUCEvaluator`). Wired,
      tuned, and **ranked against all ten baselines** — the one node-level ranking (§ Results log).
- [x] ~~Add `ogbn-products`, `PPI`, `Reddit`; GraphSAINT sampling; layer-wise exact inference~~ —
      **dropped, not deferred.** The node-level question was answered by the two datasets above:
      LDNA's distinguishing mechanism is inert in this regime for a measured structural reason
      (`node_rank` ranks within one fixed global multiset, so the rank is a deterministic function
      of a node's own feature row and `H(δ | x_i, x_j) = 0`; § Results log). `Questions` and
      `ogbn-arxiv` already span heterophilous-binary and homophilous-40-class, and each remaining
      dataset would cost 50–200 GPU-h to reconfirm the same null. The sampling and layer-wise
      inference infrastructure existed only to reach those datasets, so it goes with them. Node
      level is reported as a measured boundary condition, not as a benchmark sweep.

**Graph-level expansion (planned 2026-08-04 — the suite's gaps are long-range, multi-target
regression, and featureless graphs; these four close all three):**

> **A criterion for judging any future graph-level dataset, from this project's own measurements.**
> The canonical rank is useful only when it carries information the message MLP does not already
> receive, and at graph level that holds structurally (the rank is computed within each graph, so it
> encodes graph context — `H(δ | x_i, x_j)` is 2.14 bits on ZINC, 4.49 on molhiv, 1.69 on MNIST,
> against 0.000 at node level). But the *quality* of that signal depends on the feature layout:
> `np.lexsort` makes the LAST feature column primary, so a single low-dimensional categorical
> column recodes a semantically primary quantity losslessly (ZINC's atom type, correlating +0.82
> with the endpoint chemistry), while a high-dimensional continuous embedding degenerates the order
> into the percentile of one arbitrary coordinate (measured at node level: `Spearman(rank, last
> column) = 1.000000` on arxiv, where that column is 123rd of 128 by variance). **Prefer graph-level
> datasets with low-dimensional, semantically meaningful node features** — which is why `Peptides`
> and `CIFAR10` are the safe additions, and why any future candidate carrying learned continuous
> node embeddings deserves this check before it is wired.

- [ ] **`CIFAR10` superpixels** — the smallest diff, do it first. `GNNBenchmarkDataset(name='CIFAR10')`
      in `utils/resolver.py`, reusing the `MNISTSuperpixels` branch verbatim (`T.Cartesian` +
      `RemoveEdgeAttr`, `sort_graphs`, `MNISTEvaluator`, `nn.Linear(num_features, hidden)` encoder).
      **Check its split**: MNIST has no validation set and the resolver silently aliases val to test
      (see its dataset row); confirm whether CIFAR10 does the same and disclose it if so. Edge-less
      after `RemoveEdgeAttr`, so **GIN**, not GINE.
- [ ] **`Peptides-func` and `Peptides-struct`** — `LRGBDataset(name=...)`, one resolver branch
      covering both since they share graphs and differ only in targets. Atom-level node features and
      bond `edge_attr`, so the existing feature lexsort and the `models.Encoder` / `nn.Embedding`
      encoder pattern apply as they do for ZINC — **GINE**. Two new evaluators: multi-label **AP**
      for `-func` (the OGB AP computation over 10 tasks) and **multi-target MAE** for `-struct`
      (mean over the 11 targets, normalized per the LRGB protocol — read the protocol, do not
      invent the normalization). `BCEWithLogitsLoss` and `L1Loss` respectively.
- [ ] **`ogbg-ppa` via nauty** — the only one needing a new mechanism. `PygGraphPropPredDataset`
      + `canonicalize_graphs` in place of `sort_graphs`, and LDNA reads the order via
      `rank_mode: position` (as on `ogbg-code2`). Four things to get right, each of which has
      already bitten this project once:
      1. **Cache the canonicalization.** ~1 CPU-hour otherwise repeated on every job start, and a
         ranking launches 11+ processes. Follow `add_eig_vecs`' sidecar pattern in
         `processed_dir`, keyed by a digest of the connectivity so a revised ordering rule
         invalidates it rather than silently misaligning.
      2. **Ordering constraint.** DGN's `eig_vec` must be attached AFTER canonicalization (which
         permutes nodes) and BEFORE any index slicing (an index view cannot be re-collated) — the
         same constraint the other OGB branches already document.
      3. **`rank_mode: position` is mandatory** for LDNA here, exactly as the ZINC `canonical`
         experiment found: a feature-mode rank over a canonically-ordered featureless graph is a
         silent no-op, not an error. Guard it the way that experiment did.
      4. **GINE, not GIN** — ppa has 7-dim `edge_attr`; do not strip it.

**Expressiveness (custom evaluation):**

- [x] Add `EXP` / `CEXP` (ported dataset + the paper's 10-fold CV; `AccuracyEvaluator`).
- [x] Add `BREC` (dataset + ported RPC pairwise evaluation; `data_utils/brec.py`, `scripts/pairwise_eval.py`).
- [x] ~~Extend `hyperparam_search.py` to expressiveness datasets~~ — resolved as **not needed**:
      batch-3 hyperparameters are protocol constants (see § Datasets → batch-3 notes). Node-level
      search support is wired for the two node datasets that exist (`Questions`, `ogbn-arxiv`),
      with its own space and step-counted regime (§ Search spaces → node-level).

---

## Experiment queue

Runs to do once the code is in place. Keep status current; move finished items to
the results log.

**Graph-level protocol** (per § Tuning methodology: tune LDNA → `scripts/broadcast.py` the tuned
shared config to all baselines → run → rank → append to § Results log): **complete on all
five datasets** — `ogbg-molhiv`, `ogbg-molpcba`, `ZINC`, `MNISTSuperpixels`, `ogbg-code2` —
plus the `DGN` baseline added afterward. Final rankings in § Results log.

**Graph-level expansion (planned 2026-08-04).** Per § Tuning methodology, each dataset runs the
standard protocol: tune LDNA → `scripts/broadcast.py` the tuned shared config to every baseline →
run → rank → append a § Results log row. Suggested order, cheapest and lowest-risk first:

- [ ] `CIFAR10` — pipeline already exists; the only work is the resolver branch and the configs.
- [ ] `Peptides-func`, then `Peptides-struct` — new evaluators, but the sort and encoders are the
      ZINC pattern. These are the two that close the **long-range** gap, so they carry the most
      weight for the paper and should not be cut first if the budget shrinks.
- [ ] `ogbg-ppa` — do last: it needs the nauty path, the cache, and the largest compute
      (158,100 graphs). Consider the `MalNetTiny` de-risking run first (see its note in § Datasets).

Two things to decide before running, not while running. **`batch_size` is a fixed per-dataset
power of two** chosen so steps/epoch lands ~100–300 (§ Search spaces) — set it from each dataset's
train size before the search, not after. And **DGN needs a directional field on each new dataset**:
the Laplacian eigenvector path applies to all four (all are small-graph, connected-component
decomposable), but check the disconnected-graph fraction as was done for molhiv (7.5%), since the
field vanishes on every edge of a graph whose lowest non-trivial eigenvectors degenerate into
component indicators.

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

**Node-level gate ablation — done, and it decided the programme.** The protocol was
pre-committed before running and is kept here because the result rests on it: two arms identical
except `rank_mode` (`none` forces `nrank=None` so `LDNAConv.message` skips the gate and every conv
is a plain sum-of-messages aggregator), verified single-factor by independent review — identical
parameter count, identical RNG stream, bit-identical per-run initial weights, so the arms compute
the same function at init and differ only in whether the gate can learn. Deciding metric, fixed in
advance: the reported test metric compared **paired** (by split on `Questions`, by run index on
`ogbn-arxiv`), the gate judged to contribute only if the mean paired difference exceeds its own
SEM. Stop condition, also fixed in advance: if the arms fall within SEM, stop and report rather
than spend the ranking compute.

- [x] **`Questions`** — run twice, at a hand-picked config and again at the tuned one, so a config
      error could not be the explanation. Null both times (−0.00121 ± 0.00183 and
      −0.00045 ± 0.00193).
- [x] **`ogbn-arxiv`** — the decisive one, run at the searched tuned config. Null
      (−0.00033 ± 0.00068) in a regime with generalisation headroom left (train–val gap 0.068
      against Questions' 0.243), while the paired TRAIN difference is significantly positive and
      validation significantly negative. Questions could not settle this alone because both arms
      there reach train ROC-AUC 0.9999 — saturation would have masked any architectural effect.
- [x] **Stop condition fired**, and the programme stopped as specified: the arxiv 11-model
      ranking and the three remaining datasets were dropped rather than run (§ Implementation
      queue). Numbers, the measured mechanism and the disclosures are in § Results log.

---

> **A checkpoint is NOT the reported model.** `train.py` saves once, after the whole run loop, so
> `out/checkpoints/<name>.pt` holds the FINAL epoch of the LAST run, while every reported number is
> best-over-epochs averaged across runs. On Questions the gap is enormous because the validation
> curve peaks near epoch 170-260 and then decays: `ldna_questions.pt` scores test ROC-AUC **0.5901**
> against the reported 0.7568, and `gin_questions.pt` **0.6242** against 0.7715. Anything that needs
> the reported model must retrain, not load — and any forensics done on a checkpoint describes a
> past-peak model, which is a caveat on the gate measurements below rather than a refutation of
> them, since they concern the gate's learned FORM rather than its accuracy.

> **EVIDENCE GAP: the graph-level case for the gate is a TWO-FACTOR comparison, and no
> single-factor graph-level gate ablation exists.** Commit `2278661` introduced the rank gate and
> **simultaneously removed `lin_aggr`** (a `Linear(in, in)` applied to every message before the
> sum). Parameter counts confirm it exactly: ZINC's old LDNA 2,419,074 against the rank-gated
> 2,018,562, i.e. −(65,792 × 7) + (8,576 × 7). So "the rank-gated aggregator beat its predecessor
> on all five datasets" compares **{gate, no lin_aggr}** against **{no gate, lin_aggr}** — it does
> not isolate the gate. `rank_mode: none` is set by node-level configs only; no graph-level config
> has ever run it. The node-level ablation IS single-factor, so the two regimes' evidence is not
> the same comparison, and the premise "the same mechanism works at graph level" is weaker than it
> reads. (The confound cuts in LDNA's favour on capacity: the winning arm has 400k FEWER
> parameters.) Note also that the place the gate is most strongly implicated — BREC 0 → 321 — has **no
> single-factor gate ablation either**: the available comparisons are LDNA-vs-GIN (different
> architectures) and zero-init-vs-gate-init (138 → 321), which varies the gate's *initialisation*,
> not its presence. And it uses `rank_mode: position` on a **nauty canonical labelling**, a structure-derived
> individualizing rank, not the feature rank that nulls at node level. The strongest evidence for
> the gate and the null result do not use the same rank source.
>
> **The run that would close this**: `rank_mode: none` against `feature` at the committed tuned
> configs on ZINC / MNIST / molhiv, 3+ runs, identical seeds — the node-level protocol applied at
> graph level. If null, the feature-rank gate never carried the graph-level win and BREC's
> position rank is the real mechanism; if positive, the regime difference stands as measured.
> **Not run** — recorded here so the gap is visible rather than discovered later.

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
| 2026-07-22 | ogbg-molhiv | LDNA (rank-gated) rerun | same tuned shared config as 07-10 molhiv row, runs=10, 150ep; baselines frozen | test ROC-AUC: LDNA 0.7923±0.0089 (SEM 0.0028) | **#1 (nominal) / tie** | vs old LDNA 0.7867±0.0118: +0.0056, which is **within noise** under § Objective's sum-of-SEM rule (0.0037 + 0.0028 = 0.0065 > 0.0056); the earlier 'modest real gain' reading used quadrature, which § Objective does not define. vs old #1 GAT 0.7875±0.0175 (SEM 0.0055): LDNA leads by 0.0048 < combined SEM 0.0062 → statistical tie, but LDNA takes the nominal top spot (was #2). |
| 2026-07-23 | MNISTSuperpixels | LDNA (rank-gated) rerun | same tuned shared config as 07-13 MNIST row, runs=3, 150ep; baselines frozen | test acc: LDNA 0.9410±0.0023 (SEM 0.0013) | **#2** | vs old LDNA 0.9357±0.0019: +0.0053 (> combined SEM 0.0017, real gain). Still behind #1 PNA 0.9532±0.0004 (SEM 0.0002): gap 0.0122 ≫ combined SEM 0.0013 → still highly significant. Closes ~30% of the old LDNA→PNA gap but PNA's per-node degree-scaler statistics remain out of reach on this dataset. |
| 2026-07-28 | EXP | LDNA + GIN + DGN (batch 3) | canonical order (nauty) + rank_mode position; shared h64/l8, attention, residual, 10-fold CV, 500ep, constant lr 1e-3; per-fold best-of-N selected on **val**, applied symmetrically — three full 10-fold sweeps (seeds 42/43/44) plus one extra fold-10 seed (`scripts/train_fold.py`) | test acc: LDNA **1.0000** (all 10 folds) | **#1 (tie with DGN)**; 1-WL ceiling broken | DGN 1.0000±0.0000 (seed 42; best-of-N monotone → no extra seeds needed); GIN 0.5000±0.0000 at **all three seeds** — the exact 1-WL ceiling. Fold 10 was the one fold still short of 1.0 under the three sweep seeds; its extra seed reached train/val/test 1.0 (logs `ranking/exp/ldna_exp_fold10.txt` + `_results.json`). |
| 2026-07-28 | CEXP | LDNA + GIN + DGN (batch 3) | same shared config, seed 42, 10-fold CV | test acc: LDNA **1.0000±0.0000** | **#1 (tie with DGN)** | DGN 1.0000±0.0000; GIN 0.7492±0.0026 ≈ the theoretical 75% (corrupt half 1-WL-learnable, EXP-hard half pinned at 50%). No stuck folds for LDNA on CEXP — the corrupt half's easy gradient reliably bootstraps the rank gate. |
| 2026-07-28 | BREC | LDNA + GIN + DGN (batch 3) | RPC protocol, shared 64×8, BN on, S+εI, 10 seeds × 400 pairs | pairs distinguished: LDNA **138 best / 122.8±9.2 mean** of 400 | **#2** of the 3 run; ≫ every 1-WL baseline (provably and empirically 0) | Original-run numbers, kept as companions — the official LDNA/DGN figures were set later by the gate-init and canonicalization-ablation rows below. GIN 0 (exact, all seeds); DGN **273 best / 271.4±1.0 mean** (#1, ≈ the published 3-WL/PPGN level ~270). LDNA by category at its best seed: Basic 47/60, Extension 66/100, 4-Vertex 13/20, Regular 12/100, DR 0/20, CFI 0/100 — the rank gate cracks the non-regular families, not CFI. Reliability failures 0 for every model/seed. **Recorded caveat:** DGN's Regular/DR sweep (99/100, 20/20) is partly a by-product of the shared canonicalization — a deterministic node order fixes the otherwise arbitrary degenerate eigenbasis, turning it into an isomorphism-invariant feature; this is shared preprocessing (identical for all models), not DGN-specific tuning, but it materially helps DGN beyond its own paper's setting and is disclosed as such (quantified by the canonicalization-ablation row below: unaided DGN 144.4). |
| 2026-07-28 | BREC (LDNA gate-init) | LDNA | `ldna_brec_gateinit.yaml`: `gate_init_std: 0.1`, chosen by the pre-committed per-category probe (σ ∈ {0.001, 0.01, 0.1} + σ=0 arm, 20 pairs × 6 families × 3 seeds → 69.0 / 98.0 / 106.0 vs 46.7 of 120); everything else the committed BREC setting | pairs distinguished: LDNA **321 best / 312.0±4.0 mean** of 400 | **#1** | **Official LDNA BREC number** (promoted by this follow-up). Per category at the best seed (100): Basic **60/60**, Regular **100/100**, Extension **100/100**, 4-Vertex **20/20**, DR **20/20**, CFI 21/100 — perfect on every family except CFI, the sole residual. (Means: Basic 59.7, Regular 100.0, Extension 99.5, 4-Vertex 20.0, DR 19.9, CFI 12.9±3.4.) Reliability failures 0 on all 10 seeds. **Disclosure:** `data_utils/brec.py` calls only `model.reset_parameters()` per pair, which does not reach the `nn.Linear(1, emb)` node encoder, so across the 400 pairs the encoder carries trained weights forward — the pairs are not independent initialisations. Left unchanged deliberately (the protocol is init-sensitive and re-running would move the committed number), recorded here so the figure is read correctly. Under the aligned best-of-seeds convention this exceeds every published entry — I²-GNN+SGT 304, N2-GNN 298, 3-WL 270, Edge Transformer 258.1 (a mean), Graphormer 79 — and the paper's aggregated Regular column at 140/140 is the highest anywhere (previous best N2-GNN 138). Reported alongside zero-init LDNA 138 best / 122.8±9.2 mean (identical protocol otherwise). Above DGN 273 (shared-canonicalization setting) and DGN 145 (its own construction). Logs `ranking/brec/ldna_brec_gateinit.{txt,_results.json}`. |
| 2026-07-28 | BREC (canonicalization ablation) | DGN (+ GIN control) | BREC rebuilt without the shared canonicalization — the benchmark's own raw-graph6 construction (`BRECDataset(canonical=False)`, `data/brec`, processed as `brec_v3_no_canonical.pt`); everything else at the committed 64×8/BN-on/S+εI/10-seed setting | pairs distinguished: DGN **145 best / 144.4±0.7 mean** of 400 (canonical: 273 best / 271.4±1.0 mean) | n/a (ablation; LDNA not involved) | **Official DGN BREC number** (the benchmark's own construction; promoted by this ablation). The −127 drop is confined to the degenerate families — Regular 99→43, 4-Vertex 20→0, DR 20→0, CFI 29.5→3.0 — while Basic is bit-identical (32.0±0.0) and Extension moves only 70.9→66.4: the signature of the arbitrary degenerate eigenbasis, not of the changed RPC regime. GIN control: exactly 0/400 with 0 reliability failures on all 10 seeds on **both** builds — the regime change alone produces no signal. DGN's reliability failures also 0 throughout (field inconsistency deflates T² rather than inflating it). Reading: DGN's directional aggregation is genuinely >1-WL unaided (144.4, still above LDNA's 122.8); the other ~127 pairs of its committed 271.4 come from the shared canonicalization turning the degenerate eigenbasis into an isomorphism-invariant feature. LDNA's 122.8 involves no spectral pipeline and is untouched by this ablation. |
| 2026-07-29 | all graph-level (DGN added as 11th baseline) | DGN | shared tuned config per dataset (each dataset's LDNA row); runs as before; molpcba 3 runs (seed 42 ×2 + seed 43 ×1) | DGN test: ZINC 0.0507±0.0001 (#1), code2 0.1876±0.0004 (#1), MNIST 0.9409±0.0012 (#3), molpcba 0.2727±0.0050 (#3), molhiv 0.7836±0.0110 (#6) | LDNA top-2 on 4/5; **3rd on ZINC** | DGN is dataset-selective — 1st only where a clean directional field meets an anisotropic task (Fiedler-vector molecular backbone on ZINC; exact AST root-to-leaf depth on code2), and mid-pack where the field fragments (molhiv, 7.5% disconnected), collapses a 2-D layout onto one axis (MNIST), or is one axis shared across 128 tasks (molpcba). Effect on LDNA: **code2 #1→#2** (0.1850, still ahead of every other baseline incl PNA 0.1844); **ZINC #2→#3** — ZINC now reads DGN 0.0507 / PNA 0.0807 / LDNA 0.0856. ZINC is simultaneously DGN's designed showcase and LDNA's weakest rank signal (a single atom-type node feature collapses the tie-aware rank to a coarse atom-type-rank gap), and the two methods ahead are both heavyweight molecular-regression specialists — DGN's hand-supplied precomputed spectral field, PNA's multi-aggregator + degree scalers. LDNA is first-or-second on four of the five graph-level datasets (3rd on ZINC) with no per-dataset directional prior or extra engineering — the two methods ahead of it there both carry one. It is 3rd on the one node-level dataset ranked. |
| 2026-07-2x | ogbg-code2 | LDNA + 10 baselines | tuned h256/l5 shared config, `rank_mode: position` for LDNA (AST DFS order is canonical); runs vary by config (2 for the full set, 1 for six re-run configs — see § Compute) | test F1: LDNA **0.1850** | **#2** | Recorded late — the ranking was archived under `ranking/code2/` without a results-log row. #1 DGN 0.1879 (n=1) on the AST-depth directional field, LDNA 0.1850 (n=2), then PNA 0.1844±0.0012, DeeperGCN-softmax 0.1804, DeeperGCN-powermean 0.1774, EGC/GATv2 0.1747, GNN-VPA 0.1739. Sample sizes are small and uneven, so read the ordering rather than the gaps. Four configs (gat, gcn, gin, sage) have `.txt` logs but no pickle — their runs were stopped before completion, which is why the archived field is 8 rather than 11. |
| 2026-07-2x | ogbg-molpcba | LDNA rank-gated rerun | same tuned shared config as the 07-17 molpcba row, runs=3; baselines frozen | test AP: **0.2795 ± 0.0039** (SEM 0.0022) | n/a (LDNA-internal) | Recorded late. **The rank-gated aggregator REGRESSED here**: 0.2795 against the pre-gate 0.2834 ± 0.0016 (SEM 0.0009) — a 0.0040 drop against a 0.0032 sum-of-SEM threshold, i.e. significant under § Objective's own rule. This is the one graph-level dataset where the rank gate made things worse, and it is why the "beat its predecessor on all five datasets" phrasing that stood in three places was wrong. It also costs the parameter argument nothing: the rank-gated model is *smaller* (31.6M against 38.7M). |
| 2026-08-02 | Questions | LDNA hyperparameter search (two rounds) + candidate selection | round 1: 100 trials over widths {64,128,256}; round 2: 50 trials with width 32 added and round 1's best warm-started via `--enqueue` | round 1 best val **0.7626** (h64/l2), round 2 best val 0.7581 (h256/l5) | n/a (tuning) | Round 1 chose the smallest width AND depth on offer, which cannot be told from an optimum without something below it — hence round 2 with width 32 added. It chose a near-opposite architecture for a *worse* validation score, so the architecture axis is flat here: both candidates were then run over the 10 official splits (`out/logs/explore/questions_capacity/`) giving test 0.7571 (h64/l2, 51,617 params) against 0.7575 (h256/l5, 1,149,953 params), a paired difference of +0.00038 ± 0.00312 — a tie at 22× the parameters. **h64/l2 was committed on parameter efficiency, not on score**, and its depth is still on the search bound. The three regularisation knobs agreed across both rounds (dropout 0.62/0.67, lr 4.9e-3/3.8e-3, wd 5.37e-6/5.30e-6), which is what a capacity-saturated dataset looks like from the search side. |
| 2026-07-29 | ZINC (subset) | LDNA — node-ordering explorations, both NEGATIVE | one run each on the 12k subset at the tuned ZINC config; arms differ only in how the canonical rank is derived | test MAE: feature lexsort **0.3305**, nauty canonical labelling 0.3314, 1-WL-refined rank 0.3353 | n/a (LDNA-internal) | Recorded late; logs under `out/logs/explore/zinc_node_order/`. ZINC has the highest P(δ=0) of any dataset (0.53 of train edges), so two alternative orderings were tried to break the ties. Both failed: WL refinement removed the ties (P(δ=0) → 0.008) and lost 0.0048; nauty removed them entirely (P(δ=0) = 0) and was level (+0.0009, inside the 0.0038 one-run noise). Measured reason the ties were not dead weight: on ZINC `P(δ=0 | bonded atoms share an element) = 1.00` exactly and `corr(|δ|, endpoints differ in atom type) = +0.82`, so individualising them trades a crisp chemical indicator for a chemically arbitrary ordering. **ZINC keeps the plain feature lexsort and the `wl_steps`/`canonical` code paths were removed**, so this row is the only record. |
| 2026-08-02 | Questions (node-level) | LDNA + 10 baselines | tuned h64/l2/do0.616/lr4.87e-3/wd5.3e-6, constant LR (no scheduler), 1000 steps, runs=10 = the 10 official split columns; DGN excluded (see § Datasets → node-level) | test ROC-AUC: LDNA 0.7568±0.0111 (SEM 0.0035) | **#3 / significantly behind** | #1 GIN 0.7715±0.0127, #2 GNN-VPA 0.7632±0.0118, then LDNA, GCN 0.7540, EGC 0.7526, GAT 0.7437, GATv2 0.7434, SAGE 0.7364, PNA 0.7361, DeeperGCN-powermean 0.7241, DeeperGCN-softmax 0.7144. **Top-2 NOT met.** Unpaired: GIN's lead 0.0147 > combined SEM 0.0075. Paired by split (the right test — the 10 runs are shared official splits, not independent seeds): GIN +0.0147±0.0032 winning 9/10, VPA +0.0065±0.0022 winning 8/10. LDNA is still significantly ahead of 7 baselines and tied with GCN. **The two GIN-kernel models take the top two spots** and `corr(gin, vpa) = +0.847` across splits against +0.397 for GIN vs every non-GIN-kernel model, so the kernel — not the tuning — is what separates them; swapping GIN's sum for VPA's `sum/√N` alone costs 0.0083, i.e. ~57% of GIN's lead over LDNA. Every model but GAT/GATv2 exceeds train ROC-AUC 0.999, so this dataset is decided by regularisation, not capacity — which is why the heavyweights (PNA, both DeeperGCN) are at the bottom. **Disclosures:** absolute values (0.714–0.772) are NOT comparable to the published 76–79 band, because the shared-hyperparameter protocol forces h=64/lr 4.87e-3 where the references use h=512/lr 3e-5 — and an all-models-underfit regime may systematically favour simple models; `models/gat.py` and `gatv2.py` pass the shared `dropout` into `GATConv` as *attention* dropout on top of the layer dropout, which at 0.616 zeroes 61.6% of attention coefficients (suite-wide convention, not fixable through configs); `models/sage.py` defaults to `max` aggregation where the published SAGE number uses `mean`. |
| 2026-08-02 | ogbn-arxiv (node-level) | LDNA hyperparameter search | 100 Optuna trials, constant LR, full 2000-step budget per trial, `mem_fraction` 0.85 on a held card | best val accuracy **0.7040** (h128/l3/do0.487/lr1.18e-3/wd1.58e-5) | n/a (tuning) | 39 completed / 61 pruned (8 of them memory-pruned at the wide-deep corner: h256/L4 predicted 49.3 GiB, h128/L7 40.9 GiB, both above the 40.29 GiB budget); 0 divergences. **The optimum is interior on all five axes and was found at trial 15, with no improvement over the remaining 85** — so this is the method's level here, not a truncated search. It does not beat a hand probe (val 0.7062 at h128/L3, wd 0, 2000 steps), which says the region is flat. Published full-graph arxiv is **71.5–73.6%** test accuracy; we sit 1.5–3 points below. |
| 2026-08-02 | Questions + ogbn-arxiv | LDNA gate ablation (`rank_mode: none`) | identical arms except whether `LDNAConv`'s rank gate engages; identical parameter count, RNG stream and per-run initial weights; paired by split/run | **arxiv** test acc: gated 0.6945±0.0021 vs ungated 0.6948±0.0018; **Questions** test ROC-AUC: gated 0.7569±0.0118 vs ungated 0.7573±0.0140 | n/a (LDNA-internal) | **The rank gate contributes nothing at node level.** arxiv paired diff **−0.00033 ± 0.00068 SEM**, 5/10 runs; Questions **−0.00045 ± 0.00193**, 3/10 splits (and −0.00121 ± 0.00183 on an earlier hand-picked config). arxiv is the decisive one: its train–val gap is only 0.068 against Questions' 0.243, so there was generalisation headroom for the gate to exploit and it did not. **The gate is demonstrably working, just not useful** — the paired TRAIN difference on arxiv is **+0.00683 ± 0.00131, significant** (9/10 runs): the gate fits the training set better (0.7709 vs 0.7641). Validation is **significantly WORSE by this project's own SEM rule** — paired **−0.00088 ± 0.00061**, 3/10 runs — so the gate does not merely fail to help on arxiv, it costs a little; test is the null (−0.00033 ± 0.00068). That is added capacity without a useful inductive bias, and it rules out the gate simply never engaging. Contrast with graph level, where the rank-gated aggregator improved four of the five datasets — and **regressed significantly on molpcba** (AP 0.2834 → 0.2795, a 0.0040 drop against a 0.0032 sum-of-SEM threshold) — and reaches BREC 321 — **but see the two-factor caveat below, which weakens that contrast.** |
| 2026-08-03 | Questions | kernel comparison — diagnostic, retrained on CPU for per-node logits | 4 models × 3 splits reproduced (GIN 0.7691 / VPA 0.7632 / LDNA 0.7557 / GCN 0.7536 against the reported 0.7715 / 0.7632 / 0.7568 / 0.7540), because the checkpoints hold past-peak weights | n/a | n/a | **Why LDNA is 3rd, given the gate contributes nothing.** The dataset is degree-driven: **median degree 1** (53.1% of nodes), positives are 2.98% of nodes and structurally distinct (mean degree 18.9 vs 5.9, positive rate rising monotonically 0.020 → 0.111 with degree). A logistic regression on **features + log-degree alone reaches test AUC 0.7490** — within 0.008 of LDNA and above six of the eleven GNNs; degree alone gives 0.6562, and that one scalar beats the entire 301-dim neighbour-mean vector. **The property that predicts the ranking is whether a kernel preserves neighbourhood SIZE**: Spearman +0.888 (p=0.0006) against the ordering, and the measured first-conv norm ratio (deg≥33 / deg=1) separates the top five (2.71–39.28) from the bottom five (0.93–1.12) with **zero overlap**. LDNA's raw sum puts it in the right tier, which is why it is 3rd and not mid-pack. **The dilution hypothesis is REFUTED** — there is no monotone degree trend in the GIN−LDNA per-bin AUC gap, and the high-degree bins where dilution predicts the largest deficit show none. LDNA's self-term is not diluted but *amplified* in lockstep with the neighbour term, so its share is pinned near 0.58 at every degree. At this config (`num_pre/post/trans_layers: 1`) every one of these convs is exactly affine, giving `GIN: W[(1+ε)x_i + Σx_j] + b` against `LDNA: deg_i · W_post[W₁x_i + b₁ + W₂x̄_i] + b_post` — **degree multiplies all three terms including the bias**, a structural factor no weight choice removes, so LDNA has no degree-invariant ego path where GIN has one and used it (trained ε = +0.5048). Also refuted: overfitting speed as an explanation (Spearman +0.073; GIN is the *slowest* top model to memorise), and heterophily-specific machinery (ego/neighbour separation is *anti*-predictive here — SAGE and LDNA both have it, at 8th and 3rd). Questions' heterophily is real only in the adjusted sense (edge homophily 0.840, adjusted 0.079): it lives entirely in the 3% minority class, whose neighbours are 89% negative. Independent corroboration of the gate null from the signal side: δ on a real edge is indistinguishable from δ between two random nodes (mean \|δ\| 0.2362 vs 0.2340), and \|δ\| predicts whether an edge joins differently-labelled nodes at AUC 0.4921 — chance. |
| 2026-08-03 | node vs graph level | LDNA — diagnostic analysis, no training | forensics on the trained gates plus information-theoretic measurements of δ; no model, config or architecture changed | n/a | n/a | **Why the gate is null at node level, measured.** (a) **δ is REDUNDANT at node level and informative at graph level.** `H(δ \| x_i, x_j)` — what the gate learns that `pre_nn` does not already receive, since `pre_nn` takes `cat([x_i, x_j, edge_attr])` — is **0.000 bits** on both node datasets against **2.14 bits (ZINC), 4.49 (molhiv), 1.69 (MNIST)**. Structural, not sampling: `node_rank` ranks each row within ONE fixed global multiset, so `nrank_i = g(x_i)` and δ = g(x_j) − g(x_i) is a deterministic function of pre_nn's own input, whereas `_canonical_rank` ranks WITHIN each graph and divides by graph size, so the same atom in two molecules gets different ranks. Attention pooling is softmax-normalised and therefore size-invariant, which makes δ the model's **only** channel to graph size at graph level — measured `R²(δ ~ 1/graph_size) = 0.545` (ZINC), 0.483 (molhiv), 0.001 (MNIST, where every graph has 75 nodes). (b) **Cardinality**: the gate learns over 51–2,105 distinct δ values with 10³–10⁵ edges each at graph level, against 134k–721k values with ~3 edges each at node level — a memorisation substrate, which is exactly the observed train-up/val-down signature. (c) **The order keys on one arbitrary coordinate**: `np.lexsort` makes the LAST feature column primary, and `Spearman(nrank, last column) = 1.0000` on both node datasets; on arxiv that column ranks 123rd of 128 by variance. Graph level has no such ambiguity — ZINC and MNIST have a single feature column, so the rank IS the atom type / superpixel intensity, correlating 0.93 with the endpoint feature gap. Scripts and a regenerated, retained output for the load-bearing redundancy claim are under `out/logs/explore/node_level_diagnostics/` (`redundancy_check.txt` verifies `nrank_i = g(x_i)` by exact row identity — zero violations on both datasets — and reproduces the δ statistics, the Spearman-with-last-column and the column's variance rank). The gate-forensics and kernel figures in this row and the one below come from agent analyses whose scripts are preserved in the same folder but whose outputs were not retained at the time; regenerate before quoting them anywhere load-bearing. (d) **The gate is not degenerate**: only 9–18% of its deviation on arxiv's first two layers is the constant absorbable into `post_nn`, and it fits `[1, δ, \|δ\|, sign δ]` at R² ≥ 0.998 with 80–98% of the varying variance in the **sign(δ)** term — i.e. it becomes a directional aggregator along the canonical order, structurally like DGN's but with that order as its field. It moved FURTHER from identity than any graph-level layer measured. Task relevance: `I(\|δ\|; endpoints-share-a-label)` = 0.00054 bits on arxiv against 0.930 base entropy, while feature-cosine gives 19× and degree-gap 14× more — the test has power and δ fails it. |

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
