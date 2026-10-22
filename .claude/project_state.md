# Project State

## Current focus
- Maintain a compact experiment runner for LDNA and baseline GNN models.
- Keep the repository understandable through agent-facing documentation.
- Optimization objectives, search spaces, experiment queue, and results are tracked in `.claude/experiments.md`.

---

## Current progress
- The repository is a usable experiment runner (not a packaged project): `train.py`
  (training loop), `configs/` (YAML experiments), `models/` (LDNA — `ldna_conv.py` /
  `ldna_net.py` — plus the baseline wrappers), `data_utils/` (sorting/canonicalization,
  transforms, dataset classes), `utils/` (resolver, evaluators, logger, tee), `scripts/`
  (operational helpers), and `requirements.txt`.
- `hyperparam_search.py` tunes LDNA on all nine graph-level datasets (ogbg-molhiv,
  ogbg-molpcba, ZINC, MNISTSuperpixels, ogbg-code2, CIFAR10, Peptides-func, Peptides-struct,
  ogbg-ppa) and on the two wired node-level ones (Questions, ogbn-arxiv). Search mechanics — ranges, pruning, plateau early-stopping, the
  50-epoch search cap (vs 150 final), and gradient clipping — live in
  `.claude/experiments.md` (§ Search spaces), together with the node-level space, whose
  epoch-counted settings all differ because a full-graph epoch is one optimizer step: no LR
  schedule, no early stopping, no search cap (a trial runs the dataset's full budget), a trial
  scored by the best rolling median rather than the last-five median, and a memory-feasibility
  prune on the (width, depth) box.
- `readout` (incl. `attention` = `AttentionalAggregation`) and a per-layer `residual` flag
  are shared knobs supported by every model family (`residual: true` everywhere; every
  graph-level and expressiveness config uses `readout: attention`, while the node-level configs
  carry no `readout` key at all — the resolver sets `readout=None` for them); the resolver unifies every model's encoder width to `hidden_channels`, so
  the conv stack is uniform. Projection-skip and `DeeperGCN`-`res+` details are in
  `.claude/experiments.md` § Tuning methodology.
- Both `train.py` and `hyperparam_search.py` tee each run under `out/logs/` (`utils/tee.py`)
  and expose an optional `--mem_fraction` per-process GPU cap; all outputs (logs + checkpoints)
  live under `out/`. Search OOM-tolerance and the `--mem_fraction` policy (a search that owns
  its card runs uncapped) are in `.claude/experiments.md` § Compute & parallelism; log
  archival is in § Evaluation protocol.
- LDNA aggregation is rank-gated: `LDNAConv` builds each message from `[x_i, x_j, edge_attr]`
  via an MLP and scales it by a learned gate `1 + gate_nn([δ, |δ|, sign(δ)])` of the
  canonical-rank gap δ = nrank_j − nrank_i (zero-init output layer → identity at init;
  an optional `gate_init_std` starts it slightly off identity, synthetic-expressiveness only).
  `LDNA` computes the per-node rank under a `rank_mode` knob: `feature` (tie-aware dense rank
  over the lexsorted feature order; the default) or `position` (intra-graph position for
  datasets whose node order is canonical by construction — code2's AST DFS order; used by
  `configs/ldna_code2_rankgate.yaml`). This rank-gated design is the current official LDNA —
  the outcome of the escalation, superseding the earlier plain summed-linear-map aggregator;
  it was rerun with the previously tuned shared params on all five graph-level datasets, where it
  improved four (ZINC 0.0903→0.0856, molhiv 0.7867→0.7923, MNIST 0.9357→0.9410, code2
  0.1765→0.1850) and **regressed significantly on molpcba** (AP 0.2834→0.2795, a 0.0040 drop
  against a 0.0032 sum-of-SEM threshold). Results in `.claude/experiments.md` § Results log.
- Baselines: `GCN`, `GIN`, `GINE`, `GraphSAGE`, `GAT`, `GATv2`, `PNA`, `EGC`, `DeeperGCN`,
  `GNN-VPA` (`models/vpa.py`), and `DGN` (`models/dgn.py`). `GIN` xor `GINE` runs per dataset by
  edge availability. Per-model backbones, `DGN`'s directional aggregators, and the GIN/GINE rule
  are in `.claude/experiments.md` § Baselines.
- DGN is wired for all nine graph-level datasets. Its per-node directional field is attached by
  the resolver only when the model is `DGN` and forwarded by `train.py` only when present, so
  every other baseline's call path is unchanged; the field is the low-frequency Laplacian
  eigenvectors on the molecular/image datasets (molhiv, molpcba, ZINC, MNIST, Peptides, ppa), the
  AST depth on `ogbg-code2`, and the **superpixel coordinates** on `CIFAR10`, whose directed 8-NN
  connectivity makes the eigenvector path silently wrong (`torch.linalg.eigh` reads one triangle,
  so it would decompose a matrix other than the one it was handed; `add_eig_vec` now raises on an
  asymmetric `edge_index` rather than returning a plausible field). That substitution is the DGN
  paper's own for CIFAR10, but it is also the one place a baseline receives input the other eleven
  configs do not — see the CIFAR10 results-log row. Its aggregator sets follow the paper's own
  per-dataset choices minus the members PNA also uses; the one knowing exception, `ogbg-molpcba`,
  is recorded in § Results log. Field construction, caching, measurements, and rejected
  alternatives are in `.claude/experiments.md` § Baselines.
- `ogbg-code2` is wired as an edge-less (GIN-family) dataset with a dedicated
  sequence-prediction path (`ASTNodeEncoder` + `Code2Head` in `models/code2.py`, plus a guarded
  `train.py` / `hyperparam_search.py` fork); the feature lexsort is skipped because the AST DFS
  order is already canonical. `ogbg-ppa` was out of scope for exactly that reason — featureless
  nodes make the feature-based sort non-permutation-invariant, which LDNA's MLP aggregator
  requires (the sort is a correctness requirement, not a performance lever) — and was
  **readmitted 2026-08-04** once a featureless order was available: `canonicalize_graphs`
  (pynauty, disk-cached, 2.87 ms/graph) supplies it and LDNA reads it through `rank_mode:
  position`, which `utils/resolver.py` *requires* on ppa and raises otherwise, since a
  feature-mode rank over a constant node feature ties every node and turns the gate into a silent
  no-op. Both are covered in `.claude/experiments.md` § Datasets.
- Node-level (transductive, per-node) prediction is wired end to end, without sampling or
  layer-wise inference: `data_utils/preprocess.py: node_rank` / `add_node_ranks` compute the
  tie-aware canonical rank **without permuting the graph** (the rank is permutation-equivariant,
  so the per-edge gap δ and the scatter-sum are invariant regardless of node labelling; the
  physical sort exists only to make `_canonical_rank`'s adjacent-row trick valid on a batch).
  `LDNA.forward` takes an additive `nrank=None` kwarg mirroring DGN's `eig_vec`; `train.py` gains
  `_nrank_kwargs` / `_split_mask` and applies the split mask **before** the `is_labelled` NaN
  guard (that guard is molpcba's missing-label filter, not a split selector); the resolver adds a
  `NODE_DATASETS` branch, a `FullGraphLoader`, and a single `readout` guard rejecting an
  unrecognized value and each direction of a node-vs-graph-level mismatch. The guard judges the
  value the **config** supplied, captured before the node branch overwrites it — checking the
  post-update value would pass any node config, since that branch sets `readout=None`
  unconditionally. `readout=None` is set by the resolver, not by YAML, because configargparse
  delivers a YAML `null` as the string `'None'`. No baseline
  model file was touched — their readout chains are `if`/`elif` with no `else`, so `readout=None`
  already yields per-node output. Evaluators added: `OGBNodeEvaluator`, `ROCAUCEvaluator`,
  `MicroF1Evaluator`; `AccuracyEvaluator` gained a `.view(-1)` fixing an `[N]`-vs-`[N,1]`
  broadcast. Wired so far: `Cora` (`configs/ldna_cora.yaml`, an unreported smoke test) and
  `Questions` (heterophilous, binary ROC-AUC on a single BCE logit; its 10 official `[N, 10]`
  split-mask columns become 10 per-split shallow `Data` copies sharing x/edge_index/y/nrank,
  riding the train_loader as `cv_folds` the way the EXP/CEXP folds do). `rank_mode: none` is the
  ungated LDNA ablation (the gate module exists but never engages — every conv is a plain
  sum-of-messages aggregator), used by the gate-ablation configs at node level, at graph level
  and on the expressiveness benchmarks.
  `ogbn-arxiv` is wired full-graph: OGB ships split INDEX tensors, so `utils/resolver.py:
  _idx_to_masks` turns them into the boolean `[N]` masks the rest of the node path reads and
  rejects a non-partitioning split on the spot; `T.ToUndirected()` symmetrizes the directed
  citation edges as a preprocessing choice shared by every model (169,343 nodes / 2,315,598
  undirected edge entries); `y` is flattened to `[N]` for the CrossEntropy path. Measured peak
  GPU memory for full-graph LDNA bounds the `(hidden_channels, num_layers)` box the node-level
  search may sample; the fitted law and its measurements are in `.claude/experiments.md`
  (§ Datasets → node-level).
- `Questions` has its baseline config set: ten `<model>_questions.yaml` files carrying LDNA's
  tuned shared knobs from `configs/ldna_questions.yaml` (h=64, L=2, dropout 0.616143,
  lr 4.871103e-3, wd 5.3e-6), broadcast by `scripts/broadcast.py`, which learned the dataset via
  `SUFFIX`/`HAS_EDGE` and needed no node-specific special-casing (it never touches `batch_size`,
  `readout` or `scheduler`). Node-level shape: no `readout` key, no `scheduler` key, no
  `batch_size`, `runs: 10` matching the ten official split columns. `GIN` (not `GINE`) since
  Questions is edge-less; **`DGN` is excluded at node level** — `add_eig_vec` materializes a
  dense `N × N` adjacency, infeasible at 48,921 nodes, and the `NODE_DATASETS` branch never
  attaches the field — so node level ranks LDNA plus ten baselines, not eleven. All eleven
  configs verified to build and run a forward+backward producing `[48921, 1]`.
- **Node-level result (2026-08-02): the rank gate does not transfer.** Questions is fully wired
  and ranked (LDNA **#3 of 11**, significantly behind GIN and GNN-VPA); ogbn-arxiv is tuned
  (100 trials, best val 0.7040, optimum interior on every axis and unimproved for 85 trials).
  The gate ablation (`rank_mode: none`, identical parameter count / RNG stream / per-run initial
  weights) is **null on both datasets**, and on arxiv it is null where the train–val gap is only
  0.068 — i.e. with generalisation headroom left. The paired TRAIN difference there is
  significantly POSITIVE (+0.00683 ± 0.00131) while validation is significantly WORSE
  (−0.00088 ± 0.00061): the mechanism runs and adds fitting capacity, buys no generalisation,
  and costs a little.
  **Why, measured (2026-08-03):** `H(δ | x_i, x_j)` is **0.000 bits** at node level against
  2.14/4.49/1.69 at graph level — `node_rank` ranks within one fixed global multiset, so δ is a
  deterministic function of the message MLP's own input, whereas the graph-level rank is computed
  within each graph and so encodes graph context. LDNA's 3rd place is then a kernel property, not
  a gate one: `Questions` is degree-driven (median degree 1; features + log-degree alone reach
  0.7490) and what predicts the ranking is whether a kernel preserves neighbourhood size
  (Spearman +0.888). Three hypotheses were tested and refuted — self-dilution growing with
  degree, overfitting speed, and heterophily-specific machinery. Full numbers and disclosures in `.claude/experiments.md` § Results log.
  This is a contrast with graph level, where the same aggregator improved four of the five
  datasets (and regressed on molpcba) and reaches BREC 321 — though that comparison was
  two-factor; isolated single-factor, the gate helps on ZINC and is null on MNIST and molhiv,
  and without it BREC falls to 0 (below). The arxiv 11-model ranking (~50 GPU-h) is **not** run: Questions
  already gives the node-level ranking picture and the ablation explains it.
- `train.py: _reset_encoders` re-initializes the node/edge encoders at the start of each run —
  model `reset_parameters()` covers the conv stack, norms, readout and head but not the encoders —
  so runs are independent for LDNA and every baseline alike, applied in the caller with no
  baseline model file touched. It reproduces each encoder's construction-time init distribution
  (`Encoder`/`ASTNodeEncoder` define Xavier `reset_parameters`; OGB's `AtomEncoder`/`BondEncoder`
  are re-initialized explicitly since the ogb package defines no reset method).
  `scripts/train_fold.py` was already clean (it rebuilds per seed). `data_utils/brec.py` carries
  the same gap and is **deliberately left alone**: BREC's committed 321 is a headline result, its
  protocol is init-sensitive (that is the whole point of `gate_init_std`), and changing the code
  without re-running would leave that number unreproducible. Revisit only if BREC is ever re-run.
- A **validation-selected test** column (test at the best-validation epoch) sits alongside the
  reported `best_test`, computed offline from the saved curves by `scripts/val_selected.py`, for
  every graph-level ranking, Questions and the node-level gate ablation. LDNA's rank is unchanged
  except on molhiv (#1 → #5, inside a five-way statistical tie); on Questions it becomes tied for
  #2, and on ogbn-arxiv the gate's paired effect turns significantly negative. Details in
  `.claude/experiments.md` § Results log.
- **Single-factor graph-level gate ablation (2026-10-21).** Commit `2278661` added the rank gate
  and removed `lin_aggr` in one change, so the earlier graph-level comparison was two-factor.
  Config pairs `ldna_{zinc,mnist,hiv}_{gated,ungated}.yaml`, identical except `rank_mode`,
  paired by seed, deciding metric `best_test` fixed in advance: the gate **helps on ZINC** (MAE
  0.0186 lower with the gate, ± 0.0052 SEM, 5/5 seeds, paired t-test p = 0.024) and is **null on
  MNIST** (accuracy −0.0009 ± 0.0011) and **on molhiv** (ROC-AUC +0.0051 ± 0.0055; the
  val-selected column leans positive, +0.0161 ± 0.0119, p = 0.21). MNIST's earlier two-factor
  gain did not come from the gate; by elimination (unpaired, across experiments) it is
  attributable to removing `lin_aggr`. Whether δ carries graph size (R² 0.545 ZINC, 0.001 MNIST)
  fits two of the three outcomes, not molhiv (0.483). The rankings are unaffected — they compare LDNA against baselines
  whatever is inside LDNA — but the gate can be named as the mechanism only per dataset. Details
  in `.claude/experiments.md` § Results log.
- **Single-factor expressiveness gate ablation (2026-10-21).** `ldna_{exp,cexp,brec}_ungated.yaml`,
  each its gated twin with only `rank_mode: none`, same nauty labelling and protocol, prediction
  fixed in advance. Without the gate LDNA falls to the 1-WL ceiling, as predicted and on GIN's
  numbers: EXP 1.0000 → 0.5000 (all 30 folds over seeds 42/43/44), CEXP 1.0000 → 0.7500, BREC
  321 → 0 of 400 at every seed. The labelling is the source of the beyond-1-WL signal and the gate
  is the only part of LDNA that reads it. EXP test is 0.5000 at every epoch; three single-epoch
  train/val values are off 0.5 by exactly one pair, in both directions, attributed to float32
  rounding. Details in `.claude/experiments.md` § Results log.
- Two torch≥2.6 / PyG 2.7 compatibility shims (needed to load MNIST/ZINC/OGB):
  `data_utils/transforms.py` `__call__`→`forward`; `data_utils/__init__.py` `torch.load`
  `weights_only=False`.
- Expressiveness result (established; write-up lives in the git-ignored paper-prep area,
  not in project files). Proof idea on record:
  - Claim: LDNA whose canonical rank comes from a canonical labelling (isomorphism-
    invariant, tie-individualizing node order) is strictly more expressive than 1-WL.
  - ⪰ 1-WL: with the rank gate frozen at identity the layer is a plain sum-of-messages
    aggregator; standard GIN-style injective simulation (finite domains, sum-injective
    encodings, self-argument in the update).
  - Strictness: the gate reads the canonical-rank gap δ (via the [δ, |δ|, sign δ]
    triple, which makes |δ| linearly available). On C_{2m} vs C_m⊔C_m (uniform
    features — 1-WL-equivalent) with cyclic vs consecutive-block canonical labellings,
    the per-node integer gap sums are {2m,2m,2,…} vs {m,m,m,m,2,…}: a gate g = 1+|δ|
    plus injective update/readout separates them.
  - Necessity: a rank computed from any 1-WL-invariant node quantity folds into the
    endpoint colours (model collapses to exactly 1-WL); on vertex-transitive 1-WL-hard
    pairs (Rook 4×4 vs Shrikhande, CSL) every isomorphism-invariant per-node quantity
    ties all nodes, so δ ≡ 0 — only genuine canonical individualization helps, and it
    preserves exact permutation invariance (orbit choices yield the same labelled graph).
  - Extras: random continuous init attains ≥ 1-WL a.s. (embeddings refine colours);
    strictness needs no randomness. Feature-tie sorting alone (no individualization)
    is exactly 1-WL — the sort is then purely the invariance device.
- Synthetic-expressiveness batch (EXP/CEXP + BREC) is **complete**. Headline: LDNA breaks the
  1-WL ceiling (EXP 1.00 / CEXP 1.00 / BREC **321** official — the `gate_init_std: 0.1`
  gate-init variant, with zero-init 138 kept alongside) while GIN sits on it exactly
  (EXP 0.5000 exactly, CEXP 0.7492 ≈ the theoretical 0.75, BREC 0) and DGN saturates EXP/CEXP and scores BREC **145** official (273 under
  shared canonicalization). LDNA with its gate removed and the labelling kept falls to the same
  ceiling (EXP 0.5000, CEXP 0.7500, BREC 0; single-factor ablation above). BREC scores are **best across the 10 protocol seeds**, the benchmark's
  own convention (revised 2026-07-29 from mean-only, which understated us against every published
  entry; means are kept alongside in `.claude/experiments.md`). These datasets are featureless, so
  the permutation-invariance device is a nauty canonical labelling (`data_utils/preprocess.py: canonicalize_graph(s)`) read by LDNA via
  `rank_mode: position`; the model set is LDNA + GIN + DGN and there is no hyperparameter search
  (protocol constants). EXP/CEXP run through the standard loop (10-fold CV); BREC uses a
  dedicated entry point (`scripts/pairwise_eval.py`, `data_utils/brec.py`). Full results, the DGN BREC
  canonicalization ablation, and all protocol decisions are in `.claude/experiments.md`
  (§ Results log, § Datasets → batch-3 notes, § Experiment queue).

---

## Graph-level experiment protocol (complete)
- The graph-level protocol (tune LDNA → broadcast the tuned shared config to all baselines →
  rank → log) has been run on all nine datasets, plus the `DGN` baseline. Objectives, the
  per-dataset tuned configs, and final rankings live in `.claude/experiments.md`
  (§ Experiment queue, § Results log), the source of truth.
- **The expansion (2026-08-05 → 08-24) closed the three gaps it was queued for and did not
  extend LDNA's top-2 record.** `CIFAR10` (LDNA #3 of 12), `Peptides-func` (#2, but ranks 2–12
  are one statistical tie), `Peptides-struct` (#9) and `ogbg-ppa` (#7) add the long-range,
  multi-target-regression and featureless axes the suite lacked; three of the four miss the
  primary objective, and ppa and Peptides-struct are now the project's two worst placements.
  Two structural findings came out of it rather than out of LDNA's own numbers: on `ogbg-ppa`
  five of the twelve configs sit near chance because none of this repo's GCN/SAGE/GAT/GATv2/EGC
  wrappers can carry `edge_attr` into a node state (GCN/SAGE/EGC never read it; GAT/GATv2 read it
  only inside an attention softmax that cancels over a constant node feature) and ppa's entire
  signal is in its edges, and on `CIFAR10` DGN's 8.5-point
  lead traces to its directional field being the coordinates no other model receives. `ogbg-ppa`
  ran a **reduced protocol** (runs=2, 25 search trials, widths {128, 256}; the code's width entry
  now also admits 512, left from a second search round that was not adopted); every deviation is
  disclosed in § Results log.
- Helper scripts (in `scripts/`): `scripts/broadcast.py` writes the tuned shared knobs into
  every baseline config and creates the `GNN-VPA` config (`python scripts/broadcast.py
  --dataset <D> --hidden_channels .. --num_layers .. --dropout .. --lr .. --weight_decay ..
  --gpus 0,1,2,3`); `scripts/run_jobs.sh` runs a set of configs serially on one GPU (launch
  several for concurrency).
- GPU policy: prefer idle GPUs, keep them busy, share freely, never OOM anyone (see
  `.claude/experiments.md` § Compute & parallelism).
- Compute spans two machines: local `ee-kraken` (4× RTX A6000 49GB) and remote `ee-tiamat`
  (H100 95GB); the heaviest datasets (`ogbg-molpcba`, `ogbg-code2`) suit the H100. All datasets,
  logs and checkpoints are archived on `ee-kraken` under `/mnt/data` (symlinked from `data/` and
  `out/checkpoints`); `ee-tiamat` holds no data or results between jobs. SSH access, repo/env mirroring, and GitHub sync are in
  `.claude/experiments.md` § Compute & parallelism.

---

## Partial components, and what was deliberately left undone
- `README.md` is a title stub; it does not explain the project or its setup/usage.
- Node-level support covers the full-graph transductive case only (`Cora`, `Questions`,
  `ogbn-arxiv`). **The rest of the node-level programme was dropped**, not deferred: sampling
  (GraphSAINT), layer-wise exact inference, `PPI` / `ogbn-products` / `Reddit`, and the arxiv
  11-model ranking. The measured reason is that LDNA's distinguishing mechanism cannot work in
  this regime — the canonical rank is a deterministic function of a node's own feature row, so
  `H(δ | x_i, x_j) = 0` and the gate has nothing to condition on (§ Results log). Two datasets
  spanning heterophilous binary and homophilous 40-class already establish that, and the
  remaining datasets would cost 50–200 GPU-h each to reconfirm it. Node level is reported as a
  measured boundary condition rather than as a benchmark sweep.
- **Graph transformers are not run; they are compared through published numbers.** A
  primary-source review (2026-07-29) found citable numbers wherever one could run here. MNIST has
  no graph-transformer number at all, but it has its own val ≡ test problem. Running one under
  the shared LDNA-tuned config would score below its published value and read as crippling the
  baseline. The reference implementations are also unmaintained, and the viable port (PyG's
  `GPSConv`) costs about as much as the DGN port did.

---

## Future milestones
- Expand `README.md` to describe the project, setup, and usage.
- Add more baseline models for experiments.
- Add more **graph-level** datasets. (Node-level is closed — see § Partial components;
  adding node datasets would re-run a question already answered.) The 2026-08 expansion took the
  suite from five to nine; the three axes it was built to close — long-range dependency,
  multi-target regression, featureless graphs — are now covered, so a further dataset needs a
  new reason, not just a new name.

---

## Open questions
- Test LDNA performance as a standalone aggregator component on other methods.

---

## Constraints
- Avoid breaking existing structure unless necessary.
- Avoid over-engineering.
- Prefer consistent coding style with exising modules and minimal-diff implementation.
