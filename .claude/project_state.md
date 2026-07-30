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
- `hyperparam_search.py` tunes LDNA on all five graph-level datasets (ogbg-molhiv,
  ogbg-molpcba, ZINC, MNISTSuperpixels, ogbg-code2) and on the two wired node-level ones
  (Questions, ogbn-arxiv). Search mechanics — ranges, pruning, plateau early-stopping, the
  50-epoch search cap (vs 150 final), and gradient clipping — live in
  `.claude/experiments.md` (§ Search spaces), together with the node-level space, whose
  epoch-counted settings all differ because a full-graph epoch is one optimizer step
  (no schedule, no early stopping, a 333-step cap, best-epoch scoring, memory prune).
- `readout` (incl. `attention` = `AttentionalAggregation`) and a per-layer `residual` flag
  are shared knobs supported by every model family (all configs use `readout: attention` +
  `residual: true`); the resolver unifies every model's encoder width to `hidden_channels`, so
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
  it has been rerun with the previously tuned shared params on all five graph-level datasets
  (results in `.claude/experiments.md` § Results log).
- Baselines: `GCN`, `GIN`, `GINE`, `GraphSAGE`, `GAT`, `GATv2`, `PNA`, `EGC`, `DeeperGCN`,
  `GNN-VPA` (`models/vpa.py`), and `DGN` (`models/dgn.py`). `GIN` xor `GINE` runs per dataset by
  edge availability. Per-model backbones, `DGN`'s directional aggregators, and the GIN/GINE rule
  are in `.claude/experiments.md` § Baselines.
- DGN is wired for all five graph-level datasets. Its per-node directional field is attached by
  the resolver only when the model is `DGN` and forwarded by `train.py` only when present, so
  every other baseline's call path is unchanged; the field is the low-frequency Laplacian
  eigenvectors on the four molecular/image datasets and the AST depth on `ogbg-code2`. Field
  construction, caching, measurements, and rejected alternatives are in
  `.claude/experiments.md` § Baselines.
- `ogbg-code2` is wired as an edge-less (GIN-family) dataset with a dedicated
  sequence-prediction path (`ASTNodeEncoder` + `Code2Head` in `models/code2.py`, plus a guarded
  `train.py` / `hyperparam_search.py` fork); the feature lexsort is skipped because the AST DFS
  order is already canonical. `ogbg-ppa` is out of scope: featureless nodes make the
  feature-based sort non-permutation-invariant, which LDNA's MLP aggregator requires (the sort
  is a correctness requirement, not a performance lever). Both are covered in
  `.claude/experiments.md` § Datasets.
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
  sum-of-messages aggregator), paired with the hand-picked, untuned gate-ablation configs
  `ldna_questions_gated.yaml` / `ldna_questions_ungated.yaml` (identical except the gate; not
  run yet). `ogbn-arxiv` is wired full-graph (`configs/ldna_arxiv.yaml`, hand-picked and untuned,
  pending the node-level search): OGB ships split INDEX tensors, so `utils/resolver.py:
  _idx_to_masks` turns them into the boolean `[N]` masks the rest of the node path reads and
  rejects a non-partitioning split on the spot; `T.ToUndirected()` symmetrizes the directed
  citation edges as a preprocessing choice shared by every model (169,343 nodes / 2,315,598
  undirected edge entries); `y` is flattened to `[N]` for the CrossEntropy path. Measured peak
  GPU memory for full-graph LDNA (3 training steps, A6000): **18.33 GiB at h=128/L=3** and
  **35.58 GiB at h=256/L=3** — ≈6.1 GiB per layer at h=128, scaling linearly in both, which is
  what bounds the `(hidden_channels, num_layers)` box of the node-level search on a 49 GB card.
  The sampled control is still to come.
- `Questions` has its baseline config set: ten `<model>_questions.yaml` files carrying LDNA's
  tuned shared knobs from `configs/ldna_questions_h64l2.yaml` (h=64, L=2, dropout 0.616143,
  lr 4.871103e-3, wd 5.3e-6), broadcast by `scripts/broadcast.py`, which learned the dataset via
  `SUFFIX`/`HAS_EDGE` and needed no node-specific special-casing (it never touches `batch_size`,
  `readout` or `scheduler`). Node-level shape: no `readout` key, no `scheduler` key, no
  `batch_size`, `runs: 10` matching the ten official split columns. `GIN` (not `GINE`) since
  Questions is edge-less; **`DGN` is excluded at node level** — `add_eig_vec` materializes a
  dense `N × N` adjacency, infeasible at 48,921 nodes, and the `NODE_DATASETS` branch never
  attaches the field — so node level ranks LDNA plus ten baselines, not eleven. All eleven
  configs verified to build and run a forward+backward producing `[48921, 1]`.
- `train.py: _reset_encoders` re-initializes the node/edge encoders at the start of each run —
  model `reset_parameters()` covers the conv stack, norms, readout and head but not the encoders —
  so runs are independent for LDNA and every baseline alike, applied in the caller with no
  baseline model file touched. It reproduces each encoder's construction-time init distribution
  (`Encoder`/`ASTNodeEncoder` define Xavier `reset_parameters`; OGB's `AtomEncoder`/`BondEncoder`
  are re-initialized explicitly since the ogb package defines no reset method).
  `scripts/train_fold.py` was already clean (it rebuilds per seed); `data_utils/brec.py` has the
  same pattern and is deliberately left alone — see § Open questions.
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
  (0.50 / 0.75 / 0) and DGN saturates EXP/CEXP and scores BREC **145** official (273 under
  shared canonicalization). BREC scores are **best across the 10 protocol seeds**, the benchmark's
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
  rank → log) has been run on all five datasets, plus the `DGN` baseline. Objectives, the
  per-dataset tuned configs, and final rankings live in `.claude/experiments.md`
  (§ Experiment queue, § Results log), the source of truth.
- Helper scripts (in `scripts/`): `scripts/broadcast.py` writes the tuned shared knobs into
  every baseline config and creates the `GNN-VPA` config (`python scripts/broadcast.py
  --dataset <D> --hidden_channels .. --num_layers .. --dropout .. --lr .. --weight_decay ..
  --gpus 0,1,2,3`); `scripts/run_jobs.sh` runs a set of configs serially on one GPU (launch
  several for concurrency).
- GPU policy: prefer idle GPUs, keep them busy, share freely, never OOM anyone (see
  `.claude/experiments.md` § Compute & parallelism).
- Compute spans two machines: local `ee-kraken` (4× RTX A6000 49GB) and remote `ee-tiamat`
  (H100 95GB); the heaviest datasets (`ogbg-molpcba`, `ogbg-code2`) suit the H100 and are cached
  under `data/` on both. SSH access, repo/env mirroring, and GitHub sync are in
  `.claude/experiments.md` § Compute & parallelism.

---

## Partial or missing components
- `README.md` is a title stub; it does not explain the project or its setup/usage.
- Node-level support covers only the full-graph transductive case on `Cora`, `Questions` and
  `ogbn-arxiv`. Sampling (GraphSAINT, including arxiv's sampled control), layer-wise exact
  inference, the remaining datasets (PPI, ogbn-products, Reddit), and the tune → broadcast → rank
  protocol are not implemented yet.

---

## Future milestones
- Expand `README.md` to describe the project, setup, and usage.
- Add more baseline models for experiments.
- Add more node/graph datasets for experiments.
- Add graph transformers implements and run them on the benchmark datasets.

---

## Open questions
- Should `data_utils/brec.py` get the same encoder reset as `train.py`? Its per-pair
  `reset_parameters()` has the identical gap, and its backbone carries
  `node_encoder=nn.Linear(1, emb)`. Left alone deliberately: BREC's committed 321 is the headline
  result, the protocol is init-sensitive (the whole point of `gate_init_std`), and changing the
  code without re-running would leave that number unreproducible. Decide before any BREC re-run.
- Test LDNA performance on node-level predictions.
- Test LDNA performance as a standalone aggregator component on other methods.

---

## Constraints
- Avoid breaking existing structure unless necessary.
- Avoid over-engineering.
- Prefer consistent coding style with exising modules and minimal-diff implementation.
