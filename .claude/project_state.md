# Project State

## Current focus
- Maintain a compact experiment runner for LDNA and baseline GNN models.
- Keep the repository understandable through agent-facing documentation.
- Optimization objectives, search spaces, experiment queue, and results are tracked in `.claude/experiments.md`.

---

## Current progress
- The repository is a usable experiment runner (not a packaged project): `train.py`
  (training loop), `configs/` (YAML experiments), `models/` (LDNA — `ldna_conv.py` /
  `ldna_net.py` — plus the baseline wrappers), `utils/` (dataset loading/sorting, resolver,
  evaluators, logger), and `requirements.txt`. Current contents are graph-level only.
- `hyperparam_search.py` tunes LDNA on all five graph-level datasets (ogbg-molhiv,
  ogbg-molpcba, ZINC, MNISTSuperpixels, ogbg-code2). Search mechanics — ranges, pruning,
  plateau early-stopping, the 50-epoch search cap (vs 150 final), and gradient clipping —
  live in `.claude/experiments.md` (§ Search spaces).
- `readout` (incl. `attention` = `AttentionalAggregation`) and a per-layer `residual` flag
  are shared knobs supported by every model family; `DeeperGCN` keeps its intrinsic `res+`
  and ignores the flag. The resolver unifies every model's encoder width to
  `hidden_channels`, so the conv stack is uniform and residual skips are identities (a
  learnable projection covers any `out_channels ≠ hidden_channels`). All configs set
  `readout: attention` + `residual: true`.
- Both `train.py` and `hyperparam_search.py` tee each run under `out/logs/` (`utils/tee.py`)
  and expose an optional `--mem_fraction` per-process GPU cap; the search is OOM-tolerant on
  shared cards (`empty_cache` between trials, `study.optimize(catch=(RuntimeError,))`). Per the
  user (2026-07-08), a search that owns its card runs **uncapped** (`--mem_fraction 1.0`) — a
  cap silently OOM-drops large-hidden trials and biases the search; cap only when co-locating
  jobs on one card. All outputs (logs + checkpoints) live under `out/`.
- LDNA aggregation is rank-gated: `LDNAConv` builds each message from `[x_i, x_j, edge_attr]`
  via an MLP and scales it by a learned gate `1 + gate_nn([δ, |δ|, sign(δ)])` of the
  canonical-rank gap δ = nrank_j − nrank_i (zero-init output layer → identity at init).
  `LDNA` computes the per-node rank under a `rank_mode` knob: `feature` (tie-aware dense rank
  over the lexsorted feature order; the default) or `position` (intra-graph position for
  datasets whose node order is canonical by construction — code2's AST DFS order; used by
  `configs/ldna_code2_rankgate.yaml`). Reruns of the rank-gated LDNA with the previously
  tuned shared params: ZINC done, molhiv/molpcba/MNIST in flight, code2 pending a GPU slot.
- Baselines: `GCN`, `GIN`, `GINE`, `GraphSAGE`, `GAT`, `GATv2`, `PNA`, `EGC`, `DeeperGCN`,
  `GNN-VPA` (`models/vpa.py` — GIN/GINE backbone with PyG's built-in
  `VariancePreservingAggregation`, `sum/√N`), and `DGN` (`models/dgn.py` — a PNA-shaped layer
  whose aggregator set adds the directional `dirK-av`/`dirK-dx`, weighted by the gradient of
  the low-frequency Laplacian eigenvectors; `DGNConv` is a custom `MessagePassing(aggr=None)`
  port of the official DGL code, since PyG has no DGN). `GIN` xor `GINE` runs per dataset by
  edge availability (see `.claude/experiments.md` § Baselines).
- DGN is wired for `ZINC`, `ogbg-molhiv`, `ogbg-molpcba` and `MNISTSuperpixels`; `ogbg-code2`
  is deferred (lazy transform chain would re-run the eigendecomposition every epoch). Its
  per-node eigenvector field is precomputed by `utils/_utils.py: add_eig_vecs`, attached by the
  resolver only when the model is `DGN`, and forwarded by `train.py` only when present — so
  every other baseline's call path is unchanged. Details and the ordering constraints are in
  `.claude/experiments.md` § Baselines.
- `ogbg-code2` is wired as an edge-less (GIN-family) dataset with a dedicated
  sequence-prediction path — `ASTNodeEncoder` + `Code2Head` in `models/code2.py`, plus a
  guarded fork in `train.py` / `hyperparam_search.py` (per-position CE loss, decode→F1
  eval, train-split vocab pre-pass); the feature lexsort is skipped because the AST DFS
  order is already canonical. `ogbg-ppa` is out of scope: its featureless nodes make the
  feature-based sort non-permutation-invariant, which LDNA's MLP aggregator requires (the
  sort is a correctness requirement, not a performance lever).
- Two torch≥2.6 / PyG 2.7 compatibility shims (needed to load MNIST/ZINC/OGB):
  `utils/transforms.py` `__call__`→`forward`; `utils/__init__.py` `torch.load`
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

---

## Next step: graph-level experiment protocol
- The remaining work is to run the graph-level protocol on each dataset: tune LDNA →
  broadcast the tuned shared config to all baselines → rank → log. It is tracked in
  `.claude/experiments.md` (§ Experiment queue, § Results log), the source of truth for
  objectives, search spaces, and results.
- Helper scripts: `broadcast.py` writes the tuned shared knobs into every baseline config
  and creates the `GNN-VPA` config (`python broadcast.py --dataset <D> --hidden_channels ..
  --num_layers .. --dropout .. --lr .. --weight_decay .. --gpus 0,1,2,3`); `run_jobs.sh`
  runs a set of configs serially on one GPU (launch several for concurrency).
- GPU policy: prefer idle GPUs, keep them busy, share freely, never OOM anyone (see
  `.claude/experiments.md` § Compute & parallelism).
- Compute spans two machines: local `ee-kraken` (4× RTX A6000 49GB) and remote `ee-tiamat`
  (H100 95GB, `ssh ee-tiamat`, repo/env mirrored, code synced via GitHub). The heaviest
  datasets (`ogbg-molpcba`, `ogbg-code2`) suit the H100. Datasets are cached under `data/`
  on both machines.

---

## Partial or missing components
- `README.md` is a title stub; it does not explain the project or its setup/usage.
- Node-level task support is not implemented yet (planned future work).

---

## Future milestones
- Expand `README.md` to describe the project, setup, and usage.
- Add more baseline models for experiments.
- Add more node/graph datasets for experiments.
- Add graph transformers implements and run them on the benchmark datasets.

---

## Open questions
- Test LDNA performance on node-level predictions.
- Test LDNA performance as a standalone aggregator component on other methods.

---

## Constraints
- Avoid breaking existing structure unless necessary.
- Avoid over-engineering.
- Prefer consistent coding style with exising modules and minimal-diff implementation.
