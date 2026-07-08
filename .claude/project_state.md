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
  and cap per-process GPU share via `--mem_fraction`; the search is OOM-tolerant on shared
  cards (`empty_cache` between trials, `study.optimize(catch=(RuntimeError,))`). All outputs
  (logs + checkpoints) live under `out/`.
- Baselines: `GCN`, `GIN`, `GINE`, `GraphSAGE`, `GAT`, `GATv2`, `PNA`, `EGC`, `DeeperGCN`,
  and `GNN-VPA` (`models/vpa.py` — GIN/GINE backbone with PyG's built-in
  `VariancePreservingAggregation`, `sum/√N`). `GIN` xor `GINE` runs per dataset by edge
  availability (see `.claude/experiments.md` § Baselines).
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
