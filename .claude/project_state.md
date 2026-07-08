# Project State

## Current focus
- Maintain a compact experiment runner for LDNA and baseline GNN models.
- Keep the repository understandable through agent-facing documentation.
- Optimization objectives, search spaces, experiment queue, and results are tracked in `.claude/experiments.md`.

---

## Current progress
- Main training pipeline is present in `train.py`.
- YAML experiment configs are present in `configs/`.
- LDNA model implementation is present in `models/ldna_conv.py` and `models/ldna_net.py`.
- Baseline GNN model wrappers are present in `models/`.
- Dataset preprocessing, sorting, evaluators, logger, and resolver code are present in `utils/`.
- A generic Optuna search script (`hyperparam_search.py`) supports all five graph-level
  datasets (ogbg-molhiv, ogbg-molpcba, ZINC, MNISTSuperpixels, ogbg-code2). `MedianPruner`
  is enabled and search trials early-stop on plateau (`patience=20`, `min_delta=1e-3`); the
  search uses a low epoch cap of 50 while final training runs the full 150. `n_trials`
  defaults to 100, and `batch_size` is a fixed per-dataset power of 2. All training loops
  clip gradients to `max_norm=1.0`. Search ranges and rationale live in
  `.claude/experiments.md` (§ Search spaces).
- `readout` (now incl. `attention` = `AttentionalAggregation`) and an optional per-layer
  `residual` flag are shared, config-selectable knobs supported by every model family
  (LDNA + all baselines); `DeeperGCN` keeps its intrinsic `res+` and ignores the flag.
  The resolver maps every model's encoder to the model width
  (`embedding_dim = hidden_channels`, previously `128` for non-DeeperGCN), so the conv
  stack is uniform and residual skips are identity (a learnable linear projection is kept
  as a safety net for `out_channels ≠ hidden_channels`). All current configs set
  `readout: attention` + `residual: true`.
- `train.py` and `hyperparam_search.py` tee stdout/stderr to a per-run `logs/<name>.txt`
  via `utils/tee.py` (`Tee` / `tee_to_file`), and cap each process's GPU share via
  `--mem_fraction` (`set_per_process_memory_fraction`) so a co-located job cannot OOM.
  The search is also hardened for shared-GPU co-tenancy (`empty_cache` between trials,
  `study.optimize(catch=(RuntimeError,))` so an OOM trial is skipped, not fatal).
- Python dependencies are listed in `requirements.txt`. The repository is usable as an
  experiment runner; it is not documented as a packaged project. Current contents are
  centered on graph-level experiments.
- `GNN-VPA` baseline added (`models/vpa.py`, class `VPA`): GIN/GINE backbone with the PyG
  built-in `VariancePreservingAggregation` (`sum/√N`) injected via `aggr=`; dual-path
  (edge datasets → `GINEConv`, edge-less → `GINConv`); shared interface; resolver query
  `GNN-VPA`.
- `ogbg-code2` is wired as an edge-less (GIN-family) dataset with a dedicated
  sequence-prediction path — `ASTNodeEncoder` (type/attr/depth) + `Code2Model` seq-head +
  per-position CE loss + decode→F1 eval + train-split vocab pre-pass; the feature lexsort
  is skipped (AST DFS order is already canonical). `train.py` / `hyperparam_search.py` use
  a guarded fork so the other datasets' loops are unchanged. `ogbg-ppa` is out of scope:
  featureless nodes make the feature-based canonical sort non-permutation-invariant, which
  LDNA requires (the sort is a correctness requirement, not a performance lever — LDNA's
  gain is the MLP aggregator).
- Two torch≥2.6 / PyG 2.7 compatibility fixes are in place (needed to load MNIST/ZINC/OGB):
  `utils/transforms.py` `__call__`→`forward`; `utils/__init__.py` `torch.load`
  `weights_only=False` shim.

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
