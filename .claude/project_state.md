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
- A generic Optuna search script (`hyperparam_search.py`) is present for MolHIV, MolPCBA, and ZINC.
  `MedianPruner` is enabled and search trials **early-stop on plateau** (`patience=20`,
  `min_delta=1e-3`); the search uses a low **epoch cap of 50** while final training runs
  the full **150**. `n_trials` defaults to 100, and `batch_size` is a fixed per-dataset
  power of 2 (hiv/mnist 256, molpcba/zinc 512; ZINC search subset uses 128).
- `readout` (now incl. `attention` = `AttentionalAggregation`) and an optional per-layer
  `residual` flag are shared, config-selectable knobs supported by every model family
  (LDNA + all baselines); `DeeperGCN` keeps its intrinsic `res+` and ignores the flag.
  The resolver now maps every model's encoder to the model width
  (`embedding_dim = hidden_channels`, previously `128` for non-DeeperGCN), so the conv
  stack is uniform and residual skips are identity (a learnable linear projection is kept
  as a safety net for `out_channels ≠ hidden_channels`). All current configs set
  `readout: attention` + `residual: true`.
- `train.py` and `hyperparam_search.py` tee stdout/stderr to a per-run `logs/<name>.txt`
  via `utils/tee.py` (`Tee` / `tee_to_file`).
- Python dependencies are listed in `requirements.txt`.
- The repository is usable as an experiment runner.
- The repository is not documented as a packaged project.
- The current repository contents are centered on graph-level experiments.
- `GNN-VPA` baseline added (`models/vpa.py`, class `VPA`): GIN/GINE backbone with the PyG
  built-in `VariancePreservingAggregation` (`sum/√N`) injected via `aggr=`; dual-path
  (edge datasets → `GINEConv`, edge-less → `GINConv`); shared interface; resolver query
  `GNN-VPA`. Committed `a4d55e2`.
- `hyperparam_search.py` extended to `MNISTSuperpixels`, arch-aligned so the search tunes
  under the same architecture as the final runs (`readout: attention` + `residual: true`),
  and hardened for shared-GPU co-tenancy (per-process VRAM cap
  `set_per_process_memory_fraction`, `empty_cache` between trials, `catch=(RuntimeError,)`
  so an OOM trial is skipped instead of aborting the study).
- Two torch≥2.6 / PyG 2.7 compatibility fixes (needed to load MNIST/ZINC/OGB): 
  `utils/transforms.py` `__call__`→`forward`; `utils/__init__.py` `torch.load`
  `weights_only=False` shim.
- LDNA hyperparameter searches (100 trials each) currently running for `ogbg-molhiv`,
  `ZINC`, `ogbg-molpcba`, `MNISTSuperpixels`, one per GPU. GPU policy: prefer idle, keep
  all 4 busy, share freely, never OOM anyone (see `.claude/experiments.md`).
- Graph-level scope fixed to those four datasets **plus `ogbg-code2`** (edge-less / GIN
  family, dedicated sequence-prediction path — `ASTNodeEncoder` + `Code2Model` seq-head +
  per-position CE loss + decode→F1 eval + train-split vocab pre-pass; **implemented & merged
  `aade777`**, `train.py`/`hyperparam_search.py` use a guarded `if is_code2` fork so the other
  datasets' loops are unchanged). **`ogbg-ppa` dropped**: featureless nodes make the feature-based canonical
  sort non-permutation-invariant, which LDNA requires (the sort is a correctness requirement,
  not a performance lever — LDNA's gain is the MLP aggregator).

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
