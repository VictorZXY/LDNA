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
