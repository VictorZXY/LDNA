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
