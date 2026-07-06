# AGENTS.md

This file provides guidance to coding agents (Claude Code, Codex, Cursor, OpenCode, etc.) when working with code in this repository. It is the canonical agent-context doc; `CLAUDE.md` is a symlink to this file.

It has two purposes:

- Describe the project and the work it does.
- Define the constitution-level rules agents must follow when working here.

Ongoing progress is tracked separately in `.claude/project_state.md`, which agents maintain and keep in sync as work proceeds.

---

## Read first

Before doing any task, read `.claude/project_state.md` for the current focus, progress, and open questions.

For model-optimization or experiment tasks, also read `.claude/experiments.md` — the tuning playbook and experiment queue.

---

## Project overview

- This repository contains research code for Learnable Dynamic Neighborhood Aggregation (LDNA) on graph learning tasks.
- LDNA is the method studied here. It sorts graph inputs into a canonical order before model execution, then trains a learnable neighborhood aggregation model implemented in code as `LDNA` and `LDNAConv`.
- The repository is centered on running experiments rather than packaging a reusable library. It compares LDNA against several baseline GNN families under the same general training pipeline.
- The project runs in a Conda environment named `LDNA`.
- All current experiments are graph-level. LDNA is not limited to graph-level prediction; node-level experiments are planned future work and are not yet implemented in the code.

---

## Datasets

- `ogbg-molhiv` and `ogbg-molpcba`: graph-level binary classification.
- `ZINC`: graph-level regression.
- `MNISTSuperpixels`: graph-level multi-class classification.

---

## Model families

- LDNA model: `LDNA` in `models/ldna_net.py`, with `LDNAConv` in `models/ldna_conv.py`.
  - `LDNAConv` is a custom `MessagePassing` layer (`aggr=None`): each message is built from `[x_i, x_j, edge_attr]` through an MLP, passed through a learnable linear map, summed per node, then run through a post-MLP.
  - `LDNA` wraps a node/edge encoder, a stack of `LDNAConv + BatchNorm + activation + dropout` layers, a graph readout, and an MLP prediction head.
  - Readout modes: `add`, `max`, `mean`, and `gru`. The `gru` readout runs a GRU over the canonically-sorted node sequence, which is the mechanism that exploits the sort. (Current configs use `mean`.)
- Baseline models in `models/`: `GCN`, `GIN`, `GINE`, `GraphSAGE`, `GAT`, `GATv2`, `PNA`, `EGC`, and `DeeperGCN`.
- All models share the same interface — `forward(x, edge_index, edge_attr, batch)` and `reset_parameters()` — so `utils/resolver.py` can build and swap any of them interchangeably.

---

## Repo architecture

- `train.py`: main training entry point.
- `hyperparam_search.py`: a single generic Optuna search over LDNA hyperparameters; supports `ogbg-molhiv`, `ogbg-molpcba`, and `ZINC`.
- `models/`: LDNA model code (`ldna_conv.py`, `ldna_net.py`), the shared node-feature `Encoder` (`encoder.py`), and baseline GNN model wrappers.
- `utils/resolver.py`: loads datasets, applies preprocessing, and builds the requested model.
- `utils/_utils.py`: graph sorting and dataset-side preprocessing support.
- `utils/transforms.py`: small transforms.
- `utils/evaluator.py` and `utils/logger.py`: evaluators and training logs.
- `configs/`: YAML experiment configs.
- `local_tests/`: unit tests.
- `.claude/`: agent-facing project notes.

Important architecture notes:

- Canonical sorting means each graph is put into a fixed, reproducible node order before model execution. The ordering rule is a design choice, not a tuned hyperparameter — the only requirement is that a consistent order is imposed.
- The current implementation (`utils/_utils.py:sort_graph`) sorts nodes lexicographically by their feature columns, remaps and re-sorts `edge_index`/`edge_attr`, and records the permutation as `new2old` / `old2new`.
- Sorting is applied once during dataset preparation from `utils/resolver.py` (`sort_graphs`), on the data side rather than inside the model.

---

## Repo workflow

- Choose a YAML config from `configs/`.
- Run `train.py` with `--config <path>`.
- `utils/resolver.py` loads the dataset, applies preprocessing, and builds the requested model.
- Training repeats for `train_args['runs']` independent runs; each run resets parameters and trains for `epochs`.
- `utils/logger.py` tracks per-run best train/val/test metrics and prints a mean ± std summary at the end.
- If configured, a checkpoint (`<experiment_name>.pt`) and a pickled logger (`<experiment_name>_logger.pickle`) are written to `checkpoint_dir` / `log_dir`.

---

## Configuration

- Experiments are configured through YAML files in `configs/`.
- Top-level fields: `experiment_name`, `model`, `model_args`, `dataset`, `data_args`, `train_args`, `checkpoint_dir`, `log_dir`, plus `cuda` and `seed`.
- `train_args` holds the training-loop settings: `runs`, `epochs`, `eval_interval`, `loss_fn`, `optimizer` (+ `optimizer_kwargs`), `scheduler` (+ `scheduler_kwargs`), and `evaluator` (+ `evaluator_kwargs`).

---

## Naming conventions

- Prefer following the existing repository naming in code, configs, and discussions of concrete modules.

---

## Code style

- New code must match the style of the existing code as closely as possible — structure, naming, formatting, and idioms.
- Follow the patterns already used in the surrounding module rather than introducing new conventions.

---

## Documentation policy

- Use plain, factual language.
- Keep documents concise and structured.
- Do not invent features not present in the repository.
- Prefer explicit explanations over abstract descriptions.

---

## Task workflow

For each task:

1. Understand the task and relevant modules.
2. Provide a short plan.
3. Implement with minimal changes.
4. Keep consistency with existing style.
5. Validate if applicable.

---

## Handling ambiguity

If something is ambiguous:

1. Make the smallest reasonable assumption consistent with the current repo and docs.
2. Explicitly state the assumption.
3. If the assumption could affect architecture or interfaces, ask for confirmation before proceeding.

Prefer asking for clarification over making large or irreversible decisions.

---

## Task scoping

When project docs mention multiple current or future tasks, follow the task explicitly requested in the current user prompt.

Do not proactively continue to later roadmap items unless explicitly asked.

---

## Autonomous optimization

When asked to optimize LDNA, `.claude/experiments.md` is the source of truth for objectives, search spaces, the experiment queue, and results. Follow it, and:

- Stay within the compute budget and stop-and-report conditions defined there.
- Do not change off-limits parts (the canonical sort, model architecture contracts, dataset splits/preprocessing). Tune through configs and search spaces, not by editing model internals, unless explicitly asked.
- Log every run in the results log and keep the best-so-far in sync.
- Work the queue in order; do not add new directions beyond it without asking.
- Report results and stop when a target is reached, the budget is exhausted, or progress stalls per the stop conditions.

---

## Documentation updates

After completing a task, update `.claude/project_state.md` and any other affected docs so they stay consistent with the current implementation and task status.

Keep documentation updates scoped:

- update only the docs affected by the completed task
- do not rewrite unrelated sections
- keep project docs concise and current

---

## Code explanation requirement

All non-trivial code changes must be accompanied by concise explanations.

Explain:

- what the code does
- why this design is used
- how it fits into the overall system in this repository
- how it interacts with other components in the repository, and with other repositories when that is actually relevant

Focus on system role and design rationale rather than line-by-line narration.

Avoid:

- pedantic walkthroughs
- restating obvious syntax
- overly long explanations without architectural insight

---

## Output format

When completing a task, include:

- assumptions
- changes made
- affected modules
- concise code explanation
- risks / follow-ups

Be concise and technical.
