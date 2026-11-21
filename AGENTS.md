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
- Graph-level and synthetic-expressiveness experiments are complete. The graph-level suite ran to nine datasets after the 2026-08 expansion (`CIFAR10`, `Peptides-func`, `Peptides-struct`, `ogbg-ppa`), which closed the suite's long-range, multi-target-regression and featureless gaps and **did not extend LDNA's top-2 record** — LDNA places 3rd, 2nd, 9th and 7th of 12 on them. Node-level (transductive, per-node) prediction is wired and **ranked on `Questions`** (LDNA 3rd of 11) with `ogbn-arxiv` tuned; `Cora` remains a smoke test. The headline node-level finding is that LDNA's rank gate contributes nothing there — the canonical rank is a deterministic function of a node's own feature row, so the gap δ carries no information the message MLP does not already receive, where at graph level the within-graph rank encodes graph context. At graph level a single-factor gate ablation finds the gate helps on `ZINC` and is null on `MNISTSuperpixels` and `ogbg-molhiv`; on the expressiveness benchmarks removing it drops LDNA to the 1-WL ceiling (EXP 0.5, CEXP 0.75, BREC 0 of 400), and so does replacing it by an order-sensitive sequence aggregator (PyG MLP/GRU/LSTM aggregation, padded or real-length) over the same nauty order. Measurements, the two corrections they forced, and the gate ablations are in `.claude/experiments.md` § Results log. Sampling, layer-wise inference and the remaining node datasets were dropped rather than deferred, because the finding is structural rather than dataset-specific.

---

## Datasets

- `ogbg-molhiv` and `ogbg-molpcba`: graph-level binary classification.
- `ZINC`: graph-level regression. An optional `data_args: nauty_canonical: true` replaces the feature lexsort with the edge-coloured nauty labelling below (default off; no reported run uses it).
- `MNISTSuperpixels`: graph-level multi-class classification.
- `CIFAR10` (GNNBenchmarkDataset superpixels): graph-level 10-class classification, the MNIST sibling on the same pipeline but a different upstream build. Two differences are load-bearing. It ships a **real validation split**, where `MNISTSuperpixels` has none and the resolver aliases val to test. And its `edge_index` is a **directed** 8-NN graph — out-degree exactly 8, 20.05% of train edges with no reverse — which is **deliberately left directed**, matching the published benchmarking-gnns protocol. Do not read the `ogbn-arxiv` symmetrization below as a house rule: there the citation neighbourhood is meaningfully undirected for every model, here the directedness is the dataset's own construction. It is why DGN takes the image-coordinate field instead of the Laplacian one (`.claude/experiments.md` § Baselines).
- `Peptides-func` / `Peptides-struct` (LRGBDataset): graph-level 10-task multi-label AP and 11-target regression MAE over the **same graphs**. They carry the suite's long-range axis (median diameter 53) and its only multi-target regression. `-struct`'s targets arrive **already standardized by LRGB itself, on the whole dataset rather than on train** — nothing in this repo normalizes them, and that must never be described as train-only normalization.
- `ogbg-ppa`: graph-level 37-way species classification over protein-association graphs, on OGB's out-of-distribution **species** split. The suite's only **featureless** dataset: nodes carry no features, so the feature lexsort cannot impose a permutation-invariant order and an **edge-coloured nauty canonical labelling** (`canonicalize_graphs_edge_colored`, disk-cached) does it instead, which LDNA reads through `rank_mode: position`. Each edge becomes a subdivision vertex coloured by its attribute tuple (and its copy count, for repeated edges), so nodes that differ only in their edge attributes still get an invariant order. The ranked ppa results were produced under the earlier vertex-only labelling (`canonicalize_graphs`), which is not exactly invariant on such nodes, and were not re-run. `utils/resolver.py` **requires** that mode here and raises otherwise — a feature-mode rank over a constant node feature ties every node, so the gate becomes a silent no-op rather than an error. Runs a **reduced protocol** (`.claude/experiments.md` § Search spaces): widths {128, 256} for the reported search (the code's `GRAPH_WIDTHS` entry now also admits 512, left from a second round that found nothing), 2 runs, ~25 search trials, full 12-config ranking; `epochs: 100` is not a reduction but a match with OGB's own ppa reference. By far the most expensive job in the project, and bounded by **host RAM** rather than GPU — one job at a time.
- `ogbg-code2`: graph-level method-name prediction over Python ASTs (5 subtoken positions, F1). **Edge-less**, so GIN rather than GINE. The feature lexsort is skipped because the AST DFS order is canonical by construction; LDNA reads it through `rank_mode: position`. It has a dedicated sequence head (`ASTNodeEncoder` + `Code2Head` in `models/code2.py`) and a guarded fork in `train.py` / `hyperparam_search.py`. One of the two heaviest datasets, with `ogbg-molpcba`.
- `Cora` (Planetoid): node-level multi-class classification, transductive full-graph. A smoke test for the node-level path, not a reported result.
- `Questions` (HeterophilousGraphDataset): node-level binary classification (ROC-AUC), transductive full-graph, heterophilous; its 10 official split masks map to `runs` the way the EXP/CEXP folds do. Tuned and ranked against all ten baselines (LDNA 3rd), and the site of the gate ablation (config pair `ldna_questions_{gated,ungated}.yaml`).
- `ogbn-arxiv`: node-level 40-class classification (accuracy), transductive full-graph, OGB time split. The citation edges are symmetrized with `T.ToUndirected()` for every model, and the OGB split index tensors are converted to boolean node masks. Tuned by a 100-trial search and used for the decisive gate ablation; the 11-model ranking was deliberately not run (see `.claude/experiments.md` § Implementation queue).
- `EXP` / `CEXP` and `BREC`: synthetic expressiveness benchmarks built from 1-WL-indistinguishable graph pairs. They are featureless, so a nauty canonical labelling replaces the feature lexsort and LDNA reads it through `rank_mode: position`. They have no hyperparameter search (the hyperparameters are protocol constants), and the model set is LDNA + GIN + DGN. EXP/CEXP run through the standard loop, with 10-fold CV folds mapped to `runs`. BREC uses its pairwise RPC protocol through a dedicated entry point (`scripts/pairwise_eval.py`, `data_utils/brec.py`).

---

## Model families

- LDNA model: `LDNA` in `models/ldna_net.py`, with `LDNAConv` in `models/ldna_conv.py`.
  - `LDNAConv` is a custom `MessagePassing` layer (`aggr=None`): each message is built from `[x_i, x_j, edge_attr]` through an MLP, modulated by a learned gate of the canonical-rank gap (initialized to identity by default; an optional `gate_init_std` starts it slightly off identity, used only on the synthetic expressiveness benchmarks), summed per node, then run through a post-MLP. The `LDNA` wrapper computes a per-node canonical rank feeding that gate, with two modes (`rank_mode`): a feature-tie rank for feature-sorted datasets, and a position rank for datasets whose node order is canonical by construction (e.g. `ogbg-code2`). A third value, `rank_mode: none`, is the ungated ablation (the gate module exists but never engages), used only by the gate-ablation configs (`ldna_*_ungated.yaml`, at node level, at graph level and on the expressiveness benchmarks). An `aggr_mode` knob (default `sum`) replaces the gated sum by a PyG sequence aggregator over each node's messages in canonical order: `mlp` (MLPAggregation), `gru` and `lstm` (as shipped, zero-padded to a fixed `max_num_elements`), and `gru_packed` / `lstm_packed` (the recurrent output read at the last real neighbour). It is used only by the aggregator-comparison configs `ldna_{exp,cexp,brec}_{mlpaggr,gruaggr,gruaggr_packed,lstmaggr,lstmaggr_packed}.yaml`, all with `rank_mode: none`.
  - `LDNA` wraps a node/edge encoder, a stack of `LDNAConv + BatchNorm + activation + dropout` layers (with optional per-layer residual connections, `residual`), a graph readout, and an MLP prediction head.
  - Readout modes: `add`, `max`, `mean`, `gru`, and `attention`. The `gru` readout runs a GRU over the canonically-sorted node sequence and is the only readout that reads physical node order — but no reported run uses it; every graph-level and expressiveness config uses `attention`, a learnable `AttentionalAggregation` whose softmax normalisation makes it permutation- and size-invariant. So in every reported run the canonical rank reaches the output only through `LDNAConv`'s gate.
- Baseline models in `models/`: `GCN`, `GIN`, `GINE`, `GraphSAGE`, `GAT`, `GATv2`, `PNA`, `EGC`, `DeeperGCN`, `GNN-VPA`, and `DGN`.
- All models share the same interface — `forward(x, edge_index, edge_attr, batch)` and `reset_parameters()` — so `utils/resolver.py` can build and swap any of them interchangeably. `DGN` is the one extension: it takes an additional optional `eig_vec` keyword (the precomputed Laplacian eigenvector field its directional aggregators need), which `train.py` passes only when the data carries it.
- `readout` (all modes above) and an optional per-layer `residual` flag are **shared** knobs supported by every model family, so they can be set identically for LDNA and all baselines. `residual` uses a learnable linear projection on the skip where a layer changes width (identity where it does not). `DeeperGCN` keeps its intrinsic `res+` residual and ignores the shared `residual` flag.

---

## Repo architecture

- `train.py`: main training entry point.
- `hyperparam_search.py`: a single generic Optuna search over LDNA hyperparameters; supports all nine graph-level datasets (`ogbg-molhiv`, `ogbg-molpcba`, `ZINC`, `MNISTSuperpixels`, `ogbg-code2`, `CIFAR10`, `Peptides-func`, `Peptides-struct`, `ogbg-ppa`) and the two wired node-level ones (`Questions`, `ogbn-arxiv`). The node path differs wherever a setting is counted in epochs, since a full-graph epoch is one optimizer step: no LR schedule, no early stopping, no search cap (a trial runs the dataset's full budget), a trial scored by the best rolling median rather than the last-five median, and a memory-feasibility prune (see `.claude/experiments.md` § Search spaces). The expressiveness datasets deliberately have no search (their hyperparameters are protocol constants).
- `models/`: LDNA model code (`ldna_conv.py`, `ldna_net.py`), the shared node-feature `Encoder` (`encoder.py`), and baseline GNN model wrappers.
- `data_utils/`: dataset-side code — `preprocess.py` (graph sorting/canonicalization, vertex-only and edge-coloured nauty, the node-level canonical rank, DGN eigenvector field, degree histogram), `transforms.py` (small transforms), `exp.py` (EXP/CEXP dataset), `brec.py` (BREC dataset + RPC evaluation core), `code2.py` (ogbg-code2 target-sequence utilities). Its `__init__.py` carries the torch>=2.6 `torch.load` compatibility shim.
- `utils/resolver.py`: loads datasets, applies preprocessing, and builds the requested model.
- `utils/evaluator.py` and `utils/logger.py`: evaluators and training logs.
- `utils/tee.py`: tees run output to a log file.
- `scripts/`: helpers and secondary entry points, run from the repo root — `broadcast.py` (write tuned shared knobs into all baseline configs), `run_jobs.sh` (run configs serially on one GPU), `pairwise_eval.py` (pairwise graph-distinguishing evaluation — the BREC RPC protocol; run as `python -m scripts.pairwise_eval`), `train_fold.py` (train a single CV fold, best-of-N seeds; run as `python -m scripts.train_fold`), `val_selected.py` (the validation-selected test next to `best_test`, computed offline from saved logger pickles or `.txt` logs; run as `python -m scripts.val_selected`).
- `configs/`: YAML experiment configs.
- `.claude/`: agent-facing project notes.

Important architecture notes:

- Canonical sorting means each graph is put into a fixed, reproducible node order before model execution. The ordering rule is a design choice, not a tuned hyperparameter — the only requirement is that a consistent order is imposed.
- The current implementation (`data_utils/preprocess.py:sort_graph`) sorts nodes lexicographically by their feature columns, remaps and re-sorts `edge_index`/`edge_attr`, and records the permutation as `new2old` / `old2new`.
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
- YAML float literals: PyYAML (YAML 1.1) parses `1e-1`, `1e1`, and `1.0e1` as **strings** — write
  floats with a dot mantissa and a signed exponent (`1.0e-3`) or as plain decimals (`0.1`).
  **That rule governs the file parse, and it is not sufficient on its own.** `train.py` reads
  `train_args` as a single mapping, so configargparse re-serializes what the file parsed and
  `type=yaml.safe_load` parses it a second time — and that pass sees `repr(value)`. A float whose
  repr has no dot fails there however it was written, because `1.0e-6` and `0.000001` are the same
  float with the same repr (`1e-06`). So **avoid exact powers of ten in `optimizer_kwargs`**; the
  failure surfaces only when training starts, as `Adam` receiving a string weight decay. Searched
  values are never round, and `scripts/broadcast.py::_fmt` restores the dot for hand-passed ones,
  but a hand-authored config can still reintroduce it.
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
