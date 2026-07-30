import argparse
import os
from collections import Counter
from statistics import median

import optuna
import torch
from torch import nn

from train import _nrank_kwargs, _split_mask
from utils import evaluator_resolver, loss_resolver, model_and_data_resolver, tee_to_file
from utils.evaluator import Code2Evaluator
from utils.resolver import NODE_DATASETS

# Steps a node-level search trial runs: the dataset's FULL final budget, i.e. no search cap. Full-
# graph transductive training runs ONE optimizer step per epoch where a graph-level epoch runs
# 130-680, so the graph-level cap of 50 would score a node trial after 50 steps and the ratio-
# preserving alternative (a third of the budget, 333) still truncates. There is nothing to concede
# anyway: the cap exists only to bound wall-clock, and a full Questions trial is ~79 s. Removing it
# also buys exactness — measured over six configs spanning this space, the rank correlation between
# the search score and the reported metric is 0.60 at a 333-step cap and 1.00 uncapped, because a cap
# under-rates any config whose optimum lands past it and the measured optima ranged from step 117 to
# step 559. Uncapped, no assumption about where the optimum sits has to transfer between datasets.
# Per-dataset because the budgets differ: arxiv is still climbing at step 1000 (measured), and 2000
# is what every work that beats OGB's reference uses (§ Evaluation protocol).
NODE_SEARCH_EPOCHS = {'Questions': 1000, 'ogbn-arxiv': 2000}

# Width of the rolling median a node trial is scored by (see the end of `train_LDNA`). Five matches
# the graph-level rule's window, and measured rank correlation with the reported metric falls as the
# window widens (+0.60 at 5, +0.47 at 11 and 25) because a wider window blurs a peak the curve only
# holds briefly.
NODE_SCORE_WINDOW = 5

# Predicted full-graph peak RESERVED memory, which is what `mem_fraction` actually caps. Fit affine
# in depth rather than purely proportional: the per-layer term alone missed the fixed encoder and
# feature cost and read low everywhere, worst at L=2. The affine form reproduces both arxiv
# measurements at h=128 exactly (13.36 GiB allocated at L=2, 18.33 at L=3) and stays ~3%
# conservative at h=256/L=3 (36.60 predicted vs 35.58 measured). The cost is dominated by
# LDNAConv's per-EDGE activations rather than by its 0.2-0.7M parameters, so the constants are
# carried per edge entry and scaled to each dataset's own edge_index width.
NODE_MEM_GIB_FIXED = 3.42
NODE_MEM_GIB_PER_LAYER = 4.97
# Allocator overhead: `mem_fraction` caps RESERVED memory, which sits above allocated. Measured
# across six configs on both datasets the gap is 0.42-0.85 GiB and does NOT scale with the
# allocation — it looks large in relative terms only on the small configs (28% of a 1.9 GiB
# Questions run, 1.9% of a 35.6 GiB arxiv one). So it is carried as an ADDITIVE headroom. A
# multiplicative margin would over-predict exactly where it matters: at h=256/L=3 on arxiv a 1.2x
# factor gives 44 GiB and prunes a config measured to reserve 36.27.
NODE_MEM_GIB_OVERHEAD = 1.5
NODE_EDGE_ENTRIES = {'Cora': 10_556, 'Questions': 307_080, 'ogbn-arxiv': 2_315_598}


def _node_peak_gib(dataset: str, hidden_channels: int, num_layers: int) -> float:
    # The prune must err CONSERVATIVE, which is the opposite of what it looks like it should do.
    # A pruned trial and a failed one are not equivalent to the sampler: optuna's TPE splits
    # RUNNING/COMPLETE/PRUNED trials and scores a prune with no intermediate values as worst, so a
    # prune actively steers sampling away from that region, while a FAIL trial is filtered out
    # upstream and teaches it nothing. Measured on this control flow (arxiv box, 100 trials,
    # seed 42): pruning the infeasible region put 55/100 trials on the feasible point against 9/100
    # when the same configs OOMed instead. Being permissive therefore buys nothing and costs OOMs.
    #
    # CAVEAT the depth term is fit at L=2 and L=3, so anything past ~L=4 is an extrapolation; at
    # h=128/L=7 on arxiv it lands just inside a 0.85 budget and could still OOM into the study's
    # `catch`. That costs one trial slot, which is the acceptable side of the asymmetry above.
    return ((NODE_MEM_GIB_FIXED + NODE_MEM_GIB_PER_LAYER * num_layers)
            * (hidden_channels / 128)
            * NODE_EDGE_ENTRIES[dataset] / NODE_EDGE_ENTRIES['ogbn-arxiv']
            + NODE_MEM_GIB_OVERHEAD)


def _get_study_settings(dataset: str) -> dict:
    if dataset == 'ogbg-molhiv':
        # direction='maximize' because we want to maximize ROC-AUC
        return {
            'direction': 'maximize',
            'best_value_label': 'Best Validation ROC-AUC',
        }
    elif dataset == 'ogbg-molpcba':
        return {
            'direction': 'maximize',
            'best_value_label': 'Best Validation AP',
        }
    elif dataset == 'ZINC':
        return {
            'direction': 'minimize',
            'best_value_label': 'Best Validation MAE',
        }
    elif dataset == 'MNISTSuperpixels':
        return {
            'direction': 'maximize',
            'best_value_label': 'Best Validation Accuracy',
        }
    elif dataset == 'ogbg-code2':
        return {
            'direction': 'maximize',
            'best_value_label': 'Best Validation F1',
        }
    elif dataset == 'Questions':
        return {
            'direction': 'maximize',
            'best_value_label': 'Best Validation ROC-AUC',
        }
    elif dataset == 'ogbn-arxiv':
        return {
            'direction': 'maximize',
            'best_value_label': 'Best Validation Accuracy',
        }

    raise ValueError(f"Unsupported dataset '{dataset}'")


def _get_training_config(
        dataset: str,
        batch_size: int,
        hidden_channels: int,
        num_layers: int,
        dropout: float,
):
    model_args = {
        'hidden_channels': hidden_channels,
        'out_channels': hidden_channels,
        'num_layers': num_layers,
        'dropout': dropout,
        'readout': 'attention',
        'residual': True,
    }

    # Node-level datasets predict one row per node: the resolver sets `readout=None` itself and its
    # level guard REJECTS a config that carries a graph readout, so the key is dropped rather than
    # overridden (the guard judges what the config asked for, not the post-update value).
    if dataset in NODE_DATASETS:
        del model_args['readout']

    if dataset == 'ogbg-molhiv':
        return {
            'model_args': model_args,
            'data_args': {
                'root': 'data/',
                'batch_size': batch_size,
            },
            'loss_query': 'BCEWithLogitsLoss',
            'evaluator_query': 'OGBGraphPropPredEvaluator',
            'evaluator_kwargs': {
                'name': 'ogbg-molhiv',
            },
            'scheduler_mode': 'max',
        }
    elif dataset == 'ogbg-molpcba':
        return {
            'model_args': model_args,
            'data_args': {
                'root': 'data/',
                'batch_size': batch_size,
            },
            'loss_query': 'BCEWithLogitsLoss',
            'evaluator_query': 'OGBGraphPropPredEvaluator',
            'evaluator_kwargs': {
                'name': 'ogbg-molpcba',
            },
            'scheduler_mode': 'max',
        }
    elif dataset == 'ZINC':
        return {
            'model_args': model_args,
            'data_args': {
                'root': 'data/zinc/',
                'batch_size': batch_size,
                'subset': True,
            },
            'loss_query': 'L1Loss',
            'evaluator_query': 'ZINCEvaluator',
            'evaluator_kwargs': {},
            'scheduler_mode': 'min',
        }
    elif dataset == 'MNISTSuperpixels':
        return {
            'model_args': model_args,
            'data_args': {
                'root': 'data/mnist/',
                'batch_size': batch_size,
            },
            'loss_query': 'CrossEntropyLoss',
            'evaluator_query': 'MNISTEvaluator',
            'evaluator_kwargs': {},
            'scheduler_mode': 'max',
        }
    elif dataset == 'ogbg-code2':
        # Edge-less sequence task: loss/evaluator are handled by the code2 fork in train_LDNA
        # (per-position CE + the OGB F1 evaluator built from the model's idx2vocab), so
        # `loss_query` is None here.
        return {
            'model_args': model_args,
            'data_args': {
                'root': 'data/',
                'batch_size': batch_size,
            },
            'loss_query': None,
            'evaluator_query': 'Code2Evaluator',
            'evaluator_kwargs': {
                'name': 'ogbg-code2',
            },
            'scheduler_mode': 'max',
        }
    elif dataset == 'Questions':
        # Full-graph transductive: one graph is one batch, so `data_args` carries no `batch_size`.
        # `scheduler_mode` only names the metric direction here — the node path runs no scheduler
        # (see train_LDNA). Questions ships 10 official splits; the resolver returns split 0 as the
        # plain loaders and rides the rest on the train_loader as `cv_folds`, so the search scores
        # split 0, which is the single run § Evaluation protocol specifies for a trial.
        return {
            'model_args': model_args,
            'data_args': {
                'root': 'data/questions/',
            },
            'loss_query': 'BCEWithLogitsLoss',
            'evaluator_query': 'ROCAUCEvaluator',
            'evaluator_kwargs': {},
            'scheduler_mode': 'max',
        }
    elif dataset == 'ogbn-arxiv':
        return {
            'model_args': model_args,
            'data_args': {
                'root': 'data/',
            },
            'loss_query': 'CrossEntropyLoss',
            'evaluator_query': 'OGBNodeEvaluator',
            'evaluator_kwargs': {
                'name': 'ogbn-arxiv',
            },
            'scheduler_mode': 'max',
        }

    raise ValueError(f"Unsupported dataset '{dataset}'")


def train_LDNA(
        trial,
        dataset: str,
        hidden_channels: int = 128,
        num_layers: int = 4,
        dropout: float = 0.3,
        batch_size: int = 256,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        epochs: int = 50,
        patience: int = 20,
        min_delta: float = 1e-3,
        device='cuda:0',
        mem_fraction: float = 0.30,
):
    # ogbg-code2 uses a sequence head (per-position CE loss + F1 eval); the code2 fork mirrors this
    # function, and everything below is left unchanged for all single-target datasets.
    if dataset == 'ogbg-code2':
        return _train_LDNA_code2(
            trial, dataset, hidden_channels, num_layers, dropout, batch_size, lr,
            weight_decay, epochs, patience, min_delta, device, mem_fraction,
        )

    # ---- Load device ----
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    # Cap this process's share of the (shared) GPU and release cache left by the previous
    # trial, so a large sampled model can never OOM a co-located job on the same card. The
    # cap is deterministic: configs that don't fit the budget raise OOM and are skipped by
    # the study's `catch=(RuntimeError,)` rather than eating another tenant's memory.
    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(mem_fraction, device.index)
        torch.cuda.empty_cache()

    # ---- Load data ----
    config = _get_training_config(
        dataset=dataset,
        batch_size=batch_size,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        dropout=dropout,
    )

    # ---- Instantiate model ----
    model, train_loader, val_loader, _ = model_and_data_resolver(
        'LDNA',
        dataset,
        model_args=config['model_args'],
        data_args=config['data_args'],
    )
    loss_fn = loss_resolver(config['loss_query'])
    evaluator = evaluator_resolver(config['evaluator_query'], **config['evaluator_kwargs'])

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # No LR schedule at node level, mirroring train.py's `_lr_scheduler`: one epoch is one optimizer
    # step there, so `ReduceLROnPlateau`'s default patience of 10 is ten steps and collapses the LR
    # from 1e-3 to 1e-7 in ~44 of them — a trial would be scored on a dead model. The final runs hold
    # the LR constant (§ Evaluation protocol), so the search has to score configs in that regime.
    node_level = dataset in NODE_DATASETS
    scheduler = None if node_level else torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode=config['scheduler_mode'])

    all_results = []
    mode = config['scheduler_mode']
    best_metric = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        # ---- Training ----
        model.train()

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch,
                        **_nrank_kwargs(batch))

            # As in train.py: the split mask comes BEFORE the `is_labelled` filter, which is
            # molpcba's missing-label (NaN) guard and not a split selector — leaving the split to it
            # would train a node trial on the whole graph, validation and test labels included. A
            # graph-level loader carries no split tag and gets a full slice.
            mask = _split_mask(batch, train_loader)
            out, y = out[mask], batch.y[mask]

            if isinstance(loss_fn, nn.CrossEntropyLoss):
                is_labelled = (y == y).view(-1)
                loss = loss_fn(out[is_labelled], y[is_labelled].long())
            else:
                is_labelled = (y == y)
                loss = loss_fn(out[is_labelled], y[is_labelled].float())

            # A diverged trial is pruned rather than allowed to reach the evaluator. Gradient
            # clipping only rescales FINITE gradients, so a high sampled lr can still produce
            # non-finite activations — and `roc_auc_score` raises ValueError on NaN scores, which
            # `study.optimize(catch=(RuntimeError,))` does NOT catch, so one diverged trial would
            # abort the whole search. Pruning is also the right signal: a prune is scored worst and
            # steers TPE away from that region, while a failed trial teaches the sampler nothing.
            if not torch.isfinite(loss):
                print(f"Trial {trial.number}: pruned at epoch {epoch} — loss diverged to "
                      f"{loss.item()}")
                raise optuna.TrialPruned()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # ---- Validation ----
        model.eval()
        y_true = []
        y_pred = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch,
                            **_nrank_kwargs(batch))
                mask = _split_mask(batch, val_loader)
                y_true.append(batch.y[mask])
                y_pred.append(out[mask])

        result = evaluator.eval({
            'y_true': torch.cat(y_true, dim=0),
            'y_pred': torch.cat(y_pred, dim=0),
        })[evaluator.eval_metric]
        all_results.append(result)
        if scheduler is not None:
            scheduler.step(result)

        # ---- Pruning ----
        trial.report(result, step=epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        # ---- Early stopping: stop once the validation metric has saturated ----
        # An improvement counts only if it clears `min_delta` over the best so far, so
        # sub-`min_delta` wiggles near saturation don't keep resetting the patience counter.
        #
        # Off at node level. `patience` is counted in epochs, which is 20 optimizer steps there —
        # 2% of the budget, against the thousands of steps it buys at graph level — and its
        # calibration is void anyway: patience=20 was chosen as twice ReduceLROnPlateau's 10 so a
        # trial is scored AFTER the LR drop, and the node path runs no scheduler to drop. Rescaling
        # would need `min_delta` rescaled with it (one step moves the metric far less than one
        # graph-level epoch), inventing two numbers no reference supports; the full-graph protocols
        # run their fixed step budget without early stopping. The epoch cap and the pruner bound the
        # cost instead, and a node trial's score (its best epoch, below) can only improve by running
        # on — stopping early could only lose the optimum, never find it sooner.
        if best_metric is None or (result > best_metric + min_delta if mode == 'max'
                                   else result < best_metric - min_delta):
            best_metric = result
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if not node_level and epochs_no_improve >= patience:
            break

    # A node trial is scored by the best ROLLING MEDIAN over the run, not by the median of its last
    # five epochs. The last-five rule bundles two assumptions: a median resists single-epoch noise
    # (still true here), and the last epochs are where the config has settled (false here — one
    # optimizer step per epoch means the model overfits fast and the validation curve peaks early
    # then decays, measured train ROC-AUC 0.9996 against validation 0.75). Keeping the median and
    # moving WHERE it is taken preserves the noise robustness while measuring the best sustained
    # level the config reached, which is what `utils/logger.py` reports (§ Evaluation protocol).
    # Measured over six configs spanning this space, rank correlation with the reported metric:
    # last-five median -0.20, and still negative at windows of 25/50/100 (the decayed tail says
    # nothing about the peak, and a slow config's tail is HIGHER); rolling median +0.60 at a 333-step
    # cap and +1.00 uncapped, matching a raw max on ranking while a lone spike cannot move a
    # five-point median.
    if node_level:
        window = min(NODE_SCORE_WINDOW, len(all_results))
        rolling = [median(all_results[i:i + window]) for i in range(len(all_results) - window + 1)]
        return max(rolling) if mode == 'max' else min(rolling)

    if epochs < 5:
        return median(all_results)

    return median(all_results[-5:])


def _train_LDNA_code2(
        trial,
        dataset: str,
        hidden_channels: int,
        num_layers: int,
        dropout: float,
        batch_size: int,
        lr: float,
        weight_decay: float,
        epochs: int,
        patience: int,
        min_delta: float,
        device,
        mem_fraction: float,
):
    # ---- Load device ----
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    if device.type == 'cuda':
        torch.cuda.set_per_process_memory_fraction(mem_fraction, device.index)
        torch.cuda.empty_cache()

    # ---- Load data ----
    config = _get_training_config(
        dataset=dataset,
        batch_size=batch_size,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        dropout=dropout,
    )

    # ---- Instantiate model ----
    model, train_loader, val_loader, _ = model_and_data_resolver(
        'LDNA',
        dataset,
        model_args=config['model_args'],
        data_args=config['data_args'],
    )
    # code2: per-position CE loss and the OGB F1 evaluator built from the model's vocabulary.
    criterion = nn.CrossEntropyLoss()
    evaluator = Code2Evaluator(name='ogbg-code2', idx2vocab=model.idx2vocab)

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=config['scheduler_mode'])

    all_results = []
    mode = config['scheduler_mode']
    best_metric = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        # ---- Training ----
        model.train()

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred_list = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss = sum(criterion(pred_list[i].float(), batch.y_arr[:, i])
                       for i in range(len(pred_list))) / len(pred_list)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # ---- Validation ----
        model.eval()
        seq_ref = []
        seq_pred = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                pred_list = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                mat = torch.stack([p.argmax(dim=1) for p in pred_list], dim=1)
                seq_pred.extend(evaluator.decode(mat))
                seq_ref.extend([batch.y[i] for i in range(len(batch.y))])

        result = evaluator.eval(seq_ref, seq_pred)[evaluator.eval_metric]
        all_results.append(result)
        scheduler.step(result)

        # ---- Pruning ----
        trial.report(result, step=epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        # ---- Early stopping: stop once the validation metric has saturated ----
        if best_metric is None or (result > best_metric + min_delta if mode == 'max'
                                   else result < best_metric - min_delta):
            best_metric = result
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        if epochs_no_improve >= patience:
            break

    if epochs < 5:
        return median(all_results)

    return median(all_results[-5:])


def objective(trial, dataset: str, epochs: int = 50, patience: int = 20,
              min_delta: float = 1e-3, device: str = 'cuda:0',
              mem_fraction: float = 0.30,
              fix_hidden_channels=None, fix_num_layers=None) -> float:
    # --- Search spaces for the hyperparameters ---
    # Architecture can be pinned via `fix_*` to search only the training knobs (dropout/lr/wd).
    # This is faster: it drops the expensive large-model trials and shrinks the search space,
    # while keeping the pinned width/depth as a shared value for the fairness protocol.
    #
    # The node-level datasets get their own (width, depth) box. 64 is included because the wired
    # datasets can use it: the floor exists only to keep `num_pred_heads < out_channels`, which binds
    # at 121 heads but not at 40 (ogbn-arxiv) or 1 (Questions), and it is the cheapest width that
    # keeps the wide/deep corner of the arxiv box feasible at all. 256 is the widest that fits arxiv
    # full-graph. Depth is per dataset, the measured k-hop covering depth + 2: Questions [2, 6],
    # ogbn-arxiv [2, 7] (§ Datasets → node-level).
    node_level = dataset in NODE_DATASETS
    hidden_channels = fix_hidden_channels if fix_hidden_channels is not None \
        else trial.suggest_categorical('hidden_channels',
                                       [64, 128, 256] if node_level else [128, 256, 512, 1024])
    num_layers = fix_num_layers if fix_num_layers is not None \
        else trial.suggest_int('num_layers', 2,
                               {'Questions': 6, 'ogbn-arxiv': 7}.get(dataset, 8))
    dropout = trial.suggest_float('dropout', 0.1, 0.7)
    # Per-dataset batch size (power of 2), sized to keep steps/epoch reasonable; not searched.
    # ZINC search uses the subset (~10k graphs), so a smaller batch than the full-data configs.
    # The node-level datasets train full-graph — one graph is one batch — so they have none, and
    # `_get_training_config` leaves `batch_size` out of their `data_args`.
    batch_size = {'ogbg-molhiv': 256, 'ogbg-molpcba': 512, 'ZINC': 128, 'MNISTSuperpixels': 256,
                  'ogbg-code2': 128, 'Questions': None, 'ogbn-arxiv': None}[dataset]
    # Node level widens the lr range in both directions. The published full-graph arxiv configs sit
    # at 1e-2 (OGB's own node example, tunedGNN), which is the graph-level range's exact ENDPOINT,
    # and a log-uniform prior over [1e-5, 1e-2] puts only ~14% of its mass above 3e-3 — TPE would
    # rarely settle on a boundary it can only approach. The upper bound is 5e-2 rather than 3e-2
    # because the choice costs almost nothing and the boundary should not rest on an assumption:
    # measured, widening moves 3 trials in 100 out of the band our own probes found best
    # ([3e-4, 1e-2] holds 56/100 draws at 3e-2 against 53/100 at 5e-2), and lr optima shift with
    # width and depth, which we only spot-checked at two widths and one depth. A lr that diverges is
    # pruned in the training loop, so the extra headroom cannot cost more than a pruned trial.
    # `weight_decay` is left alone: every large-node-graph reference uses exactly 0, which this range
    # cannot express, but over a fixed step budget 1e-6 is numerically indistinguishable from it.
    lr = trial.suggest_float('lr', 1e-4 if node_level else 1e-5, 5e-2 if node_level else 1e-2,
                             log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)

    # --- Memory feasibility ---
    # Full-graph node training materializes every edge's activations, so the wide/deep corner of the
    # box can exceed the card. Predicting the peak and pruning HERE is what keeps an infeasible
    # config visible: left to OOM it would be swallowed by `study.optimize(catch=(RuntimeError,))`
    # as a failed trial, and the sampler would keep proposing from a region it never scored.
    # Gated on the REQUESTED device being a GPU, not on one merely being present: `torch.device('cpu')`
    # has index None, which `get_device_properties` resolves to GPU 0, so a CPU run would otherwise be
    # pruned against a card it is not using.
    if node_level and torch.device(device).type == 'cuda':
        budget = mem_fraction * torch.cuda.get_device_properties(
            torch.device(device).index).total_memory / 1024 ** 3
        peak = _node_peak_gib(dataset, hidden_channels, num_layers)
        if peak > budget:
            print(f"Trial {trial.number}: pruned as infeasible — predicted peak {peak:.2f} GiB "
                  f"exceeds the {budget:.2f} GiB budget (hidden_channels={hidden_channels}, "
                  f"num_layers={num_layers})")
            raise optuna.TrialPruned()

    # --- Train model with these hyperparameters ---
    result = train_LDNA(
        trial=trial,
        dataset=dataset,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        dropout=dropout,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        epochs=epochs,
        patience=patience,
        min_delta=min_delta,
        device=device,
        mem_fraction=mem_fraction,
    )

    return result


def main(args):
    log_path = os.path.join(args.log_dir, f'search_{args.dataset}.txt') if args.log_dir is not None else None
    with tee_to_file(log_path):
        settings = _get_study_settings(args.dataset)

        # Epoch cap, resolved per level unless the CLI names one: a node-level epoch is a single
        # optimizer step, so the graph-level cap of 50 is not the same amount of training there. At
        # node level a trial runs the dataset's full budget (see NODE_SEARCH_EPOCHS).
        node_level = args.dataset in NODE_DATASETS
        epochs = args.epochs if args.epochs is not None \
            else (NODE_SEARCH_EPOCHS[args.dataset] if node_level else 50)

        # `mem_fraction` also resolves per level. The default of 0.30 exists so a graph-level search
        # can share a card; at node level it is a hard constraint on which ARCHITECTURES the search
        # can see at all, because the feasibility prune below measures every (width, depth) against
        # it. At 0.30 of a 47.4 GiB card the arxiv box collapses to a single admissible point, so the
        # search would silently stop searching width and depth. A node search therefore takes the
        # card (0.85), the same value the node configs use; pass --mem_fraction explicitly to
        # override when co-locating, and read the admissible-set report printed below before doing so.
        mem_fraction = args.mem_fraction if args.mem_fraction is not None \
            else (0.85 if node_level else 0.30)

        # Create a study object. `n_warmup_steps` is counted in epochs and so inherits the same unit
        # mismatch as the cap: 10 epochs is a fifth of the graph-level cap but a thirtieth of the
        # node one, where the pruner would rank trials before any had trained. Held at the same
        # fraction of the cap on both levels (graph-level with the default cap is unchanged at 10).
        study = optuna.create_study(
            direction=settings['direction'],
            sampler=optuna.samplers.TPESampler(seed=args.seed),
            pruner=optuna.pruners.MedianPruner(
                n_warmup_steps=round(0.2 * epochs) if node_level else 10,  # No pruning while warming up
                interval_steps=1),  # Check for pruning every epoch
        )

        # Optimize the objective function for N trials. `catch=(RuntimeError,)` keeps the
        # study alive when an individual trial fails (e.g. a CUDA OOM from a large sampled
        # model while sharing the GPU) — that trial is marked failed and the search continues
        # instead of aborting the whole run.
        # Report the architecture box the memory prune actually admits, BEFORE spending anything. The
        # prune is what keeps an infeasible config visible rather than silently OOMed, but an over-
        # tight budget turns it into the opposite failure: a box narrowed to a point looks like a
        # normal search while no longer searching width or depth. An empty box is fatal outright.
        if node_level:
            box = [(h, l) for h in (64, 128, 256)
                   for l in range(2, {'Questions': 6, 'ogbn-arxiv': 7}[args.dataset] + 1)
                   if _node_peak_gib(args.dataset, h, l)
                   <= mem_fraction * torch.cuda.get_device_properties(
                       torch.device(args.device).index).total_memory / 1024 ** 3]
            print(f"Memory-admissible (hidden_channels, num_layers) at mem_fraction "
                  f"{mem_fraction}: {len(box)} of "
                  f"{3 * ({'Questions': 6, 'ogbn-arxiv': 7}[args.dataset] - 1)} — {box}")
            if not box:
                raise ValueError(
                    f"No (hidden_channels, num_layers) point fits {mem_fraction} of "
                    f"{args.device}; raise --mem_fraction or use a freer card")

        study.optimize(
            lambda trial: objective(trial, dataset=args.dataset, device=args.device,
                                    epochs=epochs, patience=args.patience,
                                    min_delta=args.min_delta, mem_fraction=mem_fraction,
                                    fix_hidden_channels=args.fix_hidden_channels,
                                    fix_num_layers=args.fix_num_layers),
            n_trials=args.n_trials,
            catch=(RuntimeError,),
        )

        # `study.best_trial` raises if NOTHING completed, and it is outside study.optimize so
        # `catch=(RuntimeError,)` does not apply — an all-pruned or all-failed study would abort here
        # and discard every result the run did produce. Report the breakdown instead.
        completed = study.get_trials(states=(optuna.trial.TrialState.COMPLETE,), deepcopy=False)
        if not completed:
            states = Counter(t.state.name for t in study.trials)
            print(f"No trial completed. States: {dict(states)}. Nothing to report — a search that "
                  f"prunes or fails everything usually means the memory budget or the search space "
                  f"is wrong, not the model.")
            return study

        best_trial = study.best_trial
        print(f"Number of finished trials: {len(study.trials)}")
        print("Best trial:")
        print(f"  Value ({settings['best_value_label']}): {best_trial.value}")
        print("  Params: ")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}")

    return study


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        choices=['ZINC', 'ogbg-molhiv', 'ogbg-molpcba', 'MNISTSuperpixels', 'ogbg-code2',
                                 'Questions', 'ogbn-arxiv'],
                        required=True)
    parser.add_argument('--device', default='cuda:0')
    # Cap on this process's share of the (shared) GPU. Lower it on cards with an extra
    # co-tenant so a large sampled model can't OOM another user's job (see § GPU policy).
    # Left unset it resolves per level in main(): 0.85 for node datasets (a node search's admissible
    # architecture box is defined by this value), 0.30 for graph-level.
    parser.add_argument('--mem_fraction', type=float, default=None)
    # Left unset the cap resolves per level in main(): 50 graph-level epochs, or NODE_SEARCH_EPOCHS
    # on the node-level datasets, where one epoch is one optimizer step.
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--min_delta', type=float, default=1e-3)
    parser.add_argument('--n_trials', type=int, default=100)
    # Pin architecture to search only the training knobs (dropout/lr/wd); leave unset to search
    # width/depth as well. Faster search when set (drops large-model trials, lower-dim space).
    parser.add_argument('--fix_hidden_channels', type=int, default=None)
    parser.add_argument('--fix_num_layers', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log_dir', type=str, default='out/logs/')

    main(parser.parse_args())
