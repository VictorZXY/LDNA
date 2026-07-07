import argparse
import os
from statistics import median

import optuna
import torch
from torch import nn

from utils import evaluator_resolver, loss_resolver, model_and_data_resolver, tee_to_file


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
        'readout': 'mean',
    }

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
):
    # ---- Load device ----
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

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
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            if isinstance(loss_fn, nn.CrossEntropyLoss):
                is_labelled = (batch.y == batch.y).view(-1)
                loss = loss_fn(out[is_labelled], batch.y[is_labelled].long())
            else:
                is_labelled = (batch.y == batch.y)
                loss = loss_fn(out[is_labelled], batch.y[is_labelled].float())
            loss.backward()
            optimizer.step()

        # ---- Validation ----
        model.eval()
        y_true = []
        y_pred = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
                y_true.append(batch.y)
                y_pred.append(out)

        result = evaluator.eval({
            'y_true': torch.cat(y_true, dim=0),
            'y_pred': torch.cat(y_pred, dim=0),
        })[evaluator.eval_metric]
        all_results.append(result)
        scheduler.step(result)

        # ---- Pruning ----
        trial.report(result, step=epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        # ---- Early stopping: stop once the validation metric has saturated ----
        # An improvement counts only if it clears `min_delta` over the best so far, so
        # sub-`min_delta` wiggles near saturation don't keep resetting the patience counter.
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
              min_delta: float = 1e-3, device: str = 'cuda:0') -> float:
    # --- Search spaces for the hyperparameters ---
    hidden_channels = trial.suggest_categorical('hidden_channels', [128, 256, 512, 1024])
    num_layers = trial.suggest_int('num_layers', 2, 10)
    dropout = trial.suggest_float('dropout', 0.0, 0.7)
    # Per-dataset batch size (power of 2), sized to keep steps/epoch reasonable; not searched.
    # ZINC search uses the subset (~10k graphs), so a smaller batch than the full-data configs.
    batch_size = {'ogbg-molhiv': 256, 'ogbg-molpcba': 512, 'ZINC': 128}[dataset]
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)

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
    )

    return result


def main(args):
    log_path = os.path.join(args.log_dir, f'search_{args.dataset}.txt') if args.log_dir is not None else None
    with tee_to_file(log_path):
        settings = _get_study_settings(args.dataset)

        # Create a study object
        study = optuna.create_study(
            direction=settings['direction'],
            sampler=optuna.samplers.TPESampler(seed=args.seed),
            pruner=optuna.pruners.MedianPruner(
                n_warmup_steps=10,  # No pruning the first 10 epochs
                interval_steps=1),  # Check for pruning every epoch
        )

        # Optimize the objective function for N trials
        study.optimize(
            lambda trial: objective(trial, dataset=args.dataset, device=args.device,
                                    epochs=args.epochs, patience=args.patience,
                                    min_delta=args.min_delta),
            n_trials=args.n_trials,
        )

        # Print out the best trial
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
    parser.add_argument('--dataset', choices=['ZINC', 'ogbg-molhiv', 'ogbg-molpcba'], required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--min_delta', type=float, default=1e-3)
    parser.add_argument('--n_trials', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--log_dir', type=str, default='logs/')

    main(parser.parse_args())
