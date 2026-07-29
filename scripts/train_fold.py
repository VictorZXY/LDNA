# Train a single CV fold once per seed and keep the val-selected best-of-N.
#
# Works for any dataset whose resolver builds cross-validation loaders (currently EXP/CEXP: the
# fold list rides on the returned train_loader as `cv_folds`). Each seed re-seeds and rebuilds
# the full construction (dataset order, encoder/model init, loader shuffling), then trains ONLY
# the requested fold as a single run through the unmodified train.train loop. The seeds form a
# self-contained candidate family — NOT a continuation of a full sweep's RNG stream — so the
# reported fold number REPLACES a sweep's fold number rather than extending its candidate set
# (which would make this fold best-of-(N+M) while every other fold stays best-of-N). The
# per-fold best-of-N protocol behind the committed EXP numbers is in `.claude/experiments.md`.
#
# Usage (from the repo root):
#   python -m scripts.train_fold --config configs/gin_exp.yaml --fold 10 --seeds 45 46
import json
import os

import configargparse
import torch
import torch_geometric
import yaml
from configargparse import YAMLConfigFileParser

from train import train
from utils import model_and_data_resolver, tee_to_file


def main(args):
    # Fold-specific name, derived rather than left to an operator flag: tee_to_file opens 'w', so
    # inheriting the config's experiment_name verbatim could truncate the committed full-sweep log
    # if this driver is ever pointed at the archive directory.
    experiment_name = f'{args.experiment_name}_fold{args.fold}'
    log_path = os.path.join(args.log_dir, f'{experiment_name}.txt') if args.log_dir is not None else None
    with tee_to_file(log_path):
        if args.cuda != -1:
            device = torch.device('cuda:' + str(args.cuda) if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device('cpu')
        print(f"Device: {device}")

        # Cap this process's share of the (shared) GPU, mirroring train.py's § GPU policy guard.
        if device.type == 'cuda':
            torch.cuda.set_per_process_memory_fraction(args.mem_fraction, device.index)

        # train.py maps runs one-to-one onto folds; here the fold is fixed and the seeds vary,
        # so each candidate is a single run and the config's `runs` is overridden.
        train_args = dict(args.train_args)
        train_args['runs'] = 1

        print(f"Dataset name: {args.dataset}")
        print(f"Model name: {args.model}")
        print(f"Fold: {args.fold}, seeds: {args.seeds}")

        per_seed = []
        early_stopped = False
        mode = None
        for i, seed in enumerate(args.seeds):
            print(f"=== Fold {args.fold}, seed {seed} ===")
            # Re-seed and rebuild everything per seed. The resolver mutates the dicts it is given
            # (it pops `batch_size` and inserts the encoders), so each iteration passes fresh
            # copies of the config dicts.
            torch_geometric.seed_everything(seed)
            model, train_loader, _, _ = model_and_data_resolver(
                args.model, args.dataset,
                model_args=dict(args.model_args or {}), data_args=dict(args.data_args or {}))
            if getattr(train_loader, 'cv_folds', None) is None:
                raise ValueError(f"Dataset '{args.dataset}' does not build CV fold loaders")
            fold_train, fold_val, fold_test = train_loader.cv_folds[args.fold - 1]
            # Fold 1's train loader is the very loader the resolver tagged with the full fold
            # list; strip the tag so train() runs it as the single fold it is here.
            fold_train.cv_folds = None
            _, logger, _ = train(model, fold_train, fold_val, fold_test, train_args, device)
            # Selection follows the metric direction the Logger resolved from the evaluator
            # (max for accuracy-style metrics, min for MAE-style).
            mode = logger.mode
            best_of = torch.max if mode == 'max' else torch.min

            result = {k: torch.tensor(v) for k, v in logger.results[0].items()}
            per_seed.append({'seed': seed,
                             'best_train': best_of(result['train']).item(),
                             'best_val': best_of(result['val']).item(),
                             'best_test': best_of(result['test']).item()})

            # Max-direction metrics in this repo are bounded above by 1.0 (accuracy, ROC-AUC, AP),
            # so once a seed reaches that ceiling on val no later candidate can displace it under
            # pure val-selection — skip the remaining seeds. Min-direction metrics have no
            # attainable floor, so every seed runs.
            if mode == 'max' and per_seed[-1]['best_val'] >= 1.0:
                early_stopped = i + 1 < len(args.seeds)
                if early_stopped:
                    print(f"Seed {seed} reached val 1.0; skipping remaining seeds {args.seeds[i + 1:]}")
                break

        # Pure val-selection, first optimum (lowest-index seed) on ties — the convention behind
        # the committed per-fold best-of-N numbers, and what makes the ceiling early stop above
        # exactly equivalent to running all N seeds.
        select = max if mode == 'max' else min
        best = select(per_seed, key=lambda r: r['best_val'])
        print(f"Fold {args.fold} best-of-{len(args.seeds)} (ran {len(per_seed)}): seed {best['seed']}, "
              f"val {best['best_val']:.4f}, test {best['best_test']:.4f}")

        # Per-seed fold results as JSON (like scripts/pairwise_eval.py): a single-fold run does not
        # fit the Logger pickle's runs-by-epochs shape, whose runs are folds on these datasets.
        if args.log_dir is not None:
            os.makedirs(args.log_dir, exist_ok=True)
            out = {
                'experiment_name': experiment_name,
                'model': args.model,
                'dataset': args.dataset,
                'fold': args.fold,
                'seeds_requested': list(args.seeds),
                'early_stopped': early_stopped,
                'per_seed': per_seed,
                'best': best,
            }
            with open(os.path.join(args.log_dir, f'{experiment_name}_results.json'), 'w') as f:
                json.dump(out, f, indent=2)


if __name__ == '__main__':
    parser = configargparse.ArgParser(config_file_parser_class=YAMLConfigFileParser)
    parser.add_argument('--config', is_config_file=True)
    parser.add_argument('--experiment_name', type=str, required=True)

    # Model specific arguments/hyperparameters
    parser.add_argument('--model', default='LDNA', type=str)
    parser.add_argument('--model_args', default=None, type=yaml.safe_load)

    # Dataset specific arguments/hyperparameters
    parser.add_argument('--dataset', default='EXP', type=str)
    parser.add_argument('--data_args', default=None, type=yaml.safe_load)

    # Training specific arguments/hyperparameters. `--seed` and `--checkpoint_dir` are accepted so
    # the train.py configs parse unchanged; the candidate seeds come from `--seeds`, and this
    # driver saves no checkpoint.
    parser.add_argument('--cuda', default=-1, type=int)
    parser.add_argument('--mem_fraction', default=0.30, type=float)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--train_args', default=None, type=yaml.safe_load)
    parser.add_argument('--checkpoint_dir', default=None, type=str)
    parser.add_argument('--log_dir', default='out/logs/', type=str)

    # Fold is 1-based, matching the run numbering in the train.py logs.
    parser.add_argument('--fold', type=int, required=True)
    parser.add_argument('--seeds', default=[42, 43, 44, 45, 46], type=int, nargs='+')

    args = parser.parse_args()
    main(args)
