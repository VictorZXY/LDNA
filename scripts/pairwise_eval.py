# Pairwise graph-distinguishing evaluation entry point (the RPC protocol).
#
# The pairwise protocol trains and tests per graph pair, so it cannot run through train.py's
# epoch x train/val/test loop. This driver owns only the scaffolding (config/CLI, device,
# logging, JSON results); the protocol itself — dataset, model wrapper, per-pair RPC
# evaluation, summary — lives in `data_utils/brec.py`. BREC is currently the only pairwise
# dataset wired up.
#
# Usage (from the repo root):
#   python -m scripts.pairwise_eval --config configs/ldna_brec.yaml
import json
import os

import configargparse
import torch
import torch_geometric
import yaml
from configargparse import YAMLConfigFileParser

from data_utils import add_eig_vecs, degree_histogram
from data_utils.brec import BRECDataset, SEEDS, build_model, evaluate, print_summary
from utils import tee_to_file


def main(args):
    log_path = os.path.join(args.log_dir, f'{args.experiment_name}.txt') if args.log_dir is not None else None
    with tee_to_file(log_path):
        if args.cuda != -1:
            device = torch.device('cuda:' + str(args.cuda) if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device('cpu')
        print(f"Device: {device}")

        # Cap this process's share of the (shared) GPU, mirroring train.py's § GPU policy guard.
        if device.type == 'cuda':
            torch.cuda.set_per_process_memory_fraction(args.mem_fraction, device.index)

        # The argument exists so the train.py-style configs parse unchanged; only BREC implements
        # the pairwise protocol today.
        if args.dataset != 'BREC':
            raise ValueError(f"Could not resolve dataset '{args.dataset}' among choices ['BREC']")

        model_args = args.model_args or {}
        # batch_size is protocol-fixed (BATCH_SIZE = 16 in data_utils/brec.py), so it is not a data knob.
        data_args = dict(args.data_args or {})
        data_args.pop('batch_size', None)

        dataset = BRECDataset(**data_args)
        print(f"Dataset: BREC ({len(dataset)} graphs)")
        print(f"Model name: {args.model}")

        deg = None
        if args.model == 'DGN':
            # Mirror the resolver ordering: the Laplacian eigenvector field is attached AFTER the
            # canonical order (BRECDataset canonicalizes in process()), and the in-degree histogram
            # the directional scalers need is built over the whole dataset. The sidecar cache name
            # carries the `canonical` flag like the processed filename does, so alternating the
            # flag under one root can never thrash (or overwrite) the other build's cache.
            num_eig_vec = model_args.get('num_eig_vec', 1)
            cache_name = 'all' if dataset.canonical else 'all_noncanon'
            dataset = add_eig_vecs(dataset, k=num_eig_vec, cache_name=cache_name)
            deg = degree_histogram(dataset)

        model = build_model(args.model, model_args, deg=deg)
        print(f"# Params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

        seeds = args.seeds if args.seeds is not None else SEEDS
        print(f"Seeds: {seeds}, s_epsilon: {args.s_epsilon}, pair_limit: {args.pair_limit}")

        results = []
        for seed in seeds:
            print(f"=== Seed {seed} ===")
            results.append(evaluate(model, dataset, device, seed=seed,
                                    s_epsilon=args.s_epsilon, pair_limit=args.pair_limit))

        print_summary(results)
        print(f"Finished BREC evaluation of {args.model}")

        # The BREC output (per-seed per-category counts + reliability failures) does not fit the
        # train/val/test shape of the shared Logger pickle, so it is written as JSON instead.
        if args.log_dir is not None:
            os.makedirs(args.log_dir, exist_ok=True)
            out = {
                'experiment_name': args.experiment_name,
                'model': args.model,
                's_epsilon': args.s_epsilon,
                'pair_limit': args.pair_limit,
                'seeds': list(seeds),
                'results': results,
            }
            with open(os.path.join(args.log_dir, f'{args.experiment_name}_results.json'), 'w') as f:
                json.dump(out, f, indent=2)


if __name__ == '__main__':
    parser = configargparse.ArgParser(config_file_parser_class=YAMLConfigFileParser)
    parser.add_argument('--config', is_config_file=True)
    parser.add_argument('--experiment_name', type=str, required=True)

    # Model specific arguments/hyperparameters
    parser.add_argument('--model', default='LDNA', type=str)
    parser.add_argument('--model_args', default=None, type=yaml.safe_load)

    # Dataset specific arguments/hyperparameters
    parser.add_argument('--dataset', default='BREC', type=str)
    parser.add_argument('--data_args', default=None, type=yaml.safe_load)

    # Training/evaluation specific arguments/hyperparameters
    parser.add_argument('--cuda', default=-1, type=int)
    parser.add_argument('--mem_fraction', default=0.30, type=float)
    parser.add_argument('--seed', default=42, type=int)

    # Protocol-specific: the RPC rerun seeds, the S_epsilon covariance guard, and a per-category
    # pair cap for smoke/probing (0 = all SAMPLE_NUM pairs).
    parser.add_argument('--seeds', default=None, type=int, nargs='+')
    parser.add_argument('--s_epsilon', default=True, type=yaml.safe_load)
    parser.add_argument('--pair_limit', default=0, type=int)

    parser.add_argument('--log_dir', default='out/logs/', type=str)

    args = parser.parse_args()
    torch_geometric.seed_everything(args.seed)

    main(args)
