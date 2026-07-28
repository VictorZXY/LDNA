import json
import os

import configargparse
import torch
import torch_geometric
import yaml
from configargparse import YAMLConfigFileParser
from torch import nn

import models
from utils import add_eig_vecs, degree_histogram, tee_to_file
from utils.brec import BRECDataset, OUTPUT_DIM, SEEDS, evaluate


class BRECModel(nn.Module):
    """Wrap a repo backbone into the BREC test interface.

    A backbone built with `num_pred_heads=None` returns the pooled graph embedding, which a linear
    head maps to the protocol's `OUTPUT_DIM`. `forward(data)` matches the BREC calling convention
    `model(data)`; `eig_vec` is forwarded only when the batch carries it (DGN), mirroring
    `train.py`'s presence-based `_eig_kwargs`. `reset_parameters()` re-initializes both parts, which
    the per-pair protocol calls before each comparison.
    """

    def __init__(self, backbone, out_channels, output_dim=OUTPUT_DIM):
        super(BRECModel, self).__init__()
        self.backbone = backbone
        self.head = nn.Linear(out_channels, output_dim)
        self.reset_parameters()

    def reset_parameters(self):
        self.backbone.reset_parameters()
        self.head.reset_parameters()

    def forward(self, data):
        edge_attr = getattr(data, 'edge_attr', None)
        eig_kwargs = {'eig_vec': data.eig_vec} if hasattr(data, 'eig_vec') else {}
        x = self.backbone(data.x, data.edge_index, edge_attr, data.batch, **eig_kwargs)
        return self.head(x)


def build_model(model_query, model_args, deg=None):
    # Backbones are built directly (not through utils/resolver, whose dataset-coupled build path is
    # for the graph-level tasks). BREC graphs are featureless, so the node encoder is nn.Linear(1,
    # emb) and there is no edge encoder; num_pred_heads=None makes the backbone return the pooled
    # embedding that BRECModel's head turns into the OUTPUT_DIM comparison vector.
    model_args = dict(model_args)
    embedding_dim = model_args['hidden_channels']
    out_channels = model_args['out_channels']
    model_args.update({
        'in_channels': embedding_dim,
        'node_encoder': nn.Linear(1, embedding_dim),
        'num_pred_heads': None,
    })

    if model_query == 'LDNA':
        backbone = models.LDNA(**model_args)
    elif model_query == 'GIN':
        backbone = models.GIN(**model_args)
    elif model_query == 'DGN':
        backbone = models.DGN(deg=deg, **model_args)
    else:
        raise ValueError(f"Could not resolve model '{model_query}' among choices ['LDNA', 'GIN', 'DGN']")

    return BRECModel(backbone, out_channels)


def _print_summary(results):
    # Mean distinguished counts across the reruns, per category and overall.
    from statistics import mean, pstdev
    categories = list(results[0]['per_category'])
    print("=== Summary across seeds ===")
    for cat in categories:
        vals = [r['per_category'][cat]['correct'] for r in results]
        total = results[0]['per_category'][cat]['total']
        print(f"{cat}: {mean(vals):.1f} +/- {pstdev(vals):.1f} / {total}")
    correct = [r['correct'] for r in results]
    reliable = [r['reliable_correct'] for r in results]
    fail = [r['fail_reliability'] for r in results]
    total = results[0]['total']
    print(f"Correct: {mean(correct):.1f} +/- {pstdev(correct):.1f} / {total}")
    print(f"Reliable-correct: {mean(reliable):.1f} +/- {pstdev(reliable):.1f} / {total}")
    print(f"Fail in reliability: {mean(fail):.1f} +/- {pstdev(fail):.1f} / {total}")


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

        model_args = args.model_args or {}
        # batch_size is protocol-fixed (BATCH_SIZE = 16 in utils/brec.py), so it is not a data knob.
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

        _print_summary(results)
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

    # BREC-specific: the RPC-protocol rerun seeds, the S_epsilon covariance guard, and a pair cap
    # for smoke/debug (0 = all SAMPLE_NUM pairs).
    parser.add_argument('--seeds', default=None, type=int, nargs='+')
    parser.add_argument('--s_epsilon', default=True, type=yaml.safe_load)
    parser.add_argument('--pair_limit', default=0, type=int)

    parser.add_argument('--log_dir', default='out/logs/', type=str)

    args = parser.parse_args()
    torch_geometric.seed_everything(args.seed)

    main(args)
