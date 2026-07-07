"""Broadcast a tuned LDNA config to all baselines for one dataset.

Fairness by shared hyperparameters (see .claude/experiments.md): after tuning LDNA, the
tuned SHARED knobs (hidden_channels, out_channels, num_layers, dropout, lr, weight_decay)
are written verbatim into every model's config for that dataset; model-internal-only
settings are left at each model's existing value (preserved). GIN vs GINE is chosen per
dataset by edge-feature availability (edge -> GINE, edge-less -> GIN); a `GNN-VPA` config
is created from the GINE/GIN template.

This edits configs in place via targeted line replacement (keeps comments/formatting) and
round-robins the `cuda` field across the given GPUs so the runs can be launched in parallel.

Usage:
  python broadcast.py --dataset ogbg-molhiv --hidden_channels 256 --num_layers 4 \
      --dropout 0.05 --lr 6.5e-5 --weight_decay 3e-4 --gpus 0,1,2,3 [--runs 5] [--dry_run]
"""
import argparse
import glob
import os
import re

SUFFIX = {'ogbg-molhiv': 'hiv', 'ogbg-molpcba': 'molpcba', 'ZINC': 'zinc',
          'MNISTSuperpixels': 'mnist', 'ogbg-code2': 'code2'}
HAS_EDGE = {'ogbg-molhiv': True, 'ogbg-molpcba': True, 'ZINC': True,
            'MNISTSuperpixels': False, 'ogbg-code2': False}


def _fmt(v):
    # compact, YAML-friendly formatting (e.g. 6.5e-05, 0.05, 256)
    if isinstance(v, float):
        return f'{v:.6g}'
    return str(v)


def _set_key(text, key, value, indent):
    """Replace the value of `<indent spaces>key: ...` on its own line; return (text, n_hits)."""
    pat = re.compile(rf'^({" " * indent}{re.escape(key)}: ).*$', re.M)
    return pat.subn(lambda m: m.group(1) + _fmt(value), text)


def _make_vpa_config(configs_dir, suffix, edge):
    """Create configs/vpa_<suffix>.yaml from the GINE (edge) / GIN (no-edge) template."""
    tmpl_model = 'gine' if edge else 'gin'
    src = os.path.join(configs_dir, f'{tmpl_model}_{suffix}.yaml')
    dst = os.path.join(configs_dir, f'vpa_{suffix}.yaml')
    if not os.path.exists(src):
        raise FileNotFoundError(f'template {src} not found')
    text = open(src).read()
    text = re.sub(r'^experiment_name: .*$', f'experiment_name: vpa_{suffix}', text, flags=re.M)
    text = re.sub(r'^model: .*$', 'model: GNN-VPA', text, flags=re.M)
    return dst, text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=list(SUFFIX))
    ap.add_argument('--hidden_channels', type=int, required=True)
    ap.add_argument('--num_layers', type=int, required=True)
    ap.add_argument('--dropout', type=float, required=True)
    ap.add_argument('--lr', type=float, required=True)
    ap.add_argument('--weight_decay', type=float, required=True)
    ap.add_argument('--gpus', default='0,1,2,3', help='comma-separated GPU ids to round-robin over')
    ap.add_argument('--runs', type=int, default=None, help='override train_args.runs (optional)')
    ap.add_argument('--configs_dir', default='configs')
    ap.add_argument('--dry_run', action='store_true')
    args = ap.parse_args()

    suffix = SUFFIX[args.dataset]
    edge = HAS_EDGE[args.dataset]
    gpus = [g.strip() for g in args.gpus.split(',') if g.strip()]

    # existing baseline + ldna configs for this dataset, plus a fresh vpa config
    files = sorted(glob.glob(os.path.join(args.configs_dir, f'*_{suffix}.yaml')))
    vpa_dst, vpa_text = _make_vpa_config(args.configs_dir, suffix, edge)
    if not args.dry_run:
        open(vpa_dst, 'w').write(vpa_text)
    if vpa_dst not in files:
        files.append(vpa_dst)
    files = sorted(files)

    updates = {'hidden_channels': (args.hidden_channels, 2), 'out_channels': (args.hidden_channels, 2),
               'num_layers': (args.num_layers, 2), 'dropout': (args.dropout, 2),
               'lr': (args.lr, 4), 'weight_decay': (args.weight_decay, 4)}
    if args.runs is not None:
        updates['runs'] = (args.runs, 2)

    print(f'# broadcast {args.dataset} -> {len(files)} configs '
          f'(edge={edge}, GIN{"E" if edge else ""}); gpus={gpus}')
    for i, f in enumerate(files):
        text = vpa_text if f == vpa_dst else open(f).read()
        for key, (val, ind) in updates.items():
            text, n = _set_key(text, key, val, ind)
            if n == 0 and key in ('hidden_channels', 'out_channels', 'num_layers', 'dropout', 'lr', 'weight_decay'):
                print(f'  WARNING {os.path.basename(f)}: key `{key}` not found')
        cuda = gpus[i % len(gpus)]
        text, _ = _set_key(text, 'cuda', int(cuda), 0)
        if not args.dry_run:
            open(f, 'w').write(text)
        print(f'  {os.path.basename(f):28s} cuda:{cuda}')
    if args.dry_run:
        print('# (dry run — nothing written)')


if __name__ == '__main__':
    main()
