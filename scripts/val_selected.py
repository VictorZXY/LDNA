# Validation-selected test metric, recomputed offline from saved training curves.
#
# The project reports `best_test` (utils/logger.py: per run, the best test value over all epochs,
# then mean ± std over runs). This prints, next to it, the val-selected test: per run, the test
# value at the epoch of the best validation value — first occurrence on ties, i.e. torch's
# argmax/argmin, exactly the commented-out `final_test` lines in utils/logger.py — aggregated the
# same way (torch's default unbiased std). Nothing is retrained and nothing is written.
#
# Inputs, per file:
#   <name>_logger.pickle  the pickled Logger train.py writes when a job completes: exact per-epoch
#                         curves.
#   <name>.txt            train.py's tee'd log, for jobs that never wrote the pickle. It prints val
#                         and test only every `eval_interval` epochs, so val-selection runs on that
#                         grid and is marked grid-approximate; best_test is taken from the exact
#                         `Run NN: Best ...` line. A run without that line (killed or stopped before
#                         it finished) is left out.
# A `:RUNS` suffix keeps only the listed 1-based runs of a file (e.g. `dgn_cifar10.txt:1,2`), and
# --pool merges every listed file into one cell, for reported cells assembled from several jobs. A
# single-run pickle split off a job can hold the same run its `.txt` already prints (the code2
# `*_run2` pickles do), so --pool refuses two runs from different files whose val and test curves
# agree at every epoch they share — select runs with `:RUNS` instead — and refuses the same run of
# the same file listed twice.
#
# Usage (from the repo root, so `utils.logger.Logger` unpickles):
#   python -m scripts.val_selected out/logs/ranking/zinc/*_logger.pickle
#   python -m scripts.val_selected --pool out/logs/ranking/cifar10/dgn_cifar10.txt:1,2 \
#       out/logs/ranking/cifar10/dgn_cifar10_s43_logger.pickle
import argparse
import os
import pickle
import re

import torch

EPOCH_LINE = re.compile(r'^Run (\d+), Epoch\s+(\d+)/\s*(\d+), .*Train (\S+): \S+, '
                        r'Val (\S+): (\S+), Test (\S+): (\S+)$')
BEST_LINE = re.compile(r'^Run (\d+): Best train (\S+): \S+, Best val (\S+): (\S+), Best test (\S+): (\S+)$')
# The `.txt` prints 4 decimals, so a run logged in both a pickle and a `.txt` agrees to within half a
# unit in the last place.
SAME_RUN_TOL = 5.01e-5


def _mode(eval_metric):
    # Same rule as Logger.__init__.
    return 'min' if eval_metric.lower() in ['mae'] else 'max'


def _split_spec(spec):
    """`path` or `path:1,2` -> (path, [0-based run indices] or None)."""
    path, sep, runs = spec.rpartition(':')
    if sep and re.fullmatch(r'\d+(,\d+)*', runs):
        keep = [int(r) - 1 for r in runs.split(',')]
        if len(set(keep)) != len(keep):
            raise ValueError(f"{spec}: run index listed twice")
        return path, keep
    return spec, None


def _select(result, mode):
    """Per-run (best_test, val-selected test, selected epoch), mirroring Logger.print_statistics."""
    result = {k: torch.tensor(v) for k, v in result.items()}
    if mode == 'max':
        arg = result['val'].argmax().item()
        best_test = result['test'].max().item()
    else:
        arg = result['val'].argmin().item()
        best_test = result['test'].min().item()
    return best_test, result['test'][arg].item(), arg


def load_pickle(path):
    with open(path, 'rb') as f:
        logger = pickle.load(f)
    runs = []
    for i, result in enumerate(logger.results):
        if len(result['test']) == 0:
            continue
        best_test, val_sel, arg = _select(result, logger.mode)
        runs.append({'run': i + 1, 'best_test': best_test, 'val_selected': val_sel,
                     'epoch': arg + 1, 'num_epochs': len(result['test']), 'grid': False,
                     'curve': {'epochs': list(range(1, len(result['test']) + 1)),
                               'val': list(result['val']), 'test': list(result['test'])}})
    return logger.eval_metric, logger.mode, runs


def parse_txt(path):
    """Parse train.py's tee'd log into {run: {'epochs', 'val', 'test', 'best_test'}} and the metric."""
    curves, eval_metric = {}, None
    with open(path) as f:
        for line in f:
            line = line.rstrip('\n')
            m = EPOCH_LINE.match(line)
            if m is not None:
                run, epoch = int(m.group(1)), int(m.group(2))
                eval_metric = eval_metric or m.group(4)
                if not (m.group(4) == m.group(5) == m.group(7) == eval_metric):
                    raise ValueError(f"{path}: metric name differs from '{eval_metric}' in line {line!r}")
                c = curves.setdefault(run, {'epochs': [], 'val': [], 'test': [], 'best_test': None})
                if c['best_test'] is not None or (c['epochs'] and epoch <= c['epochs'][-1]):
                    raise ValueError(f"{path}: run {run} epoch {epoch} out of order (log concatenated?)")
                c['epochs'].append(epoch)
                c['val'].append(float(m.group(6)))
                c['test'].append(float(m.group(8)))
                continue
            m = BEST_LINE.match(line)
            if m is not None:
                run = int(m.group(1))
                eval_metric = eval_metric or m.group(2)
                if not (m.group(2) == m.group(3) == m.group(5) == eval_metric):
                    raise ValueError(f"{path}: metric name differs from '{eval_metric}' in line {line!r}")
                c = curves.setdefault(run, {'epochs': [], 'val': [], 'test': [], 'best_test': None})
                if c['best_test'] is not None:
                    raise ValueError(f"{path}: run {run} has two `Best` lines")
                c['best_test'] = float(m.group(6))
    if eval_metric is None:
        raise ValueError(f"{path}: no training lines found")
    return eval_metric, curves


def load_txt(path):
    eval_metric, curves = parse_txt(path)
    mode = _mode(eval_metric)
    runs = []
    for run, c in sorted(curves.items()):
        if c['best_test'] is None:  # killed or stopped before the run finished
            continue
        _, val_sel, arg = _select({'val': c['val'], 'test': c['test']}, mode)
        runs.append({'run': run, 'best_test': c['best_test'], 'val_selected': val_sel,
                     'epoch': c['epochs'][arg], 'num_epochs': c['epochs'][-1], 'grid': True,
                     'curve': {k: c[k] for k in ('epochs', 'val', 'test')}})
    return eval_metric, mode, runs


def load(spec):
    """Load one `path[:RUNS]` spec -> (eval_metric, mode, runs), each run tagged with its source."""
    path, keep = _split_spec(spec)
    eval_metric, mode, runs = load_pickle(path) if path.endswith('.pickle') else load_txt(path)
    if not runs:
        raise ValueError(f"{path}: no finished runs")
    if keep is not None:
        by_run = {r['run'] - 1: r for r in runs}
        missing = [k + 1 for k in keep if k not in by_run]
        if missing:
            raise ValueError(f"{path}: no finished run(s) {missing}")
        runs = [by_run[k] for k in keep]
    for r in runs:
        r['source'] = path
    return eval_metric, mode, runs


def _same_run(a, b):
    """True if two runs agree on val and test at every epoch both curves carry."""
    common = sorted(set(a['curve']['epochs']) & set(b['curve']['epochs']))
    if not common:
        return False
    va = dict(zip(a['curve']['epochs'], zip(a['curve']['val'], a['curve']['test'])))
    vb = dict(zip(b['curve']['epochs'], zip(b['curve']['val'], b['curve']['test'])))
    return all(abs(va[e][0] - vb[e][0]) <= SAME_RUN_TOL and abs(va[e][1] - vb[e][1]) <= SAME_RUN_TOL
               for e in common)


def check_distinct(runs):
    """Raise if a run is pooled twice: the same file run listed again, or the same run in two files."""
    seen = set()
    for r in runs:
        key = (os.path.realpath(r['source']), r['run'])
        if key in seen:
            raise ValueError(f"{r['source']} run {r['run']} is listed more than once")
        seen.add(key)
    # Curves are compared only across files: runs of one job can be genuinely identical (two runs that
    # diverged to the same constant prediction, e.g. MNIST DeeperGCN-powermean runs 2 and 3).
    for i, a in enumerate(runs):
        for b in runs[i + 1:]:
            if a['source'] != b['source'] and _same_run(a, b):
                raise ValueError(f"{a['source']} run {a['run']} and {b['source']} run {b['run']} have the same "
                                 f"val/test curve; pool only one of them (use :RUNS)")


def summarize(runs):
    """mean ± std over runs of both conventions, as Logger.print_statistics aggregates them."""
    best_test = torch.tensor([r['best_test'] for r in runs])
    val_sel = torch.tensor([r['val_selected'] for r in runs])
    return {'n': len(runs),
            'best_test': (best_test.mean().item(), best_test.std().item()),
            'val_selected': (val_sel.mean().item(), val_sel.std().item()),
            'grid': any(r['grid'] for r in runs)}


def _print_cell(name, eval_metric, mode, runs, per_run):
    s = summarize(runs)
    n_grid = sum(r['grid'] for r in runs)
    if n_grid == 0:
        source = 'pickle (exact)'
    elif n_grid == len(runs):
        source = 'txt grid (approx.)'
    else:
        source = 'pickle + txt grid (approx.)'
    print(f"{name}\n  {eval_metric} ({mode}), n={s['n']}, source: {source}\n"
          f"  best_test:    {s['best_test'][0]:.4f} ± {s['best_test'][1]:.4f}\n"
          f"  val-selected: {s['val_selected'][0]:.4f} ± {s['val_selected'][1]:.4f}")
    if per_run:
        for r in runs:
            print(f"    {r['source']} run {r['run']:02d} [{'grid' if r['grid'] else 'exact'}]: "
                  f"best_test {r['best_test']:.4f}, val-selected {r['val_selected']:.4f} "
                  f"at epoch {r['epoch']}/{r['num_epochs']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('files', nargs='+', help='<name>_logger.pickle or <name>.txt, optionally suffixed :RUNS')
    ap.add_argument('--pool', action='store_true', help='merge the runs of all files into one cell')
    ap.add_argument('--per_run', action='store_true', help='also print each run and its selected epoch')
    args = ap.parse_args()

    loaded = [(spec, *load(spec)) for spec in args.files]
    if args.pool:
        metrics = {(m, mode) for _, m, mode, _ in loaded}
        if len(metrics) != 1:
            raise ValueError(f"--pool needs one metric across files, got {sorted(metrics)}")
        (eval_metric, mode), = metrics
        runs = [r for *_, file_runs in loaded for r in file_runs]
        check_distinct(runs)
        _print_cell(' + '.join(args.files), eval_metric, mode, runs, args.per_run)
    else:
        for spec, eval_metric, mode, runs in loaded:
            _print_cell(spec, eval_metric, mode, runs, args.per_run)


if __name__ == '__main__':
    main()
