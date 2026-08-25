#!/usr/bin/env bash
# Run a list of train.py ranking configs sequentially on ONE GPU, so a freed GPU can
# auto-chew through a partition of the ranking queue without babysitting each run.
#
# Usage: bash scripts/run_jobs.sh <gpu_id> <config1.yaml> [config2.yaml ...]
# Each run: train.py --config <cfg> --cuda <gpu_id> (expandable_segments reduces fragmentation on
# the shared card). Logs to out/logs/<name>.run.log.
#
# MEM_FRACTION overrides train.py's 0.30 default, which is 14.22 GiB of a 47.40 GiB A6000. That
# suffices for every dataset the script was written for and does NOT suffice for ogbg-ppa, whose
# batch needs ~16 GiB at the tuned width: ldna_ppa died with 33 GiB free on the card because the
# self-cap, not the card, was the binding limit. Set it per queue when a dataset needs more.
set -u
MEM_FRACTION=${MEM_FRACTION:-}
PY=/home/xz9118/.conda/envs/LDNA/bin/python
EV=/home/xz9118/Projects/LDNA/out/logs/_ranking_events.log
gpu=$1; shift
cd /home/xz9118/Projects/LDNA
for cfg in "$@"; do
  name=$(basename "$cfg" .yaml)
  echo "[run_jobs gpu$gpu] START $name $(date -u +%H:%M:%S)" | tee -a "$EV"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "$PY" -u train.py --config "$cfg" --cuda "$gpu" \
      ${MEM_FRACTION:+--mem_fraction "$MEM_FRACTION"} > "out/logs/${name}.run.log" 2>&1
  echo "[run_jobs gpu$gpu] DONE $name exit=$? $(date -u +%H:%M:%S)" | tee -a "$EV"
done
echo "[run_jobs gpu$gpu] QUEUE DONE ($# configs)" | tee -a "$EV"
