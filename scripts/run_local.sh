#!/bin/bash
# Run every row of a grid on this machine with N parallel workers, skipping finished conditions.
# Usage: bash scripts/run_local.sh grids/esol.csv 2
GRID=$1; WORKERS=${2:-2}
N=$(($(wc -l < "$GRID") - 1))
export OMP_NUM_THREADS=1
seq 0 $((N - 1)) | xargs -P "$WORKERS" -I{} sh -c \
  "python3 scripts/run_benchmark.py --grid $GRID --task {} --out results/v2 --threads 1 --skip-existing >> results/v2_logs/task_{}.log 2>&1 || echo FAILED task {}"
