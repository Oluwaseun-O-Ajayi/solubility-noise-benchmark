"""Run conditions of the revised benchmark.

Either give one condition directly:
    python scripts/run_benchmark.py --dataset esol --split random --noise gaussian --level 1.0 --seed 0
or run row ``--task`` of a grid file made by scripts/make_grid.py (for Slurm array jobs):
    python scripts/run_benchmark.py --grid grids/esol.csv --task $SLURM_ARRAY_TASK_ID
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solnoise.benchmark import condition_tag, load_config, run_condition  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=["esol", "aqsoldb", "lipo"])
    p.add_argument("--split", choices=["random", "scaffold", "cluster", "cluster_unmatched"])
    p.add_argument("--noise", choices=["gaussian", "outlier", "systematic"], default="gaussian")
    p.add_argument("--level", type=float)
    p.add_argument("--seed", type=int)
    p.add_argument("--grid", help="CSV with columns dataset, split, noise, level, seed")
    p.add_argument("--task", type=int, help="row of the grid to run (0-based)")
    p.add_argument("--models", nargs="+")
    p.add_argument("--config")
    p.add_argument("--threads", type=int, default=1, help="PyTorch CPU threads (results use 1)")
    p.add_argument("--out", default="results/v2")
    p.add_argument("--skip-existing", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(args.threads)

    if args.grid is not None:
        row = pd.read_csv(args.grid).iloc[args.task]
        dataset, split, noise, level, seed = row.dataset, row.split, row.noise, float(row.level), int(row.seed)
    else:
        dataset, split, noise, level, seed = args.dataset, args.split, args.noise, args.level, args.seed
    tag = condition_tag(dataset, split, noise, level, seed)
    if args.skip_existing and (Path(args.out) / "metrics" / f"{tag}.csv").exists():
        print(f"{tag} already done")
        return
    print(f"{tag}  torch threads={torch.get_num_threads()}", flush=True)
    run_condition(dataset, split, noise, level, seed, models=args.models, config=load_config(args.config),
                  out_dir=args.out, log=lambda line: print(line, flush=True))


if __name__ == "__main__":
    main()
