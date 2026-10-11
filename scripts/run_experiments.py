"""Run the full grid of experiments and write one CSV row per model per condition.

Example:
    python scripts/run_experiments.py --seeds 0 1 2
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solnoise.data import load_esol  # noqa: E402
from solnoise.experiment import Featurized, run_condition  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--splits", nargs="+", default=["random", "scaffold"])
    parser.add_argument("--sigmas", type=float, nargs="+", default=[0.0, 0.3, 0.6, 1.0],
                        help="label noise levels, in log10 mol/L")
    parser.add_argument("--members", type=int, default=5, help="networks per deep ensemble")
    parser.add_argument("--threads", type=int, default=0, help="PyTorch CPU threads (0 = default)")
    parser.add_argument("--out", default="results/results.csv")
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    frame = load_esol()
    data = Featurized(frame["smiles"], frame["logS"])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for seed in args.seeds:
        for split in args.splits:
            for sigma in args.sigmas:
                print(f"seed={seed} split={split} sigma={sigma}", flush=True)
                rows += run_condition(data, split, sigma, seed, n_members=args.members,
                                      log=lambda line: print(line, flush=True))
                pd.DataFrame(rows).to_csv(out, index=False)  # save progress after each condition
    print(f"wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
