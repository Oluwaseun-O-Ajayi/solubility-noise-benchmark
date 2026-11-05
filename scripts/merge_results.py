"""Merge a partial results folder into the main one, row by row.

The gradient-boosting array writes to results/v2_gbt because it fits only two of
the models; this script appends those rows to the matching files in results/v2
without touching the rows already there.

    python scripts/merge_results.py results/v2_gbt results/v2
"""

import sys
from pathlib import Path

import pandas as pd


def merge(src: Path, dst: Path):
    merged = skipped = 0
    for kind, suffix, compression in [("metrics", ".csv", None), ("predictions", ".csv.gz", "gzip")]:
        (dst / kind).mkdir(parents=True, exist_ok=True)
        for path in sorted((src / kind).glob(f"*{suffix}")):
            target = dst / kind / path.name
            new = pd.read_csv(path)
            if target.exists():
                old = pd.read_csv(target)
                keep = old[~old.model.isin(set(new.model))]
                out = pd.concat([keep, new], ignore_index=True)
            else:
                out = new
                skipped += 1
            out.to_csv(target, index=False, compression=compression)
            merged += 1
    print(f"merged {merged} files into {dst} ({skipped} had no existing counterpart)")


if __name__ == "__main__":
    src = Path(sys.argv[1] if len(sys.argv) > 1 else "results/v2_gbt")
    dst = Path(sys.argv[2] if len(sys.argv) > 2 else "results/v2")
    merge(src, dst)
