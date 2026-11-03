"""Merge per-condition metric files into results/v2/metrics_all.csv and report what is missing."""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from solnoise.benchmark import condition_tag  # noqa: E402


def main(out="results/v2"):
    out = ROOT / out
    files = sorted((out / "metrics").glob("*.csv"))
    frame = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    frame.to_csv(out / "metrics_all.csv", index=False)
    print(f"{len(files)} conditions, {len(frame)} rows -> {out / 'metrics_all.csv'}")
    for grid in sorted((ROOT / "grids").glob("*.csv")):
        g = pd.read_csv(grid)
        tags = {condition_tag(r.dataset, r.split, r.noise, float(r.level), int(r.seed)) for r in g.itertuples()}
        done = {f.stem for f in files}
        missing = sorted(tags - done)
        print(f"{grid.name}: {len(tags) - len(missing)}/{len(tags)} done" + (f"; missing e.g. {missing[:3]}" if missing else ""))


if __name__ == "__main__":
    main(*sys.argv[1:])
