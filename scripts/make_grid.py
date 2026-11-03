"""Write the grid of conditions for the revised benchmark, one row per Slurm array task."""

import itertools
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "grids"


def grid(dataset, splits, gaussian_levels, outlier_rates, seeds):
    rows = []
    for seed, split in itertools.product(seeds, splits):
        rows += [dict(dataset=dataset, split=split, noise="gaussian", level=lv, seed=seed) for lv in gaussian_levels]
        rows += [dict(dataset=dataset, split=split, noise="outlier", level=r, seed=seed) for r in outlier_rates]
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(exist_ok=True)
    esol = grid("esol", ["random", "scaffold", "cluster"], [0.0, 0.3, 0.6, 1.0, 2.0], [0.05, 0.1, 0.2], range(10))
    aqsol = grid("aqsoldb", ["random", "scaffold", "cluster"], [0.0, 0.6, 1.0], [0.1], range(5))
    esol.to_csv(OUT / "esol.csv", index=False)
    aqsol.to_csv(OUT / "aqsoldb.csv", index=False)
    print(f"esol: {len(esol)} tasks, aqsoldb: {len(aqsol)} tasks")


if __name__ == "__main__":
    main()
