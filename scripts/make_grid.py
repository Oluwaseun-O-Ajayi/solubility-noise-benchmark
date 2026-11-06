"""Write the grid of conditions for the revised benchmark, one row per Slurm array task."""

import itertools
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "grids"


def grid(dataset, splits, gaussian_levels, outlier_rates, seeds, systematic_levels=()):
    rows = []
    for seed, split in itertools.product(seeds, splits):
        rows += [dict(dataset=dataset, split=split, noise="gaussian", level=lv, seed=seed) for lv in gaussian_levels]
        rows += [dict(dataset=dataset, split=split, noise="outlier", level=r, seed=seed) for r in outlier_rates]
        rows += [dict(dataset=dataset, split=split, noise="systematic", level=lv, seed=seed)
                 for lv in systematic_levels]
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(exist_ok=True)
    esol = grid("esol", ["random", "scaffold", "cluster"], [0.0, 0.3, 0.6, 1.0, 2.0], [0.05, 0.1, 0.2], range(10))
    aqsol = grid("aqsoldb", ["random", "scaffold", "cluster"], [0.0, 0.6, 1.0], [0.1], range(5))
    esol.to_csv(OUT / "esol.csv", index=False)
    aqsol.to_csv(OUT / "aqsoldb.csv", index=False)

    # Sensitivity control: the cluster split without label matching, both data sets.
    unmatched = pd.concat([
        grid("esol", ["cluster_unmatched"], [0.0, 0.3, 0.6, 1.0, 2.0], [0.05, 0.1, 0.2], range(10)),
        grid("aqsoldb", ["cluster_unmatched"], [0.0, 0.6, 1.0], [0.1], range(5)),
    ], ignore_index=True)
    unmatched.to_csv(OUT / "cluster_unmatched.csv", index=False)

    # The two gradient-boosting models across every condition of the main grid.
    gbt = pd.concat([esol, aqsol], ignore_index=True)
    gbt.to_csv(OUT / "gbt.csv", index=False)

    # Second property: lipophilicity (logD 7.4), which has no published descriptor equation.
    lipo = grid("lipo", ["random", "scaffold", "cluster"], [0.0, 0.3, 0.6, 1.0], [0.1], range(5))
    lipo.to_csv(OUT / "lipo.csv", index=False)

    # Structured noise: one offset per Butina cluster instead of one per molecule.
    systematic = pd.concat([
        grid("esol", ["random", "scaffold", "cluster"], [], [], range(10), systematic_levels=[0.3, 0.6, 1.0]),
        grid("aqsoldb", ["random", "scaffold", "cluster"], [], [], range(5), systematic_levels=[0.6, 1.0]),
    ], ignore_index=True)
    systematic.to_csv(OUT / "systematic.csv", index=False)

    print(f"esol: {len(esol)} tasks, aqsoldb: {len(aqsol)} tasks, "
          f"cluster_unmatched: {len(unmatched)} tasks, gbt: {len(gbt)} tasks, "
          f"lipo: {len(lipo)} tasks, systematic: {len(systematic)} tasks")


if __name__ == "__main__":
    main()
