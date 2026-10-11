"""Why does the Gaussian process improve on the scaffold split when labels get noisier?

Fit the Gaussian process on clean and on noisy labels and compare what the
kernel learned: the noise term and the length scales.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solnoise.baselines import GaussianProcessModel  # noqa: E402
from solnoise.data import load_esol  # noqa: E402
from solnoise.experiment import Featurized, make_split  # noqa: E402
from solnoise.metrics import rmse  # noqa: E402
from solnoise.noise import add_label_noise  # noqa: E402


def main():
    frame = load_esol()
    data = Featurized(frame["smiles"], frame["logS"])
    rows = []
    for seed in (0, 1, 2):
        train, valid, test = make_split(data, "scaffold", seed)
        fit_idx = np.concatenate([train, valid])
        for sigma in (0.0, 1.0):
            y_noisy = add_label_noise(data.y, sigma, seed=10_000 + seed)
            model = GaussianProcessModel(seed).fit(data.descriptors[fit_idx], y_noisy[fit_idx])
            mean, _ = model.predict(data.descriptors[test])
            length_scales = model.gp.kernel_.k1.k2.length_scale
            rows.append({
                "seed": seed,
                "sigma": sigma,
                "test_rmse": round(rmse(data.y[test], mean), 2),
                "mean_error": round(float((mean - data.y[test]).mean()), 2),
                "learned_noise_term": round(float(model.gp.kernel_.k2.noise_level), 3),
                "median_length_scale": round(float(np.median(length_scales)), 1),
                "train_label_mean": round(float(data.y[fit_idx].mean()), 2),
                "test_label_mean": round(float(data.y[test].mean()), 2),
            })
    table = pd.DataFrame(rows)
    table.to_csv(Path(__file__).resolve().parent.parent / "results" / "gp_diagnostic.csv", index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
