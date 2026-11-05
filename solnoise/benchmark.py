"""Revised benchmark: one condition = data set, split, noise type and level, seed.

Every model is trained on the training set with noisy labels, uses the noisy
validation set for early stopping (networks) and for recalibration (all
models), and is scored on the test set twice: against the unaltered labels
(does the model bracket the reported value?) and against an independent noisy
copy of them (does it bracket a new measurement of the same quality?).
Per-molecule predictions are saved so every figure can be recomputed.
"""

import json
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .data import load_aqsoldb, load_esol
from .features import descriptor_matrix, fingerprint_matrix, smiles_to_graphs
from .models import REGISTRY, tanimoto
from .noise import corrupt
from .splits import butina_clusters, cluster_split, random_split, scaffold_split
from .uq_metrics import score_all

ROOT = Path(__file__).resolve().parent.parent
TRAIN_NOISE_OFFSET, TEST_NOISE_OFFSET = 10_000, 20_000


class Dataset:
    """Labels plus all three representations, computed once per process."""

    def __init__(self, name):
        frame = load_esol() if name == "esol" else load_aqsoldb()
        self.name = name
        self.smiles = list(frame["smiles"])
        self.y = frame["logS"].to_numpy(dtype=np.float64)
        self.sd = frame["sd"].to_numpy() if "sd" in frame else np.zeros(len(self.y))
        self.group = frame["group"].to_numpy() if "group" in frame else np.array(["NA"] * len(self.y))
        self.descriptors = descriptor_matrix(self.smiles)
        self.fingerprints = fingerprint_matrix(self.smiles)
        self._graphs = None
        self._clusters = None

    @property
    def graphs(self):
        if self._graphs is None:
            graphs = smiles_to_graphs(self.smiles)
            self._graphs = np.empty(len(graphs), dtype=object)
            self._graphs[:] = graphs
        return self._graphs

    @property
    def clusters(self):
        if self._clusters is None:
            self._clusters = butina_clusters(self.fingerprints)
        return self._clusters

    def representation(self, kind):
        return {"descriptors": self.descriptors, "fingerprints": self.fingerprints, "graphs": self.graphs}[kind]


@lru_cache(maxsize=2)
def get_dataset(name):
    return Dataset(name)


def make_split(data, split, seed):
    if split == "random":
        return random_split(len(data.y), seed)
    if split == "scaffold":
        return scaffold_split(data.smiles, seed)
    if split == "cluster":
        return cluster_split(data.fingerprints, data.y, seed, labels=data.clusters)
    if split == "cluster_unmatched":
        # Sensitivity control for the cluster split: take the first random cluster
        # ordering instead of the one whose test mean label is closest to the
        # training mean, so no test label is read when forming the split.
        return cluster_split(data.fingerprints, data.y, seed, labels=data.clusters, n_candidates=1)
    raise ValueError(f"unknown split: {split}")


def load_config(path=None):
    path = Path(path) if path else ROOT / "configs" / "models.json"
    return json.loads(path.read_text())


def model_kwargs(config, model, dataset):
    kwargs = dict(config.get("models", {}).get(model, {}))
    kwargs.update(config.get("dataset_overrides", {}).get(dataset, {}).get(model, {}))
    return kwargs


def condition_tag(dataset, split, noise_type, level, seed):
    return f"{dataset}_{split}_{noise_type}{level:g}_seed{seed}"


def run_condition(dataset, split, noise_type, level, seed, models=None, config=None, out_dir=None,
                  log=print):
    data = get_dataset(dataset)
    config = config if config is not None else load_config()
    models = models or config.get("default_models", list(REGISTRY))
    train, valid, test = make_split(data, split, seed)
    y_noisy = corrupt(data.y, noise_type, level, TRAIN_NOISE_OFFSET + seed)
    y_test_noisy = corrupt(data.y, noise_type, level, TEST_NOISE_OFFSET + seed)[test]
    y_test = data.y[test]
    max_sim = tanimoto(data.fingerprints[test], data.fingerprints[train]).max(1)

    base = {"dataset": dataset, "split": split, "noise_type": noise_type, "level": level, "seed": seed,
            "n_train": len(train), "n_valid": len(valid), "n_test": len(test),
            "train_label_mean": float(data.y[train].mean()), "test_label_mean": float(y_test.mean()),
            "test_max_sim_median": float(np.median(max_sim))}
    rows, preds = [], []
    for name in models:
        kind, build = REGISTRY[name]
        x = data.representation(kind)
        started = time.time()
        model = build(seed, model_kwargs(config, name, dataset))
        model.fit(x[train], y_noisy[train], x[valid], y_noisy[valid])
        mu, var_epi, var_ale = model.predict(x[test])
        mu_v, var_epi_v, var_ale_v = model.predict(x[valid])
        seconds = round(time.time() - started, 1)

        row = dict(base, model=name, seconds=seconds)
        row.update(score_all(y_test, mu, var_epi, var_ale, y_noisy[valid], mu_v, var_epi_v, var_ale_v))
        row.update(score_all(y_test_noisy, mu, var_epi, var_ale, y_noisy[valid], mu_v, var_epi_v, var_ale_v,
                             prefix="noisytest_"))
        if hasattr(model, "noise_level"):
            row["fitted_noise_var"] = float(model.noise_level * model.y_std ** 2)
            row["fitted_noise_var_std_units"] = float(model.noise_level)
            row["train_label_sd"] = float(model.y_std)
        if hasattr(model, "length_scales"):
            row["median_length_scale"] = float(np.median(model.length_scales))
        rows.append(row)
        preds.append(pd.DataFrame({
            "model": name, "index": test, "y": y_test, "y_noisy_test": y_test_noisy, "mean": mu,
            "var_epi": var_epi, "var_ale": var_ale, "max_sim": max_sim,
            "sd_replicates": data.sd[test], "group": data.group[test]}))
        log(f"  {name:12s} rmse={row['rmse']:.3f} cov68={row['cov68_total']:.2f} "
            f"cov68_epi={row['cov68_epi']:.2f} rho={row['rho_total']:.2f} ({seconds}s)")

    metrics = pd.DataFrame(rows)
    predictions = pd.concat(preds, ignore_index=True)
    for key in ("dataset", "split", "noise_type", "level", "seed"):
        predictions[key] = base[key]
    if out_dir is not None:
        out_dir = Path(out_dir)
        tag = condition_tag(dataset, split, noise_type, level, seed)
        (out_dir / "metrics").mkdir(parents=True, exist_ok=True)
        (out_dir / "predictions").mkdir(parents=True, exist_ok=True)
        metrics.to_csv(out_dir / "metrics" / f"{tag}.csv", index=False)
        predictions.to_csv(out_dir / "predictions" / f"{tag}.csv.gz", index=False)
    return metrics, predictions
