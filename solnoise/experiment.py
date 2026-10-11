"""Run one experimental condition: a split, a noise level, a seed, all four models."""

import time

import numpy as np

from .baselines import GaussianProcessModel, RandomForestModel
from .features import descriptor_matrix, fingerprint_matrix, smiles_to_graphs
from .metrics import all_metrics
from .noise import add_label_noise
from .splits import random_split, scaffold_split
from .training import DeepEnsemble

MODEL_NAMES = ["train_mean", "gaussian_process", "random_forest", "fingerprint_net", "graph_net"]


class Featurized:
    """All three representations of the dataset, computed once."""

    def __init__(self, smiles, y):
        self.smiles = list(smiles)
        self.y = np.asarray(y, dtype=np.float64)
        self.descriptors = descriptor_matrix(self.smiles)
        self.fingerprints = fingerprint_matrix(self.smiles)
        graphs = smiles_to_graphs(self.smiles)
        self.graphs = np.empty(len(graphs), dtype=object)
        self.graphs[:] = graphs


def make_split(data, split, seed):
    if split == "random":
        return random_split(len(data.y), seed)
    if split == "scaffold":
        return scaffold_split(data.smiles, seed)
    raise ValueError(f"unknown split: {split}")


def run_condition(data, split, sigma, seed, n_members=5, models=MODEL_NAMES, log=print):
    """Train every model on noisy labels and score it against clean test labels.

    Noise is added to the training and validation labels, because in a real
    project the validation measurements are as noisy as the training ones. The
    test labels are left untouched so the score measures distance from the truth.
    """
    train, valid, test = make_split(data, split, seed)
    y_noisy = add_label_noise(data.y, sigma, seed=10_000 + seed)
    fit_idx = np.concatenate([train, valid])  # the baselines need no validation set
    rows = []

    def record(name, mean, std, started):
        row = {"split": split, "sigma": sigma, "seed": seed, "model": name,
               "n_train": len(train), "n_test": len(test), "seconds": round(time.time() - started, 1)}
        row.update(all_metrics(data.y[test], mean, std))
        rows.append(row)
        log(f"  {name:17s} rmse={row['rmse']:.3f}  cov68={row['coverage_68']:.2f}  ({row['seconds']}s)")

    if "train_mean" in models:
        # The floor every model has to beat: always predict the average training label.
        started = time.time()
        mean = np.full(len(test), y_noisy[fit_idx].mean())
        std = np.full(len(test), y_noisy[fit_idx].std())
        record("train_mean", mean, std, started)
    if "gaussian_process" in models:
        started = time.time()
        model = GaussianProcessModel(seed).fit(data.descriptors[fit_idx], y_noisy[fit_idx])
        record("gaussian_process", *model.predict(data.descriptors[test]), started)
    if "random_forest" in models:
        started = time.time()
        model = RandomForestModel(seed).fit(data.fingerprints[fit_idx], y_noisy[fit_idx])
        record("random_forest", *model.predict(data.fingerprints[test]), started)
    for name, kind, x in [("fingerprint_net", "fingerprint", data.fingerprints),
                          ("graph_net", "graph", data.graphs)]:
        if name not in models:
            continue
        started = time.time()
        model = DeepEnsemble(kind, seed=seed, n_members=n_members)
        model.fit(x[train], y_noisy[train], x[valid], y_noisy[valid])
        record(name, *model.predict(x[test]), started)
    return rows
