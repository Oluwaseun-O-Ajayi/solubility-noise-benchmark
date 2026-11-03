"""Simulate measurement error by adding Gaussian noise to labels."""

import numpy as np


def add_label_noise(y, sigma, seed):
    """Return a copy of ``y`` with zero-mean Gaussian noise of standard deviation ``sigma``.

    ``sigma`` is in the units of ``y`` (log10 mol/L for ESOL). ``sigma = 0`` returns
    an unchanged copy.
    """
    y = np.asarray(y, dtype=np.float64)
    if sigma == 0:
        return y.copy()
    return y + np.random.default_rng(seed).normal(0.0, sigma, size=y.shape)


def add_outlier_noise(y, rate, seed, low=1.0, high=3.0):
    """Corrupt a fraction ``rate`` of labels with a large error, as in unit or transcription mistakes.

    Each label is independently chosen with probability ``rate``; a chosen label is
    shifted by a magnitude drawn uniformly from [low, high] log units with a random
    sign (a factor of 10 to 1000 in solubility). ``rate = 0`` returns an unchanged copy.
    """
    y = np.asarray(y, dtype=np.float64)
    if rate == 0:
        return y.copy()
    rng = np.random.default_rng(seed)
    hit = rng.random(y.shape) < rate
    shift = rng.uniform(low, high, size=y.shape) * rng.choice([-1.0, 1.0], size=y.shape)
    return y + hit * shift


def corrupt(y, noise_type, level, seed):
    """Dispatch to the noise model named ``noise_type`` ("gaussian" or "outlier")."""
    if noise_type == "gaussian":
        return add_label_noise(y, level, seed)
    if noise_type == "outlier":
        return add_outlier_noise(y, level, seed)
    raise ValueError(f"unknown noise type: {noise_type}")
