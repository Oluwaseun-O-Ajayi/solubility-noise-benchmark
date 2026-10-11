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
