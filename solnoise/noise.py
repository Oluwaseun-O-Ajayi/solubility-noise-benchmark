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


def add_systematic_noise(y, sigma, seed, groups):
    """Add noise that is shared within a group of structurally related molecules.

    One offset is drawn from N(0, sigma^2) per group and added to every member of
    that group, so each label still has marginal variance sigma^2 but the errors
    are perfectly correlated within a group and independent between groups. This
    stands in for a laboratory or assay bias that moves a whole family of related
    compounds together, which independent per-molecule noise cannot represent.
    """
    y = np.asarray(y, dtype=float)
    groups = np.asarray(groups)
    if sigma == 0:
        return y.copy()
    rng = np.random.default_rng(seed)
    keys, inverse = np.unique(groups, return_inverse=True)
    offsets = rng.normal(0.0, sigma, size=len(keys))
    return y + offsets[inverse]


def corrupt(y, noise_type, level, seed, groups=None):
    """Dispatch to the named noise model ("gaussian", "outlier" or "systematic")."""
    if noise_type == "gaussian":
        return add_label_noise(y, level, seed)
    if noise_type == "outlier":
        return add_outlier_noise(y, level, seed)
    if noise_type == "systematic":
        if groups is None:
            raise ValueError("systematic noise needs a group label per molecule")
        return add_systematic_noise(y, level, seed, groups)
    raise ValueError(f"unknown noise type: {noise_type}")
