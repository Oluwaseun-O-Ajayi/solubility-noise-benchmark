"""Accuracy and uncertainty metrics."""

import numpy as np
from scipy.stats import spearmanr


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def coverage(y_true, y_pred, y_std, z):
    """Fraction of true values inside ``prediction +/- z * std``.

    For well-calibrated Gaussian uncertainty this is about 0.68 at z = 1 and
    about 0.95 at z = 1.96. Lower means overconfident; higher means underconfident.
    """
    error = np.abs(np.asarray(y_true) - np.asarray(y_pred))
    return float(np.mean(error <= z * np.asarray(y_std)))


def error_rank_correlation(y_true, y_pred, y_std):
    """Spearman correlation between predicted uncertainty and actual absolute error.

    Positive values mean the model is less sure where it is more wrong.
    """
    error = np.abs(np.asarray(y_true) - np.asarray(y_pred))
    if np.allclose(y_std, np.asarray(y_std)[0]):
        return 0.0
    return float(spearmanr(y_std, error).statistic)


def all_metrics(y_true, y_pred, y_std):
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "coverage_68": coverage(y_true, y_pred, y_std, 1.0),
        "coverage_95": coverage(y_true, y_pred, y_std, 1.96),
        "error_rank_corr": error_rank_correlation(y_true, y_pred, y_std),
    }
