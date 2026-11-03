"""Uncertainty metrics and recalibration for the revised benchmark.

All functions take the true values ``y``, predicted means ``mu`` and predicted
standard deviations ``sd`` as 1-D arrays.
"""

import numpy as np
from scipy.stats import norm, spearmanr


def coverage(y, mu, sd, z):
    """Fraction of ``y`` inside ``mu +/- z * sd``."""
    return float(np.mean(np.abs(y - mu) <= z * sd))


def gaussian_nll(y, mu, sd):
    """Mean Gaussian negative log-likelihood per molecule (natural log, includes 0.5 log 2 pi)."""
    var = np.maximum(sd, 1e-12) ** 2
    return float(np.mean(0.5 * np.log(2 * np.pi * var) + 0.5 * (y - mu) ** 2 / var))


def miscalibration_area(y, mu, sd, n_levels=99):
    """Mean absolute gap between nominal and observed coverage of central intervals.

    Levels run from 0.01 to 0.99. Zero means the observed coverage matches every
    nominal level; positive values do not say in which direction it fails.
    """
    levels = np.linspace(0.01, 0.99, n_levels)
    z = norm.ppf(0.5 + levels / 2)
    err = np.abs(y - mu) / np.maximum(sd, 1e-12)
    observed = (err[None, :] <= z[:, None]).mean(1)
    return float(np.mean(np.abs(observed - levels)))


def rank_correlation(y, mu, sd):
    """Spearman correlation between ``sd`` and absolute error; NaN when ``sd`` is constant."""
    if np.allclose(sd, sd[0]):
        return float("nan")
    return float(spearmanr(sd, np.abs(y - mu)).statistic)


def simulated_rank_correlation(sd, n_draws=200, seed=0):
    """Expected Spearman correlation if errors were drawn exactly from N(0, sd^2).

    This is the reference value of Rasmussen et al. (2023): the rank correlation a
    perfectly calibrated model with these ``sd`` values would reach on average.
    """
    if np.allclose(sd, sd[0]):
        return float("nan")
    rng = np.random.default_rng(seed)
    draws = [spearmanr(sd, np.abs(rng.normal(0.0, sd))).statistic for _ in range(n_draws)]
    return float(np.mean(draws))


def error_based_calibration(y, mu, sd, n_bins=None):
    """Binned RMSE against root mean variance (RMV), with a linear fit.

    Molecules are sorted by ``sd`` and split into equal-count bins. A calibrated
    model gives points on the line RMSE = RMV (slope 1, intercept 0).
    Returns slope, intercept, R^2, ENCE and the bin values.
    """
    n = len(y)
    if n_bins is None:
        n_bins = int(np.clip(n // 15, 4, 10))
    order = np.argsort(sd)
    bins = np.array_split(order, n_bins)
    rmv = np.array([np.sqrt(np.mean(sd[b] ** 2)) for b in bins])
    rmse = np.array([np.sqrt(np.mean((y[b] - mu[b]) ** 2)) for b in bins])
    if np.ptp(rmv) < 1e-12:
        slope = intercept = r2 = float("nan")
    else:
        slope, intercept = np.polyfit(rmv, rmse, 1)
        fitted = slope * rmv + intercept
        ss_res, ss_tot = np.sum((rmse - fitted) ** 2), np.sum((rmse - rmse.mean()) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    ence = float(np.mean(np.abs(rmv - rmse) / np.maximum(rmv, 1e-12)))
    return {"slope": float(slope), "intercept": float(intercept), "r2": r2, "ence": ence,
            "rmv_bins": rmv.tolist(), "rmse_bins": rmse.tolist()}


def variance_scale(y_cal, mu_cal, sd_cal):
    """Single factor s that makes standardized errors on a calibration set have unit variance.

    Returns s such that the recalibrated standard deviation is s * sd. This is the
    maximum-likelihood scale for Gaussian errors.
    """
    z = (y_cal - mu_cal) / np.maximum(sd_cal, 1e-12)
    return float(np.sqrt(np.mean(z ** 2)))


def conformal_quantile(y_cal, mu_cal, sd_cal, level):
    """Normalized split-conformal quantile of |y - mu| / sd for a target coverage ``level``."""
    scores = np.sort(np.abs(y_cal - mu_cal) / np.maximum(sd_cal, 1e-12))
    n = len(scores)
    k = int(np.ceil((n + 1) * level))
    return float(scores[min(k, n) - 1])


def score_all(y, mu, var_epi, var_ale, y_cal=None, mu_cal=None, var_epi_cal=None, var_ale_cal=None,
              prefix=""):
    """Every accuracy and uncertainty metric for one model on one test set."""
    sd_tot = np.sqrt(var_epi + var_ale)
    sd_epi = np.sqrt(var_epi)
    out = {
        "rmse": float(np.sqrt(np.mean((y - mu) ** 2))),
        "mae": float(np.mean(np.abs(y - mu))),
        "cov68_total": coverage(y, mu, sd_tot, 1.0),
        "cov95_total": coverage(y, mu, sd_tot, 1.96),
        "cov68_epi": coverage(y, mu, sd_epi, 1.0),
        "cov95_epi": coverage(y, mu, sd_epi, 1.96),
        "nll_total": gaussian_nll(y, mu, sd_tot),
        "miscal_area_total": miscalibration_area(y, mu, sd_tot),
        "miscal_area_epi": miscalibration_area(y, mu, sd_epi),
        "rho_total": rank_correlation(y, mu, sd_tot),
        "rho_total_sim": simulated_rank_correlation(sd_tot),
        "rho_epi": rank_correlation(y, mu, sd_epi),
        "mean_sd_total": float(np.mean(sd_tot)),
        "mean_sd_epi": float(np.mean(sd_epi)),
        "mean_sd_ale": float(np.mean(np.sqrt(var_ale))),
    }
    ebc = error_based_calibration(y, mu, sd_tot)
    out.update({"ebc_slope": ebc["slope"], "ebc_intercept": ebc["intercept"], "ebc_r2": ebc["r2"],
                "ence": ebc["ence"]})
    if y_cal is not None:
        sd_cal = np.sqrt(var_epi_cal + var_ale_cal)
        s = variance_scale(y_cal, mu_cal, sd_cal)
        q68 = conformal_quantile(y_cal, mu_cal, sd_cal, 0.68)
        q95 = conformal_quantile(y_cal, mu_cal, sd_cal, 0.95)
        out.update({
            "recal_scale": s,
            "cov68_recal": coverage(y, mu, s * sd_tot, 1.0),
            "cov95_recal": coverage(y, mu, s * sd_tot, 1.96),
            "nll_recal": gaussian_nll(y, mu, s * sd_tot),
            "conformal_q68": q68,
            "conformal_q95": q95,
            "cov68_conformal": coverage(y, mu, sd_tot, q68),
            "cov95_conformal": coverage(y, mu, sd_tot, q95),
        })
    return {prefix + k: v for k, v in out.items()}
