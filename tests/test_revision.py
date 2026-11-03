"""Tests for the revised benchmark: models, uncertainty metrics, noise, splits, runner."""

import numpy as np
import pytest

from solnoise.features import descriptor_matrix, fingerprint_matrix
from solnoise.models import ESOLLinear, RandomForest, TanimotoGP, TrainMean, tanimoto
from solnoise.noise import add_outlier_noise, corrupt
from solnoise.splits import butina_clusters, cluster_split
from solnoise.uq_metrics import (conformal_quantile, coverage, error_based_calibration, gaussian_nll,
                                 miscalibration_area, simulated_rank_correlation, variance_scale)

SMILES = ["CCO", "c1ccccc1O", "CC(=O)Oc1ccccc1C(=O)O", "CCCCCC", "c1ccc2ccccc2c1", "CC(C)O", "OCC(O)CO",
          "Clc1ccccc1", "CCN(CC)CC", "O=C(O)c1ccccc1", "CCCCO", "CCCCCCCC", "c1ccncc1", "CC(C)C",
          "OC1CCCCC1", "CCOC(C)=O", "Brc1ccccc1", "CCCCCCO", "CC(N)=O", "c1ccsc1"]


def calibrated_sample(n=20000, seed=0):
    rng = np.random.default_rng(seed)
    sd = rng.uniform(0.2, 2.0, n)
    mu = rng.normal(0, 1, n)
    return mu + rng.normal(0, sd), mu, sd


# ------------------------------------------------------------------ metrics

def test_calibrated_gaussian_errors_give_nominal_coverage_and_small_miscalibration():
    y, mu, sd = calibrated_sample()
    assert coverage(y, mu, sd, 1.0) == pytest.approx(0.683, abs=0.01)
    assert coverage(y, mu, sd, 1.96) == pytest.approx(0.95, abs=0.01)
    assert miscalibration_area(y, mu, sd) < 0.01
    assert variance_scale(y, mu, sd) == pytest.approx(1.0, abs=0.02)


def test_error_based_calibration_has_unit_slope_for_calibrated_errors():
    y, mu, sd = calibrated_sample()
    ebc = error_based_calibration(y, mu, sd, n_bins=10)
    assert ebc["slope"] == pytest.approx(1.0, abs=0.05)
    assert abs(ebc["intercept"]) < 0.05
    assert ebc["ence"] < 0.05


def test_variance_scale_recovers_a_known_miscalibration():
    y, mu, sd = calibrated_sample()
    assert variance_scale(y, mu, sd / 2) == pytest.approx(2.0, abs=0.05)


def test_conformal_quantile_gives_target_coverage_on_exchangeable_data():
    y, mu, sd = calibrated_sample(40000)
    q = conformal_quantile(y[:20000], mu[:20000], sd[:20000], 0.9)
    assert coverage(y[20000:], mu[20000:], sd[20000:], q) == pytest.approx(0.9, abs=0.01)


def test_nll_is_lower_for_the_true_sd_than_for_a_wrong_one():
    y, mu, sd = calibrated_sample()
    assert gaussian_nll(y, mu, sd) < gaussian_nll(y, mu, 3 * sd)
    assert gaussian_nll(y, mu, sd) < gaussian_nll(y, mu, sd / 3)


def test_simulated_rank_correlation_is_positive_and_nan_for_constant_sd():
    _, _, sd = calibrated_sample(500)
    assert 0.1 < simulated_rank_correlation(sd, n_draws=50) < 0.6
    assert np.isnan(simulated_rank_correlation(np.ones(10)))


# -------------------------------------------------------------------- noise

def test_outlier_noise_hits_about_the_requested_fraction_with_large_shifts():
    y = np.zeros(20000)
    noisy = add_outlier_noise(y, 0.1, seed=3)
    hit = noisy != 0
    assert hit.mean() == pytest.approx(0.1, abs=0.01)
    assert np.all((np.abs(noisy[hit]) >= 1.0) & (np.abs(noisy[hit]) <= 3.0))


def test_corrupt_is_reproducible_and_leaves_input_unchanged():
    y = np.linspace(-5, 0, 50)
    original = y.copy()
    assert np.array_equal(corrupt(y, "gaussian", 0.5, 7), corrupt(y, "gaussian", 0.5, 7))
    assert np.array_equal(y, original)
    with pytest.raises(ValueError):
        corrupt(y, "uniform", 0.5, 7)


# ------------------------------------------------------------------- splits

def test_cluster_split_is_disjoint_complete_and_keeps_clusters_together():
    fps = fingerprint_matrix(SMILES * 3)
    y = np.linspace(-4, 1, len(fps))
    labels = butina_clusters(fps, threshold=0.9)
    train, valid, test = cluster_split(fps, y, seed=0, labels=labels, n_candidates=20)
    allocated = np.concatenate([train, valid, test])
    assert sorted(allocated) == list(range(len(fps)))
    for part_a, part_b in [(train, test), (train, valid), (valid, test)]:
        assert not set(labels[part_a]) & set(labels[part_b])


def test_tanimoto_of_identical_fingerprints_is_one():
    fps = fingerprint_matrix(SMILES)
    sim = tanimoto(fps, fps)
    assert np.allclose(np.diag(sim), 1.0)
    assert np.all((sim >= 0) & (sim <= 1))


# ------------------------------------------------------------------- models

@pytest.mark.parametrize("model_cls,features", [
    (TrainMean, "descriptors"), (ESOLLinear, "descriptors"), (TanimotoGP, "fingerprints"),
    (RandomForest, "descriptors"),
])
def test_models_return_means_and_nonnegative_variances(model_cls, features):
    x = descriptor_matrix(SMILES) if features == "descriptors" else fingerprint_matrix(SMILES)
    y = np.linspace(-4, 1, len(SMILES))
    model = model_cls(0).fit(x[:15], y[:15], x[15:], y[15:])
    mu, var_epi, var_ale = model.predict(x[15:])
    assert mu.shape == var_epi.shape == var_ale.shape == (5,)
    assert np.all(var_epi >= 0) and np.all(var_ale >= 0)


def test_tanimoto_gp_interpolates_training_points_when_noise_is_small():
    x = fingerprint_matrix(SMILES)
    y = -0.3 * x.sum(1)  # a label that depends on structure
    model = TanimotoGP(0).fit(x, y)
    mu, var_epi, _ = model.predict(x)
    assert np.sqrt(np.mean((mu - y) ** 2)) < 0.5 * np.std(y)
    assert np.all(var_epi < model.amp * model.y_std ** 2)
