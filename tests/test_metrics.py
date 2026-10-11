import numpy as np

from solnoise.metrics import coverage, error_rank_correlation, rmse


def test_rmse_of_known_errors():
    assert rmse([0, 0, 0, 0], [1, -1, 1, -1]) == 1.0


def test_coverage_matches_gaussian_theory():
    rng = np.random.default_rng(0)
    truth = rng.normal(size=50_000)          # errors are exactly N(0, 1)
    prediction, std = np.zeros_like(truth), np.ones_like(truth)
    assert abs(coverage(truth, prediction, std, 1.0) - 0.683) < 0.01
    assert abs(coverage(truth, prediction, std, 1.96) - 0.950) < 0.01


def test_overconfident_model_has_low_coverage():
    rng = np.random.default_rng(1)
    truth = rng.normal(size=10_000)
    assert coverage(truth, np.zeros_like(truth), 0.1 * np.ones_like(truth), 1.0) < 0.2


def test_rank_correlation_is_positive_when_uncertainty_tracks_error():
    rng = np.random.default_rng(2)
    std = rng.uniform(0.1, 2.0, size=5_000)
    truth = rng.normal(0.0, std)
    assert error_rank_correlation(truth, np.zeros_like(truth), std) > 0.3
