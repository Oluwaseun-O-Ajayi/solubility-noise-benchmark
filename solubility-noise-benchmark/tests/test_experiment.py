import numpy as np

from solnoise.experiment import Featurized, run_condition

SMILES = ["CCO", "c1ccccc1O", "CC(=O)Oc1ccccc1C(=O)O", "CCCCCC", "c1ccc2ccccc2c1", "CC(C)O", "OCC(O)CO",
          "Clc1ccccc1", "CCN(CC)CC", "O=C(O)c1ccccc1", "CCCCO", "CCCCCCCC", "c1ccncc1", "CC(C)C",
          "OC1CCCCC1", "CCOC(C)=O", "Brc1ccccc1", "CCCCCCO", "CC(N)=O", "c1ccsc1"]
LOGS = np.linspace(-4.0, 1.0, len(SMILES))


def test_baseline_rows_have_every_metric():
    data = Featurized(SMILES, LOGS)
    rows = run_condition(data, "random", sigma=0.5, seed=0,
                         models=["train_mean", "random_forest"], log=lambda line: None)
    assert [row["model"] for row in rows] == ["train_mean", "random_forest"]
    for row in rows:
        assert {"rmse", "mae", "coverage_68", "coverage_95", "error_rank_corr"} <= set(row)
        assert row["n_train"] + row["n_test"] <= len(SMILES)


def test_noise_changes_training_labels_but_not_the_test_score_reference():
    data = Featurized(SMILES, LOGS)
    clean = run_condition(data, "random", sigma=0.0, seed=1, models=["train_mean"], log=lambda line: None)
    noisy = run_condition(data, "random", sigma=2.0, seed=1, models=["train_mean"], log=lambda line: None)
    assert clean[0]["rmse"] != noisy[0]["rmse"]          # the training mean moved
    assert np.array_equal(data.y, LOGS)                    # the stored clean labels did not
