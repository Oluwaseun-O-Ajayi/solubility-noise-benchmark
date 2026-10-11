import numpy as np

from solnoise.data import load_esol
from solnoise.noise import add_label_noise
from solnoise.splits import random_split, scaffold_of, scaffold_split


def test_random_split_uses_every_index_once():
    train, valid, test = random_split(100, seed=0)
    assert sorted(np.concatenate([train, valid, test])) == list(range(100))
    assert (len(train), len(valid), len(test)) == (80, 10, 10)


def test_scaffold_split_keeps_scaffolds_apart():
    smiles = load_esol()["smiles"].tolist()
    train, valid, test = scaffold_split(smiles, seed=0)
    assert sorted(np.concatenate([train, valid, test])) == list(range(len(smiles)))
    scaffolds = [{scaffold_of(smiles[i]) for i in part} for part in (train, valid, test)]
    assert not scaffolds[0] & scaffolds[1]
    assert not scaffolds[0] & scaffolds[2]
    assert not scaffolds[1] & scaffolds[2]


def test_scaffold_split_changes_with_seed():
    smiles = load_esol()["smiles"].tolist()
    assert set(scaffold_split(smiles, 0)[2]) != set(scaffold_split(smiles, 1)[2])


def test_noise_is_reproducible_and_has_the_requested_size():
    y = np.zeros(20_000)
    noisy = add_label_noise(y, sigma=0.5, seed=3)
    assert np.array_equal(noisy, add_label_noise(y, sigma=0.5, seed=3))
    assert abs(noisy.std() - 0.5) < 0.02
    assert abs(noisy.mean()) < 0.02


def test_zero_noise_returns_an_unchanged_copy():
    y = np.array([1.0, 2.0, 3.0])
    out = add_label_noise(y, sigma=0.0, seed=0)
    assert np.array_equal(out, y) and out is not y
