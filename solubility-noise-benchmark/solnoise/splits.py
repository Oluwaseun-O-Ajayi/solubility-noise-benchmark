"""Train/validation/test splits: random, and by Bemis-Murcko scaffold."""

from collections import defaultdict

import numpy as np
from rdkit.Chem.Scaffolds import MurckoScaffold


def random_split(n, seed, frac_train=0.8, frac_valid=0.1):
    order = np.random.default_rng(seed).permutation(n)
    n_train, n_valid = int(frac_train * n), int(frac_valid * n)
    return order[:n_train], order[n_train:n_train + n_valid], order[n_train + n_valid:]


def scaffold_of(smiles):
    """Ring-system core of a molecule. Molecules without rings share the scaffold ''."""
    return MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles, includeChirality=False)


def scaffold_split(smiles_list, seed, frac_train=0.8, frac_valid=0.1):
    """Put whole scaffold groups into train, validation, or test.

    No scaffold appears in more than one set, so the test set contains ring
    systems the model has never seen. Groups larger than half the validation
    set go to train first (otherwise one big group could swallow the test set).
    The remaining groups are shuffled by ``seed`` and assigned in that order,
    so different seeds give different splits.
    """
    groups = defaultdict(list)
    for i, smiles in enumerate(smiles_list):
        groups[scaffold_of(smiles)].append(i)
    group_list = [groups[key] for key in sorted(groups)]

    n = len(smiles_list)
    train_cap, valid_cap = frac_train * n, frac_valid * n
    big = [g for g in group_list if len(g) > valid_cap / 2]
    small = [g for g in group_list if len(g) <= valid_cap / 2]
    order = np.random.default_rng(seed).permutation(len(small))

    train, valid, test = [], [], []
    for members in big + [small[k] for k in order]:
        if len(train) + len(members) <= train_cap:
            train += members
        elif len(valid) + len(members) <= valid_cap:
            valid += members
        else:
            test += members
    return np.array(train), np.array(valid), np.array(test)
