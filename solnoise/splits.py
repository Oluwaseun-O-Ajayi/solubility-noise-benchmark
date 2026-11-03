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


def butina_clusters(fingerprints, threshold=0.4, block=2048):
    """Taylor-Butina clustering on Morgan fingerprints.

    Two molecules are neighbours when their Tanimoto similarity is at least
    ``threshold``. The molecule with the most unassigned neighbours becomes a
    centroid and takes all of them, until every molecule is assigned. Returns a
    cluster label per molecule. Memory stays bounded by working in row blocks.
    """
    from .models import tanimoto

    fps = np.asarray(fingerprints, dtype=np.float32)
    n = len(fps)
    neighbours = []
    for start in range(0, n, block):
        sim = tanimoto(fps[start:start + block], fps)
        neighbours += [np.flatnonzero(row >= threshold) for row in sim]
    labels = np.full(n, -1)
    counts = np.array([len(nb) for nb in neighbours])
    cluster = 0
    for centre in np.argsort(-counts, kind="stable"):
        if labels[centre] != -1:
            continue
        members = neighbours[centre][labels[neighbours[centre]] == -1]
        labels[members] = cluster
        labels[centre] = cluster
        cluster += 1
    return labels


def cluster_split(fingerprints, y, seed, frac_train=0.8, frac_valid=0.1, threshold=0.4,
                  n_candidates=200, labels=None):
    """Whole Butina clusters to train, validation or test, matched on the label mean.

    Like the scaffold split, no cluster is shared between sets. Unlike it, the test
    set is chosen among ``n_candidates`` random cluster orders as the one whose
    mean label is closest to the training mean, so the split tests structural
    novelty without also shifting the solubility range.
    """
    y = np.asarray(y, dtype=np.float64)
    if labels is None:
        labels = butina_clusters(fingerprints, threshold)
    groups = [np.flatnonzero(labels == c) for c in np.unique(labels)]
    n = len(y)
    valid_cap = int(round(frac_valid * n))
    test_cap = n - int(round(frac_train * n)) - valid_cap
    big = [g for g in groups if len(g) > test_cap / 2]
    small = [g for g in groups if len(g) <= test_cap / 2]
    rng = np.random.default_rng(seed)
    best, best_gap = None, np.inf
    for _ in range(n_candidates):
        order = rng.permutation(len(small))
        test, valid, train = [], [], [g for g in big]
        n_test = n_valid = 0
        for k in order:
            g = small[k]
            if n_test + len(g) <= test_cap:
                test.append(g)
                n_test += len(g)
            elif n_valid + len(g) <= valid_cap:
                valid.append(g)
                n_valid += len(g)
            else:
                train.append(g)
        tr, va, te = (np.concatenate(s) if s else np.array([], int) for s in (train, valid, test))
        if len(te) == 0 or len(tr) == 0:
            continue
        gap = abs(y[te].mean() - y[tr].mean())
        if gap < best_gap:
            best, best_gap = (np.sort(tr), np.sort(va), np.sort(te)), gap
    if best is None:
        raise ValueError("no cluster assignment produced a non-empty test set; lower the threshold")
    return best
