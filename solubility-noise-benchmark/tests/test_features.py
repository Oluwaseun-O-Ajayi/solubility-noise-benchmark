import numpy as np
import torch
from rdkit import Chem

from solnoise.features import (ATOM_DIM, BOND_DIM, DESCRIPTOR_NAMES, collate_graphs,
                               descriptor_matrix, fingerprint_matrix, mol_to_graph, smiles_to_graphs)

SMILES = ["CCO", "c1ccccc1O", "CC(=O)Oc1ccccc1C(=O)O", "C"]


def test_descriptor_and_fingerprint_shapes():
    assert descriptor_matrix(SMILES).shape == (4, len(DESCRIPTOR_NAMES))
    fingerprints = fingerprint_matrix(SMILES, n_bits=512)
    assert fingerprints.shape == (4, 512)
    assert set(np.unique(fingerprints)) <= {0.0, 1.0}


def test_graph_dimensions():
    graph = mol_to_graph(Chem.MolFromSmiles("CCO"))
    assert graph.x.shape == (3, ATOM_DIM)
    assert graph.edge_index.shape == (2, 4)      # two bonds, each stored in both directions
    assert graph.edge_attr.shape == (4, BOND_DIM)


def test_single_atom_molecule_has_no_edges():
    graph = mol_to_graph(Chem.MolFromSmiles("C"))
    assert graph.edge_index.shape == (2, 0)
    assert graph.edge_attr.shape == (0, BOND_DIM)


def test_collate_offsets_atom_indices():
    graphs = smiles_to_graphs(SMILES)
    batch = collate_graphs(graphs)
    assert batch.n_graphs == 4
    assert batch.x.shape[0] == sum(g.n_atoms for g in graphs)
    assert int(batch.edge_index.max()) < batch.x.shape[0]
    # every edge must connect two atoms of the same molecule
    source, target = batch.edge_index
    assert torch.equal(batch.batch[source], batch.batch[target])
