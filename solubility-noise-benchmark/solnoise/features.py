"""Turn molecules into numbers: descriptors, fingerprints, and graphs."""

from dataclasses import dataclass

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdFingerprintGenerator, rdMolDescriptors

# ---------------------------------------------------------------- descriptors

DESCRIPTOR_NAMES = [
    "MolWt", "MolLogP", "MolMR", "TPSA", "HBD", "HBA", "RotatableBonds",
    "RingCount", "AromaticRings", "HeavyAtoms", "FractionCSP3", "AromaticProportion",
]


def descriptor_vector(mol):
    """Twelve whole-molecule descriptors that relate to solubility."""
    heavy = mol.GetNumHeavyAtoms()
    aromatic_atoms = sum(atom.GetIsAromatic() for atom in mol.GetAtoms())
    return np.array([
        Descriptors.MolWt(mol),
        Crippen.MolLogP(mol),
        Crippen.MolMR(mol),
        rdMolDescriptors.CalcTPSA(mol),
        Lipinski.NumHDonors(mol),
        Lipinski.NumHAcceptors(mol),
        Lipinski.NumRotatableBonds(mol),
        rdMolDescriptors.CalcNumRings(mol),
        rdMolDescriptors.CalcNumAromaticRings(mol),
        heavy,
        rdMolDescriptors.CalcFractionCSP3(mol),
        aromatic_atoms / heavy if heavy else 0.0,
    ], dtype=np.float64)


def descriptor_matrix(smiles_list):
    return np.vstack([descriptor_vector(Chem.MolFromSmiles(s)) for s in smiles_list])


# --------------------------------------------------------------- fingerprints

def fingerprint_matrix(smiles_list, radius=2, n_bits=2048):
    """Morgan (circular) bit fingerprints, one row per molecule."""
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    out = np.zeros((len(smiles_list), n_bits), dtype=np.float32)
    for i, smiles in enumerate(smiles_list):
        out[i] = generator.GetFingerprintAsNumPy(Chem.MolFromSmiles(smiles))
    return out


# --------------------------------------------------------------------- graphs

ELEMENTS = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P"]  # anything else -> "other"
HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
]
BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]


def _one_hot(value, choices):
    """One-hot encode ``value``; the last slot means "none of the listed choices"."""
    vec = [0.0] * (len(choices) + 1)
    vec[choices.index(value) if value in choices else len(choices)] = 1.0
    return vec


def atom_features(atom):
    return (
        _one_hot(atom.GetSymbol(), ELEMENTS)
        + _one_hot(atom.GetDegree(), [0, 1, 2, 3, 4])
        + _one_hot(atom.GetTotalNumHs(), [0, 1, 2, 3])
        + _one_hot(atom.GetHybridization(), HYBRIDIZATIONS)
        + [float(atom.GetFormalCharge()), float(atom.GetIsAromatic()), float(atom.IsInRing())]
    )


def bond_features(bond):
    return _one_hot(bond.GetBondType(), BOND_TYPES) + [
        float(bond.GetIsConjugated()),
        float(bond.IsInRing()),
    ]


ATOM_DIM = len(ELEMENTS) + 1 + 6 + 5 + len(HYBRIDIZATIONS) + 1 + 3
BOND_DIM = len(BOND_TYPES) + 1 + 2


@dataclass
class MolGraph:
    """One molecule as a graph. Every bond appears twice, once in each direction."""

    x: torch.Tensor           # [n_atoms, ATOM_DIM]
    edge_index: torch.Tensor  # [2, n_directed_edges], rows are (source, target)
    edge_attr: torch.Tensor   # [n_directed_edges, BOND_DIM]

    @property
    def n_atoms(self):
        return self.x.shape[0]


def mol_to_graph(mol):
    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float32)
    sources, targets, attrs = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feats = bond_features(bond)
        sources += [i, j]
        targets += [j, i]
        attrs += [feats, feats]
    edge_index = torch.tensor([sources, targets], dtype=torch.long).reshape(2, -1)
    edge_attr = torch.tensor(attrs, dtype=torch.float32).reshape(-1, BOND_DIM)
    return MolGraph(x, edge_index, edge_attr)


def smiles_to_graphs(smiles_list):
    return [mol_to_graph(Chem.MolFromSmiles(s)) for s in smiles_list]


@dataclass
class GraphBatch:
    """Several molecules joined into one big disconnected graph."""

    x: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    batch: torch.Tensor  # [total_atoms], which molecule each atom belongs to
    n_graphs: int


def collate_graphs(graphs):
    xs, edge_indices, edge_attrs, owners = [], [], [], []
    offset = 0
    for k, g in enumerate(graphs):
        xs.append(g.x)
        edge_indices.append(g.edge_index + offset)
        edge_attrs.append(g.edge_attr)
        owners.append(torch.full((g.n_atoms,), k, dtype=torch.long))
        offset += g.n_atoms
    return GraphBatch(
        torch.cat(xs), torch.cat(edge_indices, dim=1), torch.cat(edge_attrs),
        torch.cat(owners), len(graphs),
    )
