"""Load the ESOL aqueous solubility dataset (Delaney, 2004)."""

from pathlib import Path

import pandas as pd
from rdkit import Chem

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TARGET_COLUMN = "measured log solubility in mols per litre"


def load_esol(path=None):
    """Return a DataFrame with columns ``smiles`` and ``logS``.

    Rows whose SMILES string RDKit cannot parse are dropped, and each SMILES is
    rewritten in canonical form so the same molecule always looks the same.
    """
    path = Path(path) if path is not None else DATA_DIR / "esol.csv"
    raw = pd.read_csv(path)
    rows = []
    for smiles, value in zip(raw["smiles"], raw[TARGET_COLUMN]):
        mol = Chem.MolFromSmiles(str(smiles).strip())
        if mol is None or mol.GetNumAtoms() == 0:
            continue
        rows.append({"smiles": Chem.MolToSmiles(mol), "logS": float(value)})
    return pd.DataFrame(rows)
