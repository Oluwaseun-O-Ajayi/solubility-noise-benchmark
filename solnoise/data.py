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


AQSOLDB_SHA256 = "363a42a4e4ea8e6039758146d34efdd95cde2da81c665316cacc1869e0a805ed"
LIPO_SHA256 = "aed41590cb30609d51d8e08ad3ff06495a76e80e211358801f596b10da69bacd"
ALLOWED_ELEMENTS = {"C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "B", "Si", "Se", "H"}


def load_aqsoldb(path=None, verify=True):
    """Return AqSolDB (Sorkun et al., 2019) curated for molecular machine learning.

    Columns: ``smiles``, ``logS``, ``sd`` (standard deviation across sources, 0 for
    single-source compounds) and ``group`` (reliability group G1 to G5).

    Curation, applied in this order and counted in ``frame.attrs["curation"]``:
    multi-component entries (salts, mixtures, hydrates) are dropped; unparseable
    SMILES are dropped; molecules containing elements outside common organic
    chemistry (metals, for example) are dropped; molecules with fewer than two
    heavy atoms are dropped; SMILES are canonicalized without stereochemistry and,
    where two entries collapse to one structure, the first is kept.
    """
    import hashlib

    path = Path(path) if path is not None else DATA_DIR / "aqsoldb.csv"
    if verify:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != AQSOLDB_SHA256:
            raise ValueError(f"{path} does not match the AqSolDB release used here (sha256 {digest})")
    raw = pd.read_csv(path)
    counts = {"raw": len(raw)}
    rows = []
    dropped = {"unparseable": 0, "multicomponent": 0, "elements": 0, "small": 0}
    for smiles, value, sd, group in zip(raw["SMILES"], raw["Solubility"], raw["SD"], raw["Group"]):
        if "." in str(smiles):
            dropped["multicomponent"] += 1
            continue
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            dropped["unparseable"] += 1
            continue
        if any(atom.GetSymbol() not in ALLOWED_ELEMENTS for atom in mol.GetAtoms()):
            dropped["elements"] += 1
            continue
        if mol.GetNumHeavyAtoms() < 2:
            dropped["small"] += 1
            continue
        rows.append({"smiles": Chem.MolToSmiles(mol, isomericSmiles=False), "logS": float(value),
                     "sd": float(sd), "group": str(group)})
    frame = pd.DataFrame(rows)
    before = len(frame)
    frame = frame.drop_duplicates("smiles", keep="first").reset_index(drop=True)
    counts.update(dropped)
    counts["duplicates"] = before - len(frame)
    counts["final"] = len(frame)
    frame.attrs["curation"] = counts
    return frame


def load_lipophilicity(path=None, verify=True):
    """Return the MoleculeNet lipophilicity set (experimental logD at pH 7.4, from ChEMBL).

    Columns: ``smiles`` and ``logS`` (the label column is named ``logS`` throughout
    the benchmark; here it holds logD values, not solubilities).

    The same curation as AqSolDB is applied, in the same order, so the two sets are
    treated identically: multi-component entries are dropped, then unparseable
    SMILES, then molecules containing elements outside common organic chemistry,
    then molecules with fewer than two heavy atoms; SMILES are canonicalized
    without stereochemistry and duplicates collapsed to the first occurrence.

    This set carries no published descriptor equation, which is why it is used
    here as a control for the representation comparison: unlike ESOL, none of the
    descriptors were selected on it.
    """
    import hashlib

    path = Path(path) if path is not None else DATA_DIR / "lipophilicity.csv"
    if verify:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != LIPO_SHA256:
            raise ValueError(f"{path} does not match the lipophilicity release used here (sha256 {digest})")
    raw = pd.read_csv(path)
    counts = {"raw": len(raw)}
    rows = []
    dropped = {"unparseable": 0, "multicomponent": 0, "elements": 0, "small": 0}
    for smiles, value in zip(raw["smiles"], raw["exp"]):
        if "." in str(smiles):
            dropped["multicomponent"] += 1
            continue
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            dropped["unparseable"] += 1
            continue
        if any(atom.GetSymbol() not in ALLOWED_ELEMENTS for atom in mol.GetAtoms()):
            dropped["elements"] += 1
            continue
        if mol.GetNumHeavyAtoms() < 2:
            dropped["small"] += 1
            continue
        rows.append({"smiles": Chem.MolToSmiles(mol, isomericSmiles=False), "logS": float(value)})
    frame = pd.DataFrame(rows)
    before = len(frame)
    frame = frame.drop_duplicates("smiles", keep="first").reset_index(drop=True)
    counts.update(dropped)
    counts["duplicates"] = before - len(frame)
    counts["final"] = len(frame)
    frame.attrs["curation"] = counts
    return frame
