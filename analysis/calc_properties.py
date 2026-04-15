import os
import pickle as pkl
from itertools import chain
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.rdchem import HybridizationType

# ===== CONFIGURATION =====
# Update this path to your data directory
DATA_ROOT = Path(".")  # Update this to your data root directory
SAMPLES_DIR = DATA_ROOT / "samples"

# ----------------------------
# Chemistry helpers
# ----------------------------
def num_carboaromatic_rings(mol: Chem.Mol) -> int:
    ring_info = mol.GetRingInfo()
    count = 0
    for ring in ring_info.AtomRings():
        if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            if all(mol.GetAtomWithIdx(i).GetAtomicNum() == 6 for i in ring):
                count += 1
    return count


def num_sp3_carbons(mol: Chem.Mol) -> int:
    return sum(
        1 for a in mol.GetAtoms()
        if a.GetAtomicNum() == 6 and a.GetHybridization().name == "SP3"
    )

def num_carbons(mol: Chem.Mol) -> int:
    return sum(
        1 for a in mol.GetAtoms()
        if a.GetAtomicNum() == 6
    )

def num_cyclic_sp3_carbons(mol: Chem.Mol) -> int:
    """
    Count the number of cyclic sp3 carbons in an RDKit molecule.

    Definition:
      - Carbon atom (atomic number == 6)
      - In a ring (atom.IsInRing() == True)
      - sp3 hybridized (atom.GetHybridization() == HybridizationType.SP3)

    Notes:
      - Requires a properly sanitized molecule for best hybridization assignment.
    """
    if mol is None:
        return 0

    return sum(
        1
        for atom in mol.GetAtoms()
        if atom.GetAtomicNum() == 6
        and atom.IsInRing()
        and atom.GetHybridization() == HybridizationType.SP3
    )

def num_aromatic_nitrogen(mol: Chem.Mol) -> int:
    """
    Count aromatic nitrogen atoms in an RDKit molecule.
    Aromatic = atom.GetIsAromatic() is True AND atomic number is 7.
    """
    if mol is None:
        return 0
    return sum(1 for a in mol.GetAtoms() if a.GetAtomicNum() == 7 and a.GetIsAromatic())


N_properties = 10

def props_from_smiles(smi: str):
    if smi is None or smi == "" or (isinstance(smi, float) and pd.isna(smi)):
        return (None,) * N_properties

    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return (None,) * N_properties

    return (
        mol.GetNumHeavyAtoms(),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumRotatableBonds(mol),
        num_sp3_carbons(mol),
        num_carboaromatic_rings(mol),
        rdMolDescriptors.CalcFractionCSP3(mol),
        num_aromatic_nitrogen(mol),
        num_carbons(mol),
        num_cyclic_sp3_carbons(mol),
    )


# ----------------------------
# Core pipeline
# ----------------------------
def compute_and_attach_properties(
    in_path: str,
    out_path: str,
    input_col: str = "input_smiles_canonical",
    generated_col: str = "generated_smiles_canonical",
):
    print(f"[INFO] Loading {in_path}")
    with open(in_path, "rb") as f:
        df = pkl.load(f)

    # ---- collect unique smiles ----
    all_input = df[input_col].dropna().astype(str).values
    gen_lists = df[generated_col].dropna().values
    all_generated = [s for s in chain.from_iterable(gen_lists) if s is not None]

    unique_smiles = pd.unique(
        pd.Series(list(chain(all_input, all_generated)), dtype="object")
    )

    print(f"[INFO] Computing properties for {len(unique_smiles):,} unique SMILES")

    n_jobs = max(os.cpu_count() - 1, 1)
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(props_from_smiles)(smi) for smi in unique_smiles
    )

    prop_map = dict(zip(unique_smiles, results))
    MISSING = (None,) * N_properties

    # ---- attach input props ----
    inp = df[input_col]
    df["input_num_heavy_atoms"]      = inp.map(lambda s: prop_map.get(s, MISSING)[0])
    df["input_hba"]                 = inp.map(lambda s: prop_map.get(s, MISSING)[1])
    df["input_hbd"]                 = inp.map(lambda s: prop_map.get(s, MISSING)[2])
    df["input_num_rot_bonds"]       = inp.map(lambda s: prop_map.get(s, MISSING)[3])
    df["input_num_sp3_carbons"]     = inp.map(lambda s: prop_map.get(s, MISSING)[4])
    df["input_num_carboarom_rings"] = inp.map(lambda s: prop_map.get(s, MISSING)[5])
    df["input_frac_sp3_carbon"]     = inp.map(lambda s: prop_map.get(s, MISSING)[6])
    df["input_num_arom_nitrogens"]     = inp.map(lambda s: prop_map.get(s, MISSING)[7])
    df["input_num_carbons"]     = inp.map(lambda s: prop_map.get(s, MISSING)[8])
    df["input_num_cyclic_sp3_carbons"]     = inp.map(lambda s: prop_map.get(s, MISSING)[9])

    # ---- attach generated props ----
    def map_generated(smis, idx):
        return [prop_map.get(smi, MISSING)[idx] for smi in smis if smi is not None]

    for i, name in enumerate([
        "num_heavy_atoms",
        "hba",
        "hbd",
        "num_rot_bonds",
        "num_sp3_carbons",
        "num_carboarom_rings",
        "frac_sp3_carbon",
        "num_arom_nitrogens",
        "num_carbons",
        "num_cyclic_sp3_carbons",
    ]):
        df[f"generated_{name}"] = df[generated_col].apply(
            lambda smis, i=i: map_generated(smis, i)
        )

    # ---- save ----
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pkl.dump(df, f)

    print(f"[INFO] Saved → {out_path}")

JOBS = [
    (
        SAMPLES_DIR / "reinvent" / "pmv17" / "pmv17_reinvent_1000.pkl",
        SAMPLES_DIR / "reinvent" / "pmv17" / "pmv17_reinvent_1000_properties.pkl",
    ),
    (
        SAMPLES_DIR / "foundation_model_outputs" / "pmv2017_mmpdb_fm_beam1000_canonicalized.pkl",
        SAMPLES_DIR / "foundation_model_outputs" / "pmv2017_mmpdb_fm_beam1000_canonicalized_properties.pkl",
    ),
]

for in_path, out_path in JOBS:
    compute_and_attach_properties(in_path, out_path)
