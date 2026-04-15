import pickle as pkl
from pathlib import Path

import pandas as pd
from chemical_space import get_umap

# ===== CONFIGURATION =====
# Update these paths to point to your data directory
DATA_ROOT = Path(".")  # Update this to your data root directory
SAMPLES_DIR = DATA_ROOT / "samples"

# ---------- load ----------
samples_dict = {}
for n in [1000, 2000, 5000, 10000]:
    input_file = SAMPLES_DIR / "reinvent" / "pmv17" / f"pmv17_reinvent_{n}.pkl"
    with open(input_file, "rb") as f:
        samples_dict[n] = pkl.load(f)

def build_all_smiles_df(df: pd.DataFrame) -> pd.DataFrame:
    return (
        pd.concat(
            [
                df[["input_smiles_canonical"]]
                .rename(columns={"input_smiles_canonical": "smiles"})
                .assign(source="input"),

                df[["generated_smiles_canonical"]]
                .explode("generated_smiles_canonical")
                .rename(columns={"generated_smiles_canonical": "smiles"})
                .assign(source="generated"),
            ],
            ignore_index=True,
        )
        .dropna(subset=["smiles"])
    )

# ---------- build all df_all_smiles + union unique smiles ----------
all_smiles_dfs = {}
union_unique = []

for n in [1000, 2000, 5000, 10000]:
    df = samples_dict[n]
    df_all_smiles = build_all_smiles_df(df)
    all_smiles_dfs[n] = df_all_smiles
    union_unique.append(df_all_smiles[["smiles"]])

df_union = pd.concat(union_unique, ignore_index=True).drop_duplicates()

# ---------- compute UMAP ONCE ----------
# IMPORTANT: smiles_column should be "smiles" (not "input_smiles_canonical")
df_union_umap = get_umap(df_union, smiles_column="smiles")

# Figure out which columns are the embedding columns (unknown names)
embed_cols = [c for c in df_union_umap.columns if c != "smiles"]

# ---------- merge back + save ----------
for n in [1000, 2000, 5000, 10000]:
    df_all_smiles = all_smiles_dfs[n].merge(df_union_umap, on="smiles", how="left")

    output_dir = SAMPLES_DIR / "reinvent" / "pmv17"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"pmv17_reinvent_{n}_umap.pkl"
    with open(out_path, "wb") as f:
        pkl.dump(df_all_smiles, f)