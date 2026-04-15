from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import umap
from joblib import Parallel, delayed
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator


def get_fp(mol, fp_size=2048):
        fpgen_rd = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=fp_size)
        return fpgen_rd.GetCountFingerprintAsNumPy(mol) if mol else np.zeros(fp_size)

def get_umap(
    df: pd.DataFrame,
    smiles_column: str = "SMILES",
    n_neighbors: int = 25,
    min_dist: float = 0.001,
    plot: bool = False
) -> pd.DataFrame:
    """
    Generate UMAP embeddings from SMILES strings with parallel fingerprint generation.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing SMILES strings.
    smiles_column : str, default="SMILES"
        Column name for SMILES strings.
    fp_size : int, default=2048
        Size of RDKit fingerprint.
    n_neighbors : int, default=25
        UMAP n_neighbors parameter.
    min_dist : float, default=0.001
        UMAP min_dist parameter.
    plot : bool, default=False
        If True, plots UMAP embeddings using seaborn.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with added columns: ['umap_x', 'umap_y'].
    """

    # Validate SMILES column
    if smiles_column not in df.columns:
        raise ValueError(f"Column '{smiles_column}' not found in DataFrame.")

    # Convert SMILES to RDKit Mol objects
    df["mol"] = df[smiles_column].apply(Chem.MolFromSmiles)
    if df["mol"].isnull().any():
        print("Warning: Some SMILES could not be parsed into molecules.")

    fingerprints = Parallel(n_jobs=-1, backend="multiprocessing")(
        delayed(get_fp)(mol) for mol in df["mol"]
    )
    all_fgrps = np.vstack(fingerprints)

    # UMAP embedding (parallelized)
    umap_model = umap.UMAP(
        metric="jaccard",
        n_neighbors=n_neighbors,
        n_components=2,
        min_dist=min_dist,
        n_jobs=-1
    )
    X_umap = umap_model.fit_transform(all_fgrps)
    df['umap_x'], df['umap_y'] = X_umap[:, 0], X_umap[:, 1]

    # Optional plotting
    if plot:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df, x="umap_x", y="umap_y")
        plt.title("UMAP Embedding")
        plt.tight_layout()
        plt.show()

    return df