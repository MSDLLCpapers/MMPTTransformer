from __future__ import annotations

import pickle as pkl
from functools import lru_cache
from pathlib import Path

import pandas as pd
from rdkit import Chem


@lru_cache(maxsize=1_000_000)
def canonicalize_cached(smi):
    if smi is None:
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def canon_set(smis):
    out = set()
    for s in smis:
        c = canonicalize_cached(s)
        if c is not None:
            out.add(c)
    return out


@lru_cache(maxsize=1_000_000)
def canonicalize_mmpt_cached(smi: str):
    if smi is None:
        return None

    if ">>" in smi:
        smi = smi.split(">>")[-1].strip()
    elif ">" in smi:
        smi = smi.split(">")[-1].strip()

    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def canon_set_mmpt(smis):
    out = set()
    for s in smis:
        c = canonicalize_mmpt_cached(s)
        if c is not None:
            out.add(c)
    return out

def save_with_same_structure(
    input_paths: list[str | Path],
    input_root: str | Path,
    output_root: str | Path,
    *,
    input_suffix: str = ".jsonl",                  # only used for name rewriting
    output_suffix: str = "_canonicalized.pkl",     # appended after removing input_suffix
    read_json_lines: bool = True,
):
    """
    For each input path:
      - read jsonl into df
      - canonicalize df["generated"] -> df["generated_canonical"]
      - write pickle into output_root / relative_path_from_input_root, preserving folders

    Example:
      input_root=/data/in
      output_root=/data/out
      /data/in/a/b/x.jsonl -> /data/out/a/b/x_canonicalized.pkl
    """
    input_root = Path(input_root).resolve()
    output_root = Path(output_root).resolve()

    results = {}  # optional: map input_path -> output_path
    for in_path in map(lambda p: Path(p).resolve(), input_paths):
        rel = in_path.relative_to(input_root)  # raises if in_path not under input_root
        out_path = output_root / rel

        # rewrite filename (keep folders the same)
        if input_suffix and out_path.name.endswith(input_suffix):
            stem = out_path.name[: -len(input_suffix)]
            out_path = out_path.with_name(stem + output_suffix)
        else:
            # fallback: just add output_suffix
            out_path = out_path.with_name(out_path.name + output_suffix)

        print(out_path)
        # if out_path.exists():
        #     print("Exists. Skipping...")
        #     continue

        out_path.parent.mkdir(parents=True, exist_ok=True)

        df = pd.read_json(in_path, lines=read_json_lines)
        df["from_mol_canonical"] = df["from_mol"].map(canonicalize_mmpt_cached)
        df["generated_canonical"] = df["generated"].map(canon_set_mmpt)

        with out_path.open("wb") as f:
            pkl.dump(df, f, protocol=pkl.HIGHEST_PROTOCOL)

        results[str(in_path)] = str(out_path)

    return results

# ===== CONFIGURATION =====
# Update these paths to your data directories
DATA_ROOT = Path(".")  # Update this to your data root directory
base_in = DATA_ROOT / "samples"  # Input directory with JSONL files
base_out = DATA_ROOT / "samples" / "MMP_model_outputs"  # Output directory

# recursive search for *.jsonl
jobs = sorted(str(p) for p in base_in.rglob("*.jsonl"))

print(f"Found {len(jobs)} .jsonl files")
print("\n".join(jobs[:20]))  # preview first 20

written = save_with_same_structure(
    input_paths=jobs,
    input_root=base_in,
    output_root=base_out,
    input_suffix=".jsonl",
    output_suffix="_canonicalized.pkl",
)

