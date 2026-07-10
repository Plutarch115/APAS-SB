"""
Preprocess the raw BindingDB_All.tsv into a compact CSV usable by the
multi-task training pipeline (pearl.data.multitask_datasets.BindingDBDataset).

The raw dump is ~8.8 GB, so it is streamed in chunks and only the handful of
columns we need are parsed. Rows are filtered to those that have a ligand
SMILES, a target protein sequence, and at least one measured affinity
(Ki / Kd / IC50 / EC50 in nM). Affinities are standardized to log10(uM),
matching the convention used by the Boltz-2 datasets.

Usage:
    python scripts/prepare_bindingdb.py \
        --tsv /nfs/lambda_stor_01/data/avasan/APAS-SB/APAS-SB/data/bindingdb/BindingDB_All.tsv \
        --out data/bindingdb/bindingdb_processed.csv \
        --max-rows 200000        # optional cap on kept rows (0 = keep all)

The resulting CSV has columns:
    smiles, target_name, target_sequence, affinity_type, affinity_nm, affinity_value
where `affinity_value = log10(affinity_nM / 1000)` = log10(uM).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Columns we read from the raw TSV (kept minimal for speed / memory).
SMILES_COL = "Ligand SMILES"
NAME_COL = "Target Name"
SEQ_COL = "BindingDB Target Chain Sequence 1"

# Affinity columns, in order of preference. All are in nM.
AFFINITY_COLS = ["Ki (nM)", "Kd (nM)", "IC50 (nM)", "EC50 (nM)"]

USECOLS = [SMILES_COL, NAME_COL, SEQ_COL] + AFFINITY_COLS


def _clean_affinity(series: pd.Series) -> pd.Series:
    """Strip qualifier characters (>, <, ~, =) and coerce to float (nM)."""
    cleaned = (
        series.astype(str)
        .str.replace(r"[><~=\s]", "", regex=True)
        .replace({"": np.nan, "nan": np.nan})
    )
    return pd.to_numeric(cleaned, errors="coerce")


def process(tsv_path: Path, out_path: Path, max_rows: int, chunksize: int) -> None:
    kept_frames = []
    total_kept = 0
    total_read = 0

    reader = pd.read_csv(
        tsv_path,
        sep="\t",
        usecols=USECOLS,
        chunksize=chunksize,
        low_memory=False,
        on_bad_lines="skip",
        dtype=str,
    )

    for chunk_idx, chunk in enumerate(reader):
        total_read += len(chunk)

        # Must have a ligand and a target protein sequence.
        chunk = chunk[chunk[SMILES_COL].notna() & chunk[SEQ_COL].notna()].copy()
        if chunk.empty:
            continue

        # Pick the first available affinity measurement per row.
        affinity_nm = pd.Series(np.nan, index=chunk.index, dtype=float)
        affinity_type = pd.Series(np.nan, index=chunk.index, dtype=object)
        for col in AFFINITY_COLS:
            if col not in chunk.columns:
                continue
            vals = _clean_affinity(chunk[col])
            take = affinity_nm.isna() & vals.notna() & (vals > 0)
            affinity_nm[take] = vals[take]
            affinity_type[take] = col.split(" ")[0]  # "Ki (nM)" -> "Ki"

        chunk["affinity_nm"] = affinity_nm
        chunk["affinity_type"] = affinity_type
        chunk = chunk[chunk["affinity_nm"].notna()]
        if chunk.empty:
            continue

        # Standardize to log10(uM): nM / 1000 -> uM, then log10.
        chunk["affinity_value"] = np.log10(chunk["affinity_nm"] / 1000.0)

        out = pd.DataFrame(
            {
                "smiles": chunk[SMILES_COL].values,
                "target_name": chunk[NAME_COL].values,
                "target_sequence": chunk[SEQ_COL].values,
                "affinity_type": chunk["affinity_type"].values,
                "affinity_nm": chunk["affinity_nm"].values,
                "affinity_value": chunk["affinity_value"].values,
            }
        )

        if max_rows > 0 and total_kept + len(out) > max_rows:
            out = out.iloc[: max_rows - total_kept]

        kept_frames.append(out)
        total_kept += len(out)
        print(
            f"chunk {chunk_idx}: read={total_read:,} kept_total={total_kept:,}",
            flush=True,
        )

        if max_rows > 0 and total_kept >= max_rows:
            print(f"Reached max-rows={max_rows:,}, stopping early.")
            break

    if not kept_frames:
        raise RuntimeError("No usable rows found in BindingDB TSV.")

    result = pd.concat(kept_frames, ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)
    print(f"\nWrote {len(result):,} rows to {out_path}")
    print(result.head())


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess BindingDB_All.tsv")
    parser.add_argument(
        "--tsv",
        type=str,
        default="/nfs/lambda_stor_01/data/avasan/APAS-SB/APAS-SB/data/bindingdb/BindingDB_All.tsv",
        help="Path to raw BindingDB_All.tsv",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/bindingdb/bindingdb_processed.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Cap on number of kept rows (0 = keep all)",
    )
    parser.add_argument("--chunksize", type=int, default=200_000)
    args = parser.parse_args()

    process(Path(args.tsv), Path(args.out), args.max_rows, args.chunksize)


if __name__ == "__main__":
    main()
