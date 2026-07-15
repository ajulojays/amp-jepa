#!/usr/bin/env python3
"""Stage 1A: curate a canonical peptide corpus for AMP-JEPA.

Accepts FASTA, CSV, or TSV inputs. The script canonicalizes amino-acid sequences,
filters by length and alphabet, removes exact duplicates, and records which source
files contributed each sequence.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

CANONICAL_AA = set("ACDEFGHIKLMNPQRSTVWY")
SEQUENCE_COLUMNS = [
    "sequence",
    "Sequence",
    "peptide_sequence",
    "PeptideSequence",
    "aa_sequence",
    "amino_acid_sequence",
]


def info(message: str) -> None:
    print(f"[INFO] {message}")


def warn(message: str) -> None:
    print(f"[WARN] {message}")


def clean_sequence(seq: str) -> str:
    return "".join(str(seq).upper().split())


def is_valid_sequence(seq: str, min_len: int, max_len: int) -> bool:
    seq = clean_sequence(seq)
    return min_len <= len(seq) <= max_len and all(aa in CANONICAL_AA for aa in seq)


def sequence_hash(seq: str) -> str:
    return hashlib.sha1(seq.encode("utf-8")).hexdigest()[:16]


def read_fasta(path: Path) -> pd.DataFrame:
    rows = []
    current_id = None
    chunks: List[str] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    rows.append({"source_record_id": current_id, "sequence": clean_sequence("".join(chunks))})
                current_id = line[1:].strip().split()[0] or f"record_{len(rows)+1}"
                chunks = []
            else:
                chunks.append(line)
        if current_id is not None:
            rows.append({"source_record_id": current_id, "sequence": clean_sequence("".join(chunks))})
    return pd.DataFrame(rows)


def read_table(path: Path) -> pd.DataFrame:
    sep = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
    df = pd.read_csv(path, sep=sep)
    seq_col = next((col for col in SEQUENCE_COLUMNS if col in df.columns), None)
    if seq_col is None:
        raise ValueError(f"No sequence column found in {path}; tried {SEQUENCE_COLUMNS}")
    id_col = next((col for col in ["id", "ID", "name", "Name", "accession", "Accession", "Unnamed: 0"] if col in df.columns), None)
    out = pd.DataFrame()
    out["source_record_id"] = df[id_col].astype(str) if id_col else [f"{path.stem}_{i+1}" for i in range(len(df))]
    out["sequence"] = df[seq_col].map(clean_sequence)
    for optional in ["label", "activity", "evidence", "source_database", "database"]:
        if optional in df.columns:
            out[optional] = df[optional]
    return out


def read_input(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".fa", ".fasta", ".faa", ".fna"}:
        return read_fasta(path)
    if suffix in {".csv", ".tsv", ".tab"}:
        return read_table(path)
    raise ValueError(f"Unsupported input type: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", nargs="+", required=True, help="Input FASTA/CSV/TSV files")
    parser.add_argument("--output", default="data/processed/stage1/peptide_corpus.csv")
    parser.add_argument("--min-len", type=int, default=5)
    parser.add_argument("--max-len", type=int, default=100)
    args = parser.parse_args()

    all_rows = []
    for raw in args.inputs:
        path = Path(raw)
        if not path.exists():
            warn(f"Skipping missing file: {path}")
            continue
        info(f"Reading {path}")
        df = read_input(path)
        if df.empty:
            warn(f"No records found in {path}")
            continue
        df["source_file"] = str(path)
        df["source_name"] = path.parent.name if path.parent.name else path.stem
        all_rows.append(df)

    if not all_rows:
        raise SystemExit("[ERROR] No usable inputs were read.")

    raw_df = pd.concat(all_rows, ignore_index=True)
    raw_df["sequence"] = raw_df["sequence"].map(clean_sequence)
    raw_df["length"] = raw_df["sequence"].str.len()
    raw_df["is_valid"] = raw_df["sequence"].map(lambda x: is_valid_sequence(x, args.min_len, args.max_len))

    invalid = int((~raw_df["is_valid"]).sum())
    info(f"Raw records: {len(raw_df):,}")
    info(f"Invalid/filtered records: {invalid:,}")

    valid = raw_df.loc[raw_df["is_valid"]].copy()
    grouped = (
        valid.groupby("sequence", as_index=False)
        .agg(
            source_record_ids=("source_record_id", lambda x: ";".join(sorted(set(map(str, x))))),
            source_files=("source_file", lambda x: ";".join(sorted(set(map(str, x))))),
            source_names=("source_name", lambda x: ";".join(sorted(set(map(str, x))))),
            n_source_records=("source_record_id", "count"),
        )
        .sort_values(["n_source_records", "sequence"], ascending=[False, True])
        .reset_index(drop=True)
    )
    grouped.insert(0, "peptide_id", [f"pep_{i:08d}" for i in range(1, len(grouped) + 1)])
    grouped["sequence_hash"] = grouped["sequence"].map(sequence_hash)
    grouped["length"] = grouped["sequence"].str.len()
    grouped["evidence_tier"] = "unassigned"

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(out_path, index=False)

    info(f"Canonical unique peptides: {len(grouped):,}")
    info(f"Length range: {grouped['length'].min()}-{grouped['length'].max()} aa")
    print(f"[DONE] Wrote {out_path}")


if __name__ == "__main__":
    main()
