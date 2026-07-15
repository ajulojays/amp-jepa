#!/usr/bin/env python3
"""Prepare a deduplicated AMP corpus for AMP-JEPA-Hybrid v3."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd

from ampjepa_hybrid_v3 import clean_sequence, is_valid_sequence

SEQ_COLUMNS = ["sequence", "Sequence", "PeptideSequence", "peptide_sequence", "aa_sequence"]


def read_fasta(path: Path) -> pd.DataFrame:
    rows = []
    rid = None
    chunks: List[str] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if rid is not None:
                    rows.append({"source_id": rid, "sequence": clean_sequence("".join(chunks))})
                rid = line[1:].split()[0] or f"record_{len(rows)+1}"
                chunks = []
            else:
                chunks.append(line)
        if rid is not None:
            rows.append({"source_id": rid, "sequence": clean_sequence("".join(chunks))})
    return pd.DataFrame(rows)


def read_table(path: Path) -> pd.DataFrame:
    sep = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
    df = pd.read_csv(path, sep=sep)
    seq_col = next((c for c in SEQ_COLUMNS if c in df.columns), None)
    if seq_col is None:
        raise ValueError(f"No sequence column found in {path}; tried {SEQ_COLUMNS}")
    id_col = next((c for c in ["id", "ID", "name", "Name", "accession", "Accession", "Unnamed: 0"] if c in df.columns), None)
    out = pd.DataFrame()
    out["source_id"] = df[id_col].astype(str) if id_col else [f"{path.stem}_{i+1}" for i in range(len(df))]
    out["sequence"] = df[seq_col].map(clean_sequence)
    return out


def read_input(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".fa", ".fasta", ".faa"}:
        return read_fasta(path)
    if path.suffix.lower() in {".csv", ".tsv", ".tab"}:
        return read_table(path)
    raise ValueError(f"Unsupported file type: {path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--output", default="v3/data/processed/peptide_corpus_v3.csv")
    p.add_argument("--min-len", type=int, default=8)
    p.add_argument("--max-len", type=int, default=64)
    args = p.parse_args()

    frames = []
    for raw in args.inputs:
        path = Path(raw)
        if not path.exists():
            print(f"[WARN] Missing input, skipping: {path}")
            continue
        df = read_input(path)
        df["source_file"] = str(path)
        frames.append(df)
        print(f"[INFO] Read {len(df):,} rows from {path}")

    if not frames:
        raise SystemExit("[ERROR] No input files were read.")

    raw = pd.concat(frames, ignore_index=True)
    raw["sequence"] = raw["sequence"].map(clean_sequence)
    raw["valid"] = raw["sequence"].map(lambda s: is_valid_sequence(s, args.min_len, args.max_len))
    valid = raw.loc[raw["valid"]].copy()
    out = (
        valid.groupby("sequence", as_index=False)
        .agg(
            n_source_records=("source_id", "count"),
            source_ids=("source_id", lambda x: ";".join(sorted(set(map(str, x))))),
            source_files=("source_file", lambda x: ";".join(sorted(set(map(str, x))))),
        )
        .sort_values(["n_source_records", "sequence"], ascending=[False, True])
        .reset_index(drop=True)
    )
    out.insert(0, "peptide_id", [f"v3_pep_{i:08d}" for i in range(1, len(out) + 1)])
    out["length"] = out["sequence"].str.len()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"[INFO] Raw rows: {len(raw):,}")
    print(f"[INFO] Valid rows: {len(valid):,}")
    print(f"[INFO] Unique sequences: {len(out):,}")
    print(f"[DONE] Wrote {out_path}")


if __name__ == "__main__":
    main()
