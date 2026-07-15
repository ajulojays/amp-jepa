#!/usr/bin/env python3
"""Summarize key AMP-JEPA-Hybrid v3 output files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

FILES = [
    "v3/data/processed/peptide_corpus_v3.csv",
    "v3/results/raw_candidates_v3.csv",
    "v3/results/ranked_candidates_v3.csv",
    "v3/results/top_panel_v3.csv",
    "v3/results/apex_comparator_v3.csv",
]


def main() -> None:
    for raw in FILES:
        path = Path(raw)
        if not path.exists():
            print(f"[MISSING] {path}")
            continue
        df = pd.read_csv(path)
        print(f"[FOUND] {path}: {len(df):,} rows, {len(df.columns):,} columns")
        print("  columns:", ", ".join(df.columns[:12]))
        if "sequence" in df.columns:
            print("  unique sequences:", df["sequence"].nunique())


if __name__ == "__main__":
    main()
