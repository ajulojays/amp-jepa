#!/usr/bin/env python3
"""Count sequences in the prepared v3 corpus."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def main() -> None:
    path = Path("v3/data/processed/peptide_corpus_v3.csv")
    if not path.exists():
        raise SystemExit(f"[ERROR] Missing {path}")
    df = pd.read_csv(path)
    print(f"n_sequences={len(df):,}")
    if "length" in df.columns:
        print(f"length_min={df['length'].min()}")
        print(f"length_median={df['length'].median()}")
        print(f"length_max={df['length'].max()}")


if __name__ == "__main__":
    main()
