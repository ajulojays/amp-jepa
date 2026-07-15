#!/usr/bin/env python3
"""Create a tiny demo FASTA from the bundled APEX table.

This is only for checking that the v3 pipeline wiring works. It is not meant for
real training.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def main() -> None:
    src = Path("v3/data/external/apex_oracle_ranked_summary.csv")
    out = Path("v3/data/raw/peptides.fasta")
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(src)
    with out.open("w") as f:
        for _, row in df.iterrows():
            f.write(f">{row['Unnamed: 0']}\n{row['PeptideSequence']}\n")
    print(f"[DONE] Wrote tiny demo FASTA: {out}")


if __name__ == "__main__":
    main()
