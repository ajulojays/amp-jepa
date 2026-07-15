#!/usr/bin/env python3
"""Export top v3 candidates to FASTA."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="v3/results/top_panel_v3.csv")
    p.add_argument("--output", default="v3/results/top_panel_v3.fasta")
    args = p.parse_args()

    df = pd.read_csv(args.input)
    id_col = "candidate_id" if "candidate_id" in df.columns else df.columns[0]
    if "sequence" not in df.columns:
        raise SystemExit("[ERROR] input needs sequence column")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for _, row in df.iterrows():
            f.write(f">{row[id_col]}\n{row['sequence']}\n")
    print(f"[DONE] Wrote {out}")


if __name__ == "__main__":
    main()
