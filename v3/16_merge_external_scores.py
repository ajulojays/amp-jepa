#!/usr/bin/env python3
"""Merge optional external predictor scores into a v3 ranked table.

The external CSV should contain a sequence column plus one or more score columns.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ampjepa_hybrid_v3 import clean_sequence


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ranked", default="v3/results/ranked_candidates_v3.csv")
    p.add_argument("--external", required=True)
    p.add_argument("--output", default="v3/results/ranked_candidates_v3_with_external_scores.csv")
    args = p.parse_args()

    ranked = pd.read_csv(args.ranked)
    ext = pd.read_csv(args.external)
    for df, name in [(ranked, "ranked"), (ext, "external")]:
        if "sequence" not in df.columns:
            alt = next((c for c in ["Sequence", "PeptideSequence", "peptide_sequence"] if c in df.columns), None)
            if alt is None:
                raise SystemExit(f"[ERROR] {name} table needs a sequence column")
            df.rename(columns={alt: "sequence"}, inplace=True)
        df["sequence"] = df["sequence"].map(clean_sequence)

    out = ranked.merge(ext, on="sequence", how="left", suffixes=("", "_external"))
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    print(f"[DONE] Wrote {path}")


if __name__ == "__main__":
    main()
