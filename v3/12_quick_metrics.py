#!/usr/bin/env python3
"""Quick metrics for v3 candidate tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="v3/results/ranked_candidates_v3.csv")
    args = p.parse_args()
    path = Path(args.input)
    if not path.exists():
        raise SystemExit(f"[ERROR] Missing {path}")
    df = pd.read_csv(path)
    print(f"rows={len(df):,}")
    if "passes_v3_filters" in df.columns:
        print(f"passes_v3_filters={int(df['passes_v3_filters'].astype(float).sum()):,}")
    for col in ["length", "net_charge_KR_minus_DE", "hydrophobic_fraction", "novelty_score", "v3_rank_score"]:
        if col in df.columns:
            print(f"{col}: mean={df[col].astype(float).mean():.3f}, min={df[col].astype(float).min():.3f}, max={df[col].astype(float).max():.3f}")


if __name__ == "__main__":
    main()
