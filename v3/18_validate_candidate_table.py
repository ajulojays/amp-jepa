#!/usr/bin/env python3
"""Validate that a v3 candidate table has the expected columns."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REQUIRED = {"sequence"}
RECOMMENDED = {"v3_rank_score", "novelty_score", "passes_v3_filters", "hydrophobic_fraction", "net_charge_KR_minus_DE"}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="v3/results/ranked_candidates_v3.csv")
    args = p.parse_args()
    path = Path(args.input)
    if not path.exists():
        raise SystemExit(f"[ERROR] Missing {path}")
    df = pd.read_csv(path)
    missing_required = REQUIRED - set(df.columns)
    missing_recommended = RECOMMENDED - set(df.columns)
    if missing_required:
        raise SystemExit(f"[ERROR] Missing required columns: {sorted(missing_required)}")
    if missing_recommended:
        print(f"[WARN] Missing recommended columns: {sorted(missing_recommended)}")
    print(f"[OK] {path} rows={len(df):,} columns={len(df.columns):,}")


if __name__ == "__main__":
    main()
