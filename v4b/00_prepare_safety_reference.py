#!/usr/bin/env python3
"""Validate and harmonize safety-labeled peptides for the V4B manifold pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

AA = set("ACDEFGHIKLMNPQRSTVWY")


def clean_sequence(value: object) -> str:
    return "".join(x for x in str(value).upper().strip() if x in AA)


def normalize_binary(series: pd.Series) -> pd.Series:
    mapping = {
        "0": 0.0,
        "1": 1.0,
        "false": 0.0,
        "true": 1.0,
        "negative": 0.0,
        "positive": 1.0,
        "nonhemolytic": 0.0,
        "hemolytic": 1.0,
        "non-hemolytic": 0.0,
        "noncytotoxic": 0.0,
        "cytotoxic": 1.0,
        "non-cytotoxic": 0.0,
        "low": 0.0,
        "high": 1.0,
    }
    text = series.astype(str).str.strip().str.lower()
    out = text.map(mapping)
    numeric = pd.to_numeric(series, errors="coerce")
    out = out.where(out.notna(), numeric)
    return out.where(out.isin([0, 1]), np.nan)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="CSV with sequence and safety labels")
    ap.add_argument("--output", default="v4b/results/safety_manifold_pilot/safety_reference_clean.csv")
    ap.add_argument("--summary", default="v4b/results/safety_manifold_pilot/safety_reference_summary.json")
    ap.add_argument("--sequence-column", default="sequence")
    ap.add_argument("--hemolysis-column", default="hemolysis_label")
    ap.add_argument("--cytotoxicity-column", default="cytotoxicity_label")
    ap.add_argument("--min-len", type=int, default=5)
    ap.add_argument("--max-len", type=int, default=64)
    args = ap.parse_args()

    src = Path(args.input)
    if not src.exists():
        raise SystemExit(f"Input not found: {src}")

    df = pd.read_csv(src, low_memory=False)
    if args.sequence_column not in df.columns:
        raise SystemExit(f"Missing sequence column: {args.sequence_column}")

    df = df.copy()
    df["sequence"] = df[args.sequence_column].map(clean_sequence)
    df = df[df["sequence"].map(lambda s: args.min_len <= len(s) <= args.max_len)].copy()
    df = df[df["sequence"].map(lambda s: all(a in AA for a in s))].copy()

    if args.hemolysis_column in df.columns:
        df["hemolysis_label"] = normalize_binary(df[args.hemolysis_column])
    else:
        df["hemolysis_label"] = np.nan

    if args.cytotoxicity_column in df.columns:
        df["cytotoxicity_label"] = normalize_binary(df[args.cytotoxicity_column])
    else:
        df["cytotoxicity_label"] = np.nan

    # Collapse exact duplicate sequences conservatively. Conflicting labels become unknown.
    rows = []
    for sequence, group in df.groupby("sequence", sort=False):
        row = group.iloc[0].copy()
        for endpoint in ["hemolysis_label", "cytotoxicity_label"]:
            labels = sorted(group[endpoint].dropna().unique().tolist())
            row[endpoint] = labels[0] if len(labels) == 1 else np.nan
        rows.append(row)
    clean = pd.DataFrame(rows).reset_index(drop=True)

    output = Path(args.output)
    summary_path = Path(args.summary)
    output.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(output, index=False)

    summary = {
        "input_rows": int(len(df)),
        "unique_sequences": int(len(clean)),
        "hemolysis_labeled": int(clean["hemolysis_label"].notna().sum()),
        "hemolysis_positive": int((clean["hemolysis_label"] == 1).sum()),
        "hemolysis_negative": int((clean["hemolysis_label"] == 0).sum()),
        "cytotoxicity_labeled": int(clean["cytotoxicity_label"].notna().sum()),
        "cytotoxicity_positive": int((clean["cytotoxicity_label"] == 1).sum()),
        "cytotoxicity_negative": int((clean["cytotoxicity_label"] == 0).sum()),
        "output": str(output),
        "missing_labels_are_safe": False,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
