#!/usr/bin/env python3
"""Rank the bundled APEX table with v3 heuristic filters and MIC summaries."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ampjepa_hybrid_v3 import candidate_score, clean_sequence

GN_COLUMNS = [
    "A. baumannii ATCC 19606",
    "E. coli ATCC 11775",
    "E. coli AIC221",
    "E. coli AIC222",
    "K. pneumoniae ATCC 13883",
    "P. aeruginosa PA01",
    "P. aeruginosa PA14",
]
GP_COLUMNS = [
    "S. aureus ATCC 12600",
    "S. aureus (ATCC BAA-1556) - MRSA",
    "vancomycin-resistant E. faecalis ATCC 700802",
    "vancomycin-resistant E. faecium ATCC 700221",
]
ALL_PATHOGEN_COLUMNS = GN_COLUMNS + GP_COLUMNS


def add_mic_summaries(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    present = [c for c in ALL_PATHOGEN_COLUMNS if c in df.columns]
    gn = [c for c in GN_COLUMNS if c in df.columns]
    gp = [c for c in GP_COLUMNS if c in df.columns]

    for c in present:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    if present:
        df["min_pred_MIC"] = df[present].min(axis=1)
        df["median_pred_MIC"] = df[present].median(axis=1)
        df["mean_all_pred_MIC"] = df[present].mean(axis=1)
        df["n_pathogens_pred_MIC_le_32"] = (df[present] <= 32).sum(axis=1)
        df["n_pathogens_pred_MIC_le_64"] = (df[present] <= 64).sum(axis=1)
        df["n_pathogens_pred_MIC_le_128"] = (df[present] <= 128).sum(axis=1)
    if gn:
        df["min_GN_pred_MIC"] = df[gn].min(axis=1)
        df["median_GN_pred_MIC"] = df[gn].median(axis=1)
        df["mean_GN_pred_MIC"] = df[gn].mean(axis=1)
    if gp:
        df["min_GP_pred_MIC"] = df[gp].min(axis=1)
        df["median_GP_pred_MIC"] = df[gp].median(axis=1)
        df["mean_GP_pred_MIC"] = df[gp].mean(axis=1)
    return df


def main() -> None:
    src = Path("v3/data/external/apex_oracle_ranked_summary.csv")
    out = Path("v3/results/apex_table_v3_filter_rank.csv")
    df = add_mic_summaries(pd.read_csv(src))
    rows = []
    for _, row in df.iterrows():
        seq = clean_sequence(row["PeptideSequence"])
        rows.append({**row.to_dict(), "sequence": seq, **candidate_score(seq)})
    ranked = pd.DataFrame(rows).sort_values(
        ["passes_v3_filters", "v3_rank_score", "min_pred_MIC", "median_pred_MIC"],
        ascending=[False, False, True, True],
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(out, index=False)
    show_cols = [
        "candidate_id",
        "sequence",
        "min_pred_MIC",
        "median_pred_MIC",
        "mean_GN_pred_MIC",
        "mean_all_pred_MIC",
        "n_pathogens_pred_MIC_le_64",
        "v3_rank_score",
        "passes_v3_filters",
    ]
    show_cols = [c for c in show_cols if c in ranked.columns]
    print(ranked[show_cols].head(20).to_string(index=False))
    print(f"[DONE] Wrote {out}")


if __name__ == "__main__":
    main()
