#!/usr/bin/env python3
"""Compare v3-ranked candidates against an APEX/ApexOracle comparator table.

This does not run APEX. It merges/scans known APEX oracle rows when present and
creates a clean summary so APEX remains an external comparator, not ground truth.

Important interpretation:
- APEX rows have predicted MIC summaries when the source table contains MIC columns.
- v3-only rows will have NaN MIC summaries until those generated candidates are
  scored by APEX or another external MIC oracle.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

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


def existing_numeric_columns(df: pd.DataFrame, columns: Iterable[str]) -> List[str]:
    out = []
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            out.append(col)
    return out


def add_mic_summaries(df: pd.DataFrame) -> pd.DataFrame:
    """Add min/median/mean MIC summaries when pathogen-level MIC columns exist."""
    df = df.copy()
    gn = existing_numeric_columns(df, GN_COLUMNS)
    gp = existing_numeric_columns(df, GP_COLUMNS)
    all_cols = existing_numeric_columns(df, ALL_PATHOGEN_COLUMNS)

    if all_cols:
        df["min_pred_MIC"] = df[all_cols].min(axis=1)
        df["median_pred_MIC"] = df[all_cols].median(axis=1)
        if "mean_all_pred_MIC" not in df.columns:
            df["mean_all_pred_MIC"] = df[all_cols].mean(axis=1)
        if "n_pathogens_pred_MIC_le_32" not in df.columns:
            df["n_pathogens_pred_MIC_le_32"] = (df[all_cols] <= 32).sum(axis=1)
        if "n_pathogens_pred_MIC_le_64" not in df.columns:
            df["n_pathogens_pred_MIC_le_64"] = (df[all_cols] <= 64).sum(axis=1)
        if "n_pathogens_pred_MIC_le_128" not in df.columns:
            df["n_pathogens_pred_MIC_le_128"] = (df[all_cols] <= 128).sum(axis=1)

    if gn:
        df["min_GN_pred_MIC"] = df[gn].min(axis=1)
        df["median_GN_pred_MIC"] = df[gn].median(axis=1)
        if "mean_GN_pred_MIC" not in df.columns:
            df["mean_GN_pred_MIC"] = df[gn].mean(axis=1)

    if gp:
        df["min_GP_pred_MIC"] = df[gp].min(axis=1)
        df["median_GP_pred_MIC"] = df[gp].median(axis=1)
        if "mean_GP_pred_MIC" not in df.columns:
            df["mean_GP_pred_MIC"] = df[gp].mean(axis=1)

    # Backward-compatible alias if an older table already had median_all_pred_MIC.
    if "median_pred_MIC" not in df.columns and "median_all_pred_MIC" in df.columns:
        df["median_pred_MIC"] = pd.to_numeric(df["median_all_pred_MIC"], errors="coerce")
    if "median_all_pred_MIC" not in df.columns and "median_pred_MIC" in df.columns:
        df["median_all_pred_MIC"] = df["median_pred_MIC"]

    return df


def standardize_apex(df: pd.DataFrame) -> pd.DataFrame:
    if "PeptideSequence" not in df.columns:
        for c in ["sequence", "Sequence", "peptide_sequence"]:
            if c in df.columns:
                df = df.rename(columns={c: "PeptideSequence"})
                break
    if "PeptideSequence" not in df.columns:
        raise SystemExit("[ERROR] APEX table needs PeptideSequence or sequence column")
    if "candidate_id" not in df.columns:
        if "Unnamed: 0" in df.columns:
            df.insert(0, "candidate_id", df["Unnamed: 0"].astype(str))
        else:
            df.insert(0, "candidate_id", [f"apex_{i+1:04d}" for i in range(len(df))])
    df["sequence"] = df["PeptideSequence"].map(clean_sequence)
    df = add_mic_summaries(df)
    rank_col = "mean_GN_pred_MIC" if "mean_GN_pred_MIC" in df.columns else "mean_all_pred_MIC"
    if rank_col in df.columns:
        df = df.sort_values(rank_col).reset_index(drop=True)
        if "apex_rank" in df.columns:
            df = df.drop(columns=["apex_rank"])
        df.insert(0, "apex_rank", range(1, len(df) + 1))
    return df


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ranked", default="v3/results/ranked_candidates_v3.csv")
    p.add_argument("--apex", default="v3/data/external/apex_oracle_ranked_summary.csv")
    p.add_argument("--output", default="v3/results/apex_comparator_v3.csv")
    args = p.parse_args()

    ranked = pd.read_csv(args.ranked) if Path(args.ranked).exists() else pd.DataFrame()
    apex = standardize_apex(pd.read_csv(args.apex))

    if not ranked.empty and "sequence" in ranked.columns:
        ranked["sequence"] = ranked["sequence"].map(clean_sequence)
        merged = ranked.merge(
            apex[[c for c in apex.columns if c not in ranked.columns or c in {"sequence", "candidate_id"}]],
            on="sequence",
            how="outer",
            suffixes=("_v3", "_apex"),
        )
        merged["comparison_note"] = merged.apply(
            lambda r: "exact_sequence_overlap" if pd.notna(r.get("v3_rank")) and pd.notna(r.get("apex_rank")) else (
                "v3_only" if pd.notna(r.get("v3_rank")) else "apex_only"
            ),
            axis=1,
        )
    else:
        merged = apex.copy()
        merged["comparison_note"] = "apex_only_no_v3_ranked_file"

    # Add v3 heuristic score for APEX-only rows so they can be judged by the same design filters.
    for col in ["v3_rank_score", "passes_v3_filters", "novelty_score"]:
        if col not in merged.columns:
            merged[col] = pd.NA
    for idx, row in merged.iterrows():
        if pd.isna(row.get("v3_rank_score")) and isinstance(row.get("sequence"), str):
            scored = candidate_score(row["sequence"], max_train_identity=float(row.get("max_train_identity", 0.0) or 0.0))
            for key, value in scored.items():
                if key not in merged.columns:
                    merged[key] = pd.NA
                merged.at[idx, key] = value

    sort_cols = [c for c in ["comparison_note", "apex_rank", "v3_rank"] if c in merged.columns]
    if sort_cols:
        merged = merged.sort_values(sort_cols, na_position="last")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    show_cols = [
        c for c in [
            "comparison_note",
            "apex_rank",
            "v3_rank",
            "candidate_id",
            "sequence",
            "min_pred_MIC",
            "median_pred_MIC",
            "mean_GN_pred_MIC",
            "mean_all_pred_MIC",
            "n_pathogens_pred_MIC_le_64",
            "v3_rank_score",
            "passes_v3_filters",
        ] if c in merged.columns
    ]
    print(merged[show_cols].head(25).to_string(index=False))
    print(f"[DONE] Wrote {out_path}")


if __name__ == "__main__":
    main()
