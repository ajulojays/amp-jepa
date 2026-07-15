#!/usr/bin/env python3
"""Compare v3-ranked candidates against an APEX/ApexOracle comparator table.

This does not run APEX. It merges/scans known APEX oracle rows when present and
creates a clean summary so APEX remains an external comparator, not ground truth.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ampjepa_hybrid_v3 import candidate_score, clean_sequence


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
    rank_col = "mean_GN_pred_MIC" if "mean_GN_pred_MIC" in df.columns else "mean_all_pred_MIC"
    if rank_col in df.columns:
        df = df.sort_values(rank_col).reset_index(drop=True)
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
    show_cols = [c for c in ["comparison_note", "apex_rank", "v3_rank", "candidate_id", "sequence", "mean_GN_pred_MIC", "mean_all_pred_MIC", "v3_rank_score", "passes_v3_filters"] if c in merged.columns]
    print(merged[show_cols].head(25).to_string(index=False))
    print(f"[DONE] Wrote {out_path}")


if __name__ == "__main__":
    main()
