#!/usr/bin/env python3
"""Stage 1F: use an APEX/ApexOracle table as an external comparator.

This script ranks APEX-selected peptides by predicted MIC summaries. It does not
claim AMP-JEPA is better than APEX. It creates a clean comparator table that can
later be joined with AMP-JEPA embeddings, novelty scores, toxicity predictions,
or wet-lab outcomes.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd

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


def charge(seq: str) -> int:
    seq = str(seq).upper()
    return sum(seq.count(x) for x in "KR") - sum(seq.count(x) for x in "DE")


def hydrophobic_fraction(seq: str) -> float:
    seq = str(seq).upper()
    if not seq:
        return 0.0
    return sum(aa in set("AILMFWVY") for aa in seq) / len(seq)


def standardize_apex_table(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "PeptideSequence" not in out.columns:
        for candidate in ["sequence", "Sequence", "peptide_sequence"]:
            if candidate in out.columns:
                out = out.rename(columns={candidate: "PeptideSequence"})
                break
    if "PeptideSequence" not in out.columns:
        raise SystemExit("[ERROR] APEX table needs a PeptideSequence column")

    if "candidate_id" not in out.columns:
        id_col = "Unnamed: 0" if "Unnamed: 0" in out.columns else None
        out.insert(0, "candidate_id", out[id_col].astype(str) if id_col else [f"candidate_{i+1:04d}" for i in range(len(out))])

    numeric_cols = [c for c in GN_COLUMNS + GP_COLUMNS if c in out.columns]
    if numeric_cols:
        gn = [c for c in GN_COLUMNS if c in out.columns]
        gp = [c for c in GP_COLUMNS if c in out.columns]
        if gn and "mean_GN_pred_MIC" not in out.columns:
            out["mean_GN_pred_MIC"] = out[gn].mean(axis=1)
        if gp and "mean_GP_pred_MIC" not in out.columns:
            out["mean_GP_pred_MIC"] = out[gp].mean(axis=1)
        if "mean_all_pred_MIC" not in out.columns:
            out["mean_all_pred_MIC"] = out[numeric_cols].mean(axis=1)
        if "median_all_pred_MIC" not in out.columns:
            out["median_all_pred_MIC"] = out[numeric_cols].median(axis=1)
        for threshold in [32, 64, 128, 256]:
            col = f"n_pathogens_pred_MIC_le_{threshold}"
            if col not in out.columns:
                out[col] = (out[numeric_cols] <= threshold).sum(axis=1)

    if "length" not in out.columns:
        out["length"] = out["PeptideSequence"].astype(str).str.len()
    if "charge_KR_minus_DE" not in out.columns:
        out["charge_KR_minus_DE"] = out["PeptideSequence"].map(charge)
    if "hydrophobic_fraction" not in out.columns:
        out["hydrophobic_fraction"] = out["PeptideSequence"].map(hydrophobic_fraction)

    rank_col = "mean_GN_pred_MIC" if "mean_GN_pred_MIC" in out.columns else "mean_all_pred_MIC"
    out = out.sort_values([rank_col, "mean_all_pred_MIC"], ascending=True).reset_index(drop=True)
    out.insert(0, "apex_rank", range(1, len(out) + 1))
    out["external_comparator_note"] = "APEX predicted MIC; lower is better; not experimental ground truth"
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apex", default="data/external/apex_oracle_ranked_summary.csv")
    parser.add_argument("--output", default="results/stage1/apex_oracle_external_comparator.csv")
    args = parser.parse_args()

    apex_path = Path(args.apex)
    if not apex_path.exists():
        raise SystemExit(f"[ERROR] Missing APEX table: {apex_path}")
    df = pd.read_csv(apex_path)
    out = standardize_apex_table(df)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    display_cols = [
        "apex_rank",
        "candidate_id",
        "PeptideSequence",
        "length",
        "charge_KR_minus_DE",
        "hydrophobic_fraction",
        "mean_GN_pred_MIC",
        "mean_all_pred_MIC",
        "n_pathogens_pred_MIC_le_64",
        "n_pathogens_pred_MIC_le_128",
    ]
    display_cols = [c for c in display_cols if c in out.columns]
    print(out[display_cols].head(10).to_string(index=False))
    print(f"[DONE] Wrote {out_path}")


if __name__ == "__main__":
    main()
