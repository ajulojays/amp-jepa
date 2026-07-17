#!/usr/bin/env python3
"""Assign stable Layer 1 novelty/IP portfolio labels to V4B Tier 1-4 candidates.

This script preserves the complete Tier 1-4 priority universe and adds a simple
portfolio label based on lead tier plus Layer 1 curated/training 75% identity
status.

Labels:
    Novel Tier 1
    Novel Tier 2
    Novel Tier 3
    Novel Tier 4
    Known-like Tier 1
    Known-like Tier 2
    Known-like Tier 3
    Known-like Tier 4
    Short follow-up

Input is expected to be the annotated output from the manifest75 merge step:
    v4b/results/novelty_layer1_manifest75_tiered/
        v4b_tier1_4_with_curated_qc_manifest75_flags.csv
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


TIER_MAP = {
    "Tier1_core_frontier": "Tier 1",
    "Tier2_balanced_elite": "Tier 2",
    "Tier3_robust_potency_reserve": "Tier 3",
    "Tier4_exploration_reserve": "Tier 4",
}

NOVEL_CLASS = "passes_broad_and_qc_75"
KNOWN_LIKE_CLASSES = {
    "removed_broad_curated_ge75",
    "removed_qc_core_ge75",
}
SHORT_FOLLOWUP_CLASS = "needs_short_peptide_followup"


def safe_filename(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unknown"


def choose_sequence_column(df: pd.DataFrame) -> str:
    for col in ["sequence_clean", "sequence", "Sequence", "seq", "peptide_sequence"]:
        if col in df.columns:
            return col
    raise ValueError(f"No sequence column found. Columns: {list(df.columns)}")


def write_fasta(df: pd.DataFrame, path: Path, id_col: str, seq_col: str) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for i, row in df.iterrows():
            rid = str(row.get(id_col, f"seq_{i + 1}")).strip() or f"seq_{i + 1}"
            rid = re.sub(r"[^A-Za-z0-9_.:-]+", "_", rid)[:180]
            seq = str(row.get(seq_col, "")).upper().replace(" ", "")
            seq = re.sub(r"\s+", "", seq)
            if seq:
                handle.write(f">{rid}\n{seq}\n")


def portfolio_label(row: pd.Series) -> str:
    tier = TIER_MAP.get(str(row.get("lead_tier", "")), "Unknown Tier")
    cls = str(row.get("layer1_manifest75_novelty_class", ""))

    if cls == NOVEL_CLASS:
        return f"Novel {tier}"
    if cls in KNOWN_LIKE_CLASSES:
        return f"Known-like {tier}"
    if cls == SHORT_FOLLOWUP_CLASS:
        return "Short follow-up"
    return f"Review {tier}"


def portfolio_group(row: pd.Series) -> str:
    label = str(row.get("layer1_portfolio_label", ""))
    if label.startswith("Novel"):
        return "Novel"
    if label.startswith("Known-like"):
        return "Known-like"
    if label == "Short follow-up":
        return "Short follow-up"
    return "Review"


def sort_for_review(df: pd.DataFrame) -> pd.DataFrame:
    sort_cols: list[str] = []
    ascending: list[bool] = []
    for col, asc in [
        ("lead_tier", True),
        ("v4b_elite_composite_score", False),
        ("ampjepa_master_score", False),
        ("APEX_median_MIC", True),
        ("APEX_worst_MIC", True),
        ("organisms_MIC_le_64", False),
    ]:
        if col in df.columns:
            sort_cols.append(col)
            ascending.append(asc)
    if sort_cols:
        df = df.sort_values(sort_cols, ascending=ascending, na_position="last")
    return df.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="v4b/results/novelty_layer1_manifest75_tiered/v4b_tier1_4_with_curated_qc_manifest75_flags.csv",
        help="Tier 1-4 CSV already annotated with manifest75 novelty flags.",
    )
    parser.add_argument(
        "--output-dir",
        default="v4b/results/novelty_layer1_portfolio_labels",
        help="Output directory for labeled CSV/FASTA files and summaries.",
    )
    parser.add_argument("--write-fasta", action="store_true", help="Also write FASTA files per portfolio label.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path, low_memory=False)
    if "lead_tier" not in df.columns:
        raise ValueError("Input must contain lead_tier column.")
    if "layer1_manifest75_novelty_class" not in df.columns:
        raise ValueError("Input must contain layer1_manifest75_novelty_class column.")

    seq_col = choose_sequence_column(df)
    if seq_col != "sequence_clean":
        df["sequence_clean"] = df[seq_col].astype(str).str.upper().str.replace(r"\s+", "", regex=True)
        seq_col = "sequence_clean"

    df["layer1_portfolio_label"] = df.apply(portfolio_label, axis=1)
    df["layer1_portfolio_group"] = df.apply(portfolio_group, axis=1)
    df["layer1_portfolio_rank"] = range(1, len(df) + 1)

    df = sort_for_review(df)
    df["layer1_portfolio_rank"] = df.groupby("layer1_portfolio_label").cumcount() + 1
    df["layer1_global_review_rank"] = range(1, len(df) + 1)

    all_out = output_dir / "v4b_tier1_4_layer1_portfolio_labeled.csv"
    df.to_csv(all_out, index=False)

    # Per-label CSV exports preserve all candidates in each portfolio class.
    per_label_dir = output_dir / "portfolio_label_files"
    per_label_dir.mkdir(exist_ok=True)
    seq_id_col = "candidate_id" if "candidate_id" in df.columns else df.columns[0]

    label_counts = []
    for label, group in df.groupby("layer1_portfolio_label", sort=False):
        safe = safe_filename(label)
        csv_path = per_label_dir / f"v4b_{safe}.csv"
        group.to_csv(csv_path, index=False)
        if args.write_fasta:
            write_fasta(group, per_label_dir / f"v4b_{safe}.fasta", seq_id_col, seq_col)
        label_counts.append({"layer1_portfolio_label": label, "n": int(len(group)), "csv": str(csv_path)})

    summary_label = df.groupby("layer1_portfolio_label").agg(
        n=("candidate_id", "count") if "candidate_id" in df.columns else (seq_col, "count"),
        unique_sequences=(seq_col, "nunique"),
        best_median_MIC=("APEX_median_MIC", "min") if "APEX_median_MIC" in df.columns else (seq_col, "count"),
        median_MIC=("APEX_median_MIC", "median") if "APEX_median_MIC" in df.columns else (seq_col, "count"),
        best_worst_MIC=("APEX_worst_MIC", "min") if "APEX_worst_MIC" in df.columns else (seq_col, "count"),
        median_worst_MIC=("APEX_worst_MIC", "median") if "APEX_worst_MIC" in df.columns else (seq_col, "count"),
        mean_hydro=("criteria_hydrophobic_fraction", "mean") if "criteria_hydrophobic_fraction" in df.columns else (seq_col, "count"),
        mean_charge=("criteria_charge", "mean") if "criteria_charge" in df.columns else (seq_col, "count"),
        mean_length=("criteria_length", "mean") if "criteria_length" in df.columns else (seq_col, "count"),
    ).reset_index()

    # Remove metric columns that were only placeholders if score columns were absent.
    summary_label.to_csv(output_dir / "v4b_layer1_portfolio_summary_by_label.csv", index=False)

    summary_group = df.groupby("layer1_portfolio_group").agg(
        n=("candidate_id", "count") if "candidate_id" in df.columns else (seq_col, "count"),
        unique_sequences=(seq_col, "nunique"),
    ).reset_index()
    summary_group.to_csv(output_dir / "v4b_layer1_portfolio_summary_by_group.csv", index=False)

    cross_tier = pd.crosstab(df["lead_tier"], df["layer1_portfolio_label"])
    cross_tier.to_csv(output_dir / "v4b_layer1_portfolio_crosstab_tier_by_label.csv")

    cross_novelty = pd.crosstab(df["lead_tier"], df["layer1_manifest75_novelty_class"])
    cross_novelty.to_csv(output_dir / "v4b_layer1_portfolio_crosstab_tier_by_novelty_class.csv")

    manifest = {
        "input": str(input_path),
        "output_dir": str(output_dir),
        "total_candidates": int(len(df)),
        "unique_sequences": int(df[seq_col].nunique()),
        "portfolio_label_counts": df["layer1_portfolio_label"].value_counts().to_dict(),
        "portfolio_group_counts": df["layer1_portfolio_group"].value_counts().to_dict(),
        "outputs": {
            "all_labeled": str(all_out),
            "summary_by_label": str(output_dir / "v4b_layer1_portfolio_summary_by_label.csv"),
            "summary_by_group": str(output_dir / "v4b_layer1_portfolio_summary_by_group.csv"),
            "per_label_dir": str(per_label_dir),
        },
    }
    (output_dir / "v4b_layer1_portfolio_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\nLAYER 1 PORTFOLIO LABEL COUNTS")
    print(df["layer1_portfolio_label"].value_counts().to_string())

    print("\nLAYER 1 PORTFOLIO GROUP COUNTS")
    print(df["layer1_portfolio_group"].value_counts().to_string())

    print("\nSUMMARY BY LABEL")
    print(summary_label.round(3).to_string(index=False))

    print(f"\nSaved: {all_out}")


if __name__ == "__main__":
    main()
