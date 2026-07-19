#!/usr/bin/env python3
"""Merge broad/QC-core novelty flags and select the dual-novel MIC32 pool.

This stage preserves the frozen V4B decision logic:
- known-like if >=75% identity to the broad curated corpus;
- otherwise known-like if >=75% identity to the QC core;
- short-peptide follow-up if either CD-HIT screen did not process the sequence;
- novelty pass only when both screens report below-threshold;
- potency pass when predicted APEX median MIC <= 32 µM.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--master", default="v4c/results/final_funnel/00_merged_qc/v4c_million_scored_qc_pass.csv")
    p.add_argument("--broad-flags", default="v4c/results/final_funnel/01_novelty_broad_curated/v4c_1m_manifest75_flags.csv")
    p.add_argument("--qc-flags", default="v4c/results/final_funnel/02_novelty_qc_core/v4c_1m_manifest75_flags.csv")
    p.add_argument("--output-dir", default="v4c/results/final_funnel/03_dual_novelty_mic32")
    p.add_argument("--mic-cutoff", type=float, default=32.0)
    p.add_argument("--expected-total", type=int, default=1000000)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def atomic_csv(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def load_flags(path: Path, prefix: str) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0)
    required = {"candidate_id", "manifest75_status"}
    missing = sorted(required - set(header.columns))
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    cols = ["candidate_id", "manifest75_status"]
    for c in ["manifest_match_reference_id", "cdhit_reported_identity", "sequence_length", "generation_source"]:
        if c in header.columns:
            cols.append(c)
    df = pd.read_csv(path, usecols=cols, low_memory=False)
    df["candidate_id"] = df["candidate_id"].astype(str)
    if df["candidate_id"].duplicated().any():
        raise ValueError(f"Duplicate candidate IDs in {path}")
    rename = {c: f"{prefix}_{c}" for c in df.columns if c != "candidate_id"}
    return df.rename(columns=rename)


def classify(row: pd.Series) -> str:
    broad = row["broad_manifest75_status"]
    qc = row["qc_manifest75_status"]
    if broad == "removed_manifest_ge_threshold":
        return "removed_broad_curated_ge75"
    if qc == "removed_manifest_ge_threshold":
        return "removed_qc_core_ge75"
    if broad == "unprocessed_by_cdhit" or qc == "unprocessed_by_cdhit":
        return "needs_short_peptide_followup"
    if broad == "kept_below_threshold" and qc == "kept_below_threshold":
        return "passes_broad_and_qc_75"
    return "review_unresolved_novelty_status"


def main() -> None:
    args = parse_args()
    master_path = Path(args.master)
    broad_path = Path(args.broad_flags)
    qc_path = Path(args.qc_flags)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    compact_path = outdir / "v4c_dual_novelty_compact_flags.csv"
    novel_all_path = outdir / "v4c_dual_novel_all_mic.csv"
    mic32_path = outdir / "v4c_dual_novel_MIC32_ranked.csv"
    class_summary_path = outdir / "v4c_dual_novelty_class_summary.csv"
    generation_summary_path = outdir / "v4c_dual_novelty_by_generation.csv"
    summary_path = outdir / "v4c_dual_novelty_MIC32_summary.json"
    outputs = [compact_path, novel_all_path, mic32_path, class_summary_path, generation_summary_path, summary_path]
    existing = [p for p in outputs if p.exists()]
    if existing and not args.overwrite:
        raise FileExistsError("Outputs already exist; use --overwrite after review:\n" + "\n".join(map(str, existing)))

    for path in [master_path, broad_path, qc_path]:
        if not path.exists() or path.stat().st_size == 0:
            raise FileNotFoundError(path)

    print("[V4C-FUNNEL] Loading dual-reference novelty flags")
    broad = load_flags(broad_path, "broad")
    qc = load_flags(qc_path, "qc")
    flags = broad.merge(qc, on="candidate_id", how="outer", validate="one_to_one", indicator=True)
    if not flags["_merge"].eq("both").all():
        raise ValueError("Broad and QC-core flag candidate sets are not identical.")
    flags = flags.drop(columns="_merge")
    if args.expected_total > 0 and len(flags) != args.expected_total:
        raise ValueError(f"Flag count {len(flags):,} != expected {args.expected_total:,}")

    flags["layer1_manifest75_novelty_class"] = flags.apply(classify, axis=1)
    unresolved = flags["layer1_manifest75_novelty_class"].eq("review_unresolved_novelty_status")
    if unresolved.any():
        raise RuntimeError(f"Unresolved novelty status for {unresolved.sum():,} candidates")

    print("[V4C-FUNNEL] Loading million-peptide APEX master table")
    master = pd.read_csv(master_path, low_memory=False)
    master["candidate_id"] = master["candidate_id"].astype(str)
    if master["candidate_id"].duplicated().any():
        raise ValueError("Duplicate candidate IDs in master table")
    if args.expected_total > 0 and len(master) != args.expected_total:
        raise ValueError(f"Master count {len(master):,} != expected {args.expected_total:,}")

    merged = master.merge(flags, on="candidate_id", how="left", validate="one_to_one")
    if merged["layer1_manifest75_novelty_class"].isna().any():
        raise ValueError("Missing novelty flags after master merge")

    merged["APEX_median_MIC"] = pd.to_numeric(merged["APEX_median_MIC"], errors="coerce")
    merged["APEX_worst_MIC"] = pd.to_numeric(merged["APEX_worst_MIC"], errors="coerce")
    merged["APEX_mean_MIC"] = pd.to_numeric(merged["APEX_mean_MIC"], errors="coerce")
    if merged[["APEX_median_MIC", "APEX_worst_MIC", "APEX_mean_MIC"]].isna().any().any():
        raise ValueError("Missing/non-numeric APEX summary MIC values")

    merged["dual_novelty_pass"] = merged["layer1_manifest75_novelty_class"].eq("passes_broad_and_qc_75")
    merged["median_MIC_le_32"] = merged["APEX_median_MIC"].le(args.mic_cutoff)

    compact_cols = [c for c in [
        "candidate_id", "generation_source", "sequence_clean", "sequence",
        "broad_manifest75_status", "qc_manifest75_status",
        "layer1_manifest75_novelty_class", "dual_novelty_pass",
        "APEX_median_MIC", "APEX_mean_MIC", "APEX_worst_MIC",
        "organisms_MIC_le_64", "median_MIC_le_32",
    ] if c in merged.columns]
    compact = merged[compact_cols].copy()

    novel = merged[merged["dual_novelty_pass"]].copy()
    mic32 = novel[novel["median_MIC_le_32"]].copy()
    sort_cols = [c for c in ["APEX_median_MIC", "APEX_worst_MIC", "APEX_mean_MIC"] if c in mic32.columns]
    ascending = [True] * len(sort_cols)
    if "organisms_MIC_le_64" in mic32.columns:
        mic32["organisms_MIC_le_64"] = pd.to_numeric(mic32["organisms_MIC_le_64"], errors="coerce").fillna(0)
        sort_cols.append("organisms_MIC_le_64")
        ascending.append(False)
    mic32 = mic32.sort_values(sort_cols, ascending=ascending, na_position="last").reset_index(drop=True)
    mic32["v4c_pre_self_similarity_rank"] = np.arange(1, len(mic32) + 1)

    class_summary = (
        merged.groupby("layer1_manifest75_novelty_class")
        .agg(
            n=("candidate_id", "count"),
            unique_sequences=("sequence_clean", "nunique") if "sequence_clean" in merged.columns else ("candidate_id", "nunique"),
            median_APEX_median_MIC=("APEX_median_MIC", "median"),
            n_median_MIC_le_32=("median_MIC_le_32", "sum"),
        )
        .reset_index()
    )
    generation_summary = (
        merged.groupby(["generation_source", "layer1_manifest75_novelty_class"])
        .size().unstack(fill_value=0).reset_index()
        if "generation_source" in merged.columns else pd.DataFrame()
    )

    print(f"[V4C-FUNNEL] Writing compact flags: {len(compact):,}")
    atomic_csv(compact, compact_path)
    print(f"[V4C-FUNNEL] Writing dual-novel pool: {len(novel):,}")
    atomic_csv(novel, novel_all_path)
    print(f"[V4C-FUNNEL] Writing dual-novel MIC32 ranked pool: {len(mic32):,}")
    atomic_csv(mic32, mic32_path)
    atomic_csv(class_summary, class_summary_path)
    if not generation_summary.empty:
        atomic_csv(generation_summary, generation_summary_path)

    summary = {
        "schema_version": "1.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "AMP-JEPA-Hybrid V4C",
        "stage": "dual_reference_novelty_and_MIC32",
        "input_candidates": int(len(merged)),
        "novelty_class_counts": {str(k): int(v) for k, v in merged["layer1_manifest75_novelty_class"].value_counts().items()},
        "dual_novelty_pass": int(len(novel)),
        "mic_cutoff_uM": float(args.mic_cutoff),
        "dual_novel_and_median_MIC_le_32": int(len(mic32)),
        "best_predicted_median_MIC_uM": float(mic32["APEX_median_MIC"].min()) if len(mic32) else None,
        "outputs": {
            "compact_flags": str(compact_path),
            "dual_novel_all_mic": str(novel_all_path),
            "dual_novel_MIC32_ranked": str(mic32_path),
            "class_summary": str(class_summary_path),
            "generation_summary": str(generation_summary_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nV4C DUAL-NOVELTY + MIC32 SUMMARY")
    print(json.dumps(summary, indent=2))
    print("\nNOVELTY CLASS SUMMARY")
    print(class_summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
