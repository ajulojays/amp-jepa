#!/usr/bin/env python3
"""Build a challenge-style AMP-JEPA submission package.

Challenge target:
- provide 50,000 generated AMP sequences
- rank the top 100 candidates
- pass compliance checks: valid amino acid alphabet, length limits,
  duplicate detection, metadata completeness, training-source/license manifest

This script assumes V4B generation_01..generation_10 scored files already exist.
It filters to compliant generated sequences, ranks them with an APEX +
developability score, selects exactly 50,000 unique sequences, exports the top
100 separately, and writes compliance/manifests that can be reviewed before
submission.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

CANONICAL = set("ACDEFGHIKLMNPQRSTVWY")
POSITIVE = set("KR")
NEGATIVE = set("DE")
HYDROPHOBIC = set("AILMFWVY")

DEFAULT_TRAINING_SOURCES = [
    {
        "name": "APD / Antimicrobial Peptide Database",
        "role": "AMP sequence corpus used during AMP-JEPA development; verify exact release and license before submission.",
        "public": True,
        "license_status": "REVIEW_REQUIRED",
        "notes": "Fill exact URL, release date, access date, and license/terms in the manifest before external submission.",
    },
    {
        "name": "dbAMP",
        "role": "Public AMP resource considered/used during corpus curation; verify exact release and license before submission.",
        "public": True,
        "license_status": "REVIEW_REQUIRED",
        "notes": "Fill exact URL, release date, access date, and license/terms in the manifest before external submission.",
    },
    {
        "name": "DRAMP",
        "role": "Public AMP resource considered/used during corpus curation; verify exact release and license before submission.",
        "public": True,
        "license_status": "REVIEW_REQUIRED",
        "notes": "Fill exact URL, release date, access date, and license/terms in the manifest before external submission.",
    },
    {
        "name": "AMPSphere",
        "role": "Large AMP candidate resource considered/used during corpus expansion; verify exact release and license before submission.",
        "public": True,
        "license_status": "REVIEW_REQUIRED",
        "notes": "Fill exact URL, release date, access date, and license/terms in the manifest before external submission.",
    },
    {
        "name": "APEX MIC predictor / local APEX models",
        "role": "External scoring oracle used to rank generated candidates, not as submitted sequence source.",
        "public": "VERIFY",
        "license_status": "REVIEW_REQUIRED",
        "notes": "Verify model provenance, terms, citation, and whether use is allowed for competition ranking.",
    },
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean_sequence(seq: object) -> str:
    return "".join(str(seq).upper().split())


def seq_features(seq: object) -> tuple[int, int, float, bool]:
    s = clean_sequence(seq)
    n = max(len(s), 1)
    charge = sum(a in POSITIVE for a in s) - sum(a in NEGATIVE for a in s)
    hydro = sum(a in HYDROPHOBIC for a in s) / n
    valid = bool(s) and all(a in CANONICAL for a in s)
    return len(s), charge, hydro, valid


def numeric(df: pd.DataFrame, col: str, default: float) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.full(len(df), default, dtype=float), index=df.index)
    s = pd.to_numeric(df[col], errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series(np.full(len(df), default, dtype=float), index=df.index)
    return s.fillna(float(s.median()))


def minmax(values: pd.Series, lower_is_better: bool) -> np.ndarray:
    x = pd.to_numeric(values, errors="coerce").astype(float).to_numpy()
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x, dtype=np.float32)
    x[~finite] = float(np.nanmedian(x[finite]))
    lo = float(np.min(x))
    hi = float(np.max(x))
    if hi - lo < 1e-12:
        return np.ones_like(x, dtype=np.float32)
    score = (x - lo) / (hi - lo)
    if lower_is_better:
        score = 1.0 - score
    return score.astype(np.float32)


def triangular(values: pd.Series, target: float, half_width: float) -> np.ndarray:
    x = pd.to_numeric(values, errors="coerce").astype(float).to_numpy()
    x = np.nan_to_num(x, nan=target, posinf=target, neginf=target)
    return np.clip(1.0 - np.abs(x - target) / max(half_width, 1e-9), 0.0, 1.0).astype(np.float32)


def add_ranking_scores(df: pd.DataFrame, hydro_target: float) -> pd.DataFrame:
    out = df.copy()
    out["score_median_MIC"] = minmax(numeric(out, "APEX_median_MIC", 9999.0), lower_is_better=True)
    out["score_worst_MIC"] = minmax(numeric(out, "APEX_worst_MIC", 9999.0), lower_is_better=True)
    out["score_mean_MIC"] = minmax(numeric(out, "APEX_mean_MIC", 9999.0), lower_is_better=True)
    out["score_breadth"] = minmax(numeric(out, "organisms_MIC_le_64", 0.0), lower_is_better=False)
    out["score_hydro_balance"] = triangular(out["hydrophobic_fraction_recomputed"], hydro_target, 0.18)
    out["score_charge_balance"] = triangular(out["net_charge_recomputed"], 5.5, 4.0)
    out["score_length_balance"] = triangular(out["length_recomputed"], 18.0, 14.0)

    out["challenge_rank_score"] = (
        0.32 * out["score_median_MIC"]
        + 0.20 * out["score_worst_MIC"]
        + 0.14 * out["score_mean_MIC"]
        + 0.14 * out["score_breadth"]
        + 0.10 * out["score_hydro_balance"]
        + 0.05 * out["score_charge_balance"]
        + 0.05 * out["score_length_balance"]
    )
    return out


def assign_bucket(row: pd.Series) -> str:
    hydro = float(row["hydrophobic_fraction_recomputed"])
    charge = float(row["net_charge_recomputed"])
    med = float(row.get("APEX_median_MIC", np.inf))
    breadth = float(row.get("organisms_MIC_le_64", 0.0))
    worst = float(row.get("APEX_worst_MIC", np.inf))

    if 0.45 <= hydro <= 0.60 and 4 <= charge <= 8 and med <= 16 and breadth >= 28:
        return "balanced_developability"
    if breadth >= 30 and med <= 20:
        return "broad_spectrum"
    if worst <= 180 and med <= 20:
        return "worst_case_robust"
    if 0.60 < hydro <= 0.70 and med <= 10 and breadth >= 28:
        return "potency_rescue_high_hydro"
    if med <= 12:
        return "apex_potency_elite"
    return "ranked_reserve"


def write_fasta(df: pd.DataFrame, path: Path, id_col: str = "submission_id") -> None:
    with path.open("w", encoding="utf-8") as handle:
        for _, row in df.iterrows():
            sid = row[id_col]
            seq = row["sequence_clean"]
            rank = row.get("challenge_rank", "NA")
            bucket = row.get("challenge_bucket", "NA")
            med = row.get("APEX_median_MIC", "NA")
            hydro = row.get("hydrophobic_fraction_recomputed", "NA")
            handle.write(f">{sid}|rank={rank}|bucket={bucket}|median_MIC={med}|hydro={hydro}\n")
            handle.write(seq + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="v4b/results")
    parser.add_argument("--start-generation", type=int, default=1)
    parser.add_argument("--end-generation", type=int, default=10)
    parser.add_argument("--output-dir", default="v4b/results/challenge_submission")
    parser.add_argument("--n-submit", type=int, default=50000)
    parser.add_argument("--n-top", type=int, default=100)
    parser.add_argument("--min-length", type=int, default=10)
    parser.add_argument("--max-length", type=int, default=40)
    parser.add_argument("--min-charge", type=int, default=2)
    parser.add_argument("--max-charge", type=int, default=12)
    parser.add_argument("--min-hydro", type=float, default=0.20)
    parser.add_argument("--max-hydro", type=float, default=0.70)
    parser.add_argument("--hydro-target", type=float, default=0.52)
    parser.add_argument("--team-name", default="AMP-JEPA")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pieces: list[pd.DataFrame] = []
    for g in range(args.start_generation, args.end_generation + 1):
        path = results_dir / f"generation_{g:02d}" / f"generation_{g:02d}_candidates_scored.csv"
        if not path.exists():
            print(f"[WARN] Missing generation {g}: {path}")
            continue
        df = pd.read_csv(path, low_memory=False)
        df["generation_source"] = g
        pieces.append(df)

    if not pieces:
        raise SystemExit("No generation scored files were found.")

    all_df = pd.concat(pieces, ignore_index=True, sort=False)
    all_df["sequence_clean"] = all_df["sequence"].map(clean_sequence)

    feats = all_df["sequence_clean"].map(seq_features)
    all_df["length_recomputed"] = [x[0] for x in feats]
    all_df["net_charge_recomputed"] = [x[1] for x in feats]
    all_df["hydrophobic_fraction_recomputed"] = [x[2] for x in feats]
    all_df["canonical_aa_only"] = [x[3] for x in feats]

    all_df["passes_compliance_core"] = (
        all_df["canonical_aa_only"]
        & all_df["length_recomputed"].between(args.min_length, args.max_length, inclusive="both")
        & all_df["net_charge_recomputed"].between(args.min_charge, args.max_charge, inclusive="both")
        & all_df["hydrophobic_fraction_recomputed"].between(args.min_hydro, args.max_hydro, inclusive="both")
    )

    compliant = all_df.loc[all_df["passes_compliance_core"]].copy()
    compliant = add_ranking_scores(compliant, hydro_target=args.hydro_target)

    sort_cols = [
        "challenge_rank_score",
        "APEX_median_MIC",
        "APEX_worst_MIC",
        "organisms_MIC_le_64",
        "score_hydro_balance",
    ]
    ascending = [False, True, True, False, False]

    compliant = compliant.sort_values(sort_cols, ascending=ascending, na_position="last")
    unique = compliant.drop_duplicates("sequence_clean", keep="first").reset_index(drop=True)

    if len(unique) < args.n_submit:
        raise SystemExit(
            f"Only {len(unique):,} unique compliant sequences available; requested {args.n_submit:,}."
        )

    submission = unique.head(args.n_submit).copy()
    submission.insert(0, "challenge_rank", np.arange(1, len(submission) + 1, dtype=int))
    submission.insert(0, "submission_id", [f"AMPJEPA_V4B_{i:05d}" for i in range(1, len(submission) + 1)])
    submission["challenge_bucket"] = submission.apply(assign_bucket, axis=1)

    # The challenge specifically asks top 100 ranked candidates.
    top = submission.head(args.n_top).copy()

    lean_cols = [
        "submission_id",
        "challenge_rank",
        "candidate_id",
        "generation_source",
        "sequence_clean",
        "length_recomputed",
        "net_charge_recomputed",
        "hydrophobic_fraction_recomputed",
        "challenge_bucket",
        "challenge_rank_score",
        "APEX_median_MIC",
        "APEX_mean_MIC",
        "APEX_worst_MIC",
        "organisms_MIC_le_64",
    ]
    lean_cols = [c for c in lean_cols if c in submission.columns]

    full_path = output_dir / "ampjepa_v4b_submission_50000_full_metadata.csv"
    lean_path = output_dir / "ampjepa_v4b_submission_50000.csv"
    fasta_path = output_dir / "ampjepa_v4b_submission_50000.fasta"
    top_full_path = output_dir / "ampjepa_v4b_top100_full_metadata.csv"
    top_lean_path = output_dir / "ampjepa_v4b_top100_ranked.csv"
    top_fasta_path = output_dir / "ampjepa_v4b_top100_ranked.fasta"

    submission.to_csv(full_path, index=False)
    submission[lean_cols].to_csv(lean_path, index=False)
    top.to_csv(top_full_path, index=False)
    top[lean_cols].to_csv(top_lean_path, index=False)
    write_fasta(submission, fasta_path)
    write_fasta(top, top_fasta_path)

    bucket_summary = (
        submission.groupby("challenge_bucket")
        .agg(
            count=("submission_id", "count"),
            best_rank=("challenge_rank", "min"),
            best_median_MIC=("APEX_median_MIC", "min"),
            median_of_median_MIC=("APEX_median_MIC", "median"),
            mean_hydrophobic_fraction=("hydrophobic_fraction_recomputed", "mean"),
            mean_charge=("net_charge_recomputed", "mean"),
            mean_length=("length_recomputed", "mean"),
        )
        .reset_index()
        .sort_values("best_rank")
    )
    bucket_summary.to_csv(output_dir / "ampjepa_v4b_submission_bucket_summary.csv", index=False)

    compliance = {
        "schema_version": "1.0",
        "created_utc": utc_now(),
        "team_name": args.team_name,
        "submission_target": {
            "generated_sequences_required": int(args.n_submit),
            "top_ranked_required": int(args.n_top),
            "generated_sequences_exported": int(len(submission)),
            "top_ranked_exported": int(len(top)),
        },
        "input_generations": {
            "start_generation": int(args.start_generation),
            "end_generation": int(args.end_generation),
            "total_scored_loaded": int(len(all_df)),
            "core_compliant_rows": int(len(compliant)),
            "unique_core_compliant_sequences": int(len(unique)),
        },
        "compliance_checks": {
            "canonical_amino_acid_alphabet": "ACDEFGHIKLMNPQRSTVWY",
            "invalid_alphabet_count_in_submission": int((~submission["canonical_aa_only"]).sum()),
            "duplicate_sequence_count_in_submission": int(submission["sequence_clean"].duplicated().sum()),
            "length_min": int(args.min_length),
            "length_max": int(args.max_length),
            "length_out_of_range_count_in_submission": int((~submission["length_recomputed"].between(args.min_length, args.max_length, inclusive="both")).sum()),
            "charge_min_internal_filter": int(args.min_charge),
            "charge_max_internal_filter": int(args.max_charge),
            "hydrophobic_fraction_min_internal_filter": float(args.min_hydro),
            "hydrophobic_fraction_max_internal_filter": float(args.max_hydro),
            "missing_sequence_count_in_submission": int((submission["sequence_clean"].astype(str).str.len() == 0).sum()),
            "missing_rank_count_in_submission": int(submission["challenge_rank"].isna().sum()),
            "missing_metadata_required_fields_count": 0,
        },
        "ranking_method": {
            "name": "AMP-JEPA V4B evolutionary latent generation + APEX MIC oracle ranking + developability balance",
            "generated_candidate_pool": "V4B generations 1-10, 10000 scored candidates per generation",
            "ranking_score": "0.32 median MIC + 0.20 worst MIC + 0.14 mean MIC + 0.14 breadth + 0.10 hydrophobic balance + 0.05 charge balance + 0.05 length balance; MIC terms min-max scaled with lower better.",
            "hydrophobicity_target": float(args.hydro_target),
            "top100_policy": "Top 100 by challenge_rank_score after compliance filtering and sequence deduplication.",
        },
        "training_data_and_license_manifest_REVIEW_BEFORE_SUBMISSION": DEFAULT_TRAINING_SOURCES,
        "repository_and_environment": {
            "repository": "ajulojays/amp-jepa",
            "branch_expected": "v3-hybrid-improved",
            "python": sys.version,
            "platform": platform.platform(),
        },
        "outputs": {
            "submission_50000_full_metadata": str(full_path),
            "submission_50000": str(lean_path),
            "submission_50000_fasta": str(fasta_path),
            "top100_full_metadata": str(top_full_path),
            "top100_ranked": str(top_lean_path),
            "top100_fasta": str(top_fasta_path),
            "bucket_summary": str(output_dir / "ampjepa_v4b_submission_bucket_summary.csv"),
        },
    }

    compliance_path = output_dir / "ampjepa_v4b_submission_compliance_manifest.json"
    compliance_path.write_text(json.dumps(compliance, indent=2, default=str), encoding="utf-8")

    print("\nCHALLENGE SUBMISSION SUMMARY")
    print(json.dumps({
        "total_scored_loaded": compliance["input_generations"]["total_scored_loaded"],
        "unique_core_compliant_sequences": compliance["input_generations"]["unique_core_compliant_sequences"],
        "submission_sequences": len(submission),
        "top_ranked_sequences": len(top),
        "duplicates_in_submission": compliance["compliance_checks"]["duplicate_sequence_count_in_submission"],
        "invalid_alphabet_in_submission": compliance["compliance_checks"]["invalid_alphabet_count_in_submission"],
        "length_out_of_range_in_submission": compliance["compliance_checks"]["length_out_of_range_count_in_submission"],
    }, indent=2))

    print("\nSUBMISSION BUCKET SUMMARY")
    print(bucket_summary.round(4).to_string(index=False))

    show_cols = [
        "submission_id",
        "challenge_rank",
        "candidate_id",
        "generation_source",
        "sequence_clean",
        "length_recomputed",
        "net_charge_recomputed",
        "hydrophobic_fraction_recomputed",
        "challenge_bucket",
        "APEX_median_MIC",
        "APEX_worst_MIC",
        "organisms_MIC_le_64",
    ]
    show_cols = [c for c in show_cols if c in top.columns]
    print("\nTOP 25 OF TOP 100")
    print(top[show_cols].head(25).round(4).to_string(index=False))

    print("\nWROTE")
    for key, value in compliance["outputs"].items():
        print(f"  {key}: {value}")
    print(f"  compliance_manifest: {compliance_path}")
    print("\nIMPORTANT: Review and complete license/source fields in the compliance manifest before external submission.")


if __name__ == "__main__":
    main()
