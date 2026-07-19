#!/usr/bin/env python3
"""Merge and audit all V4C APEX-scored generations without modifying raw outputs.

The script joins each generation's frozen candidate table to its APEX score table,
verifies one-to-one candidate/sequence alignment, recomputes core sequence QC fields,
and writes a single downstream-ready million-peptide table.

Raw ``v4c/results/generation_*`` directories are read-only inputs. All new files are
written under ``v4c/results/final_funnel/00_merged_qc`` by default.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

CANONICAL_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="v4c/results")
    parser.add_argument(
        "--output-dir",
        default="v4c/results/final_funnel/00_merged_qc",
    )
    parser.add_argument("--start-generation", type=int, default=1)
    parser.add_argument("--end-generation", type=int, default=10)
    parser.add_argument("--expected-per-generation", type=int, default=100000)
    parser.add_argument("--expected-total", type=int, default=1000000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def clean_sequence(value: object) -> str:
    return re.sub(r"\s+", "", str(value).upper())


def sequence_charge(sequence: str) -> int:
    return sequence.count("K") + sequence.count("R") - sequence.count("D") - sequence.count("E")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_to_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def validate_output_paths(paths: list[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        rendered = "\n".join(f"  {path}" for path in existing)
        raise FileExistsError(
            "Refusing to overwrite existing final-funnel outputs. "
            "Use --overwrite only after reviewing them:\n" + rendered
        )


def align_scored_to_candidates(
    candidates: pd.DataFrame,
    scored: pd.DataFrame,
    generation: int,
) -> pd.DataFrame:
    required_candidate = {"candidate_id", "sequence"}
    missing_candidate = sorted(required_candidate - set(candidates.columns))
    if missing_candidate:
        raise ValueError(
            f"Generation {generation:02d} candidate table is missing: {missing_candidate}"
        )

    candidates = candidates.copy()
    candidates["candidate_id"] = candidates["candidate_id"].astype(str)
    candidates["sequence"] = candidates["sequence"].map(clean_sequence)

    if candidates["candidate_id"].duplicated().any():
        raise ValueError(f"Generation {generation:02d} has duplicate candidate IDs.")
    if candidates["sequence"].duplicated().any():
        raise ValueError(f"Generation {generation:02d} has duplicate candidate sequences.")

    scored = scored.copy()
    if "candidate_id" not in scored.columns:
        if "apex_candidate_id" in scored.columns:
            proposed = scored["apex_candidate_id"].astype(str)
            if set(proposed) == set(candidates["candidate_id"]):
                scored["candidate_id"] = proposed
            else:
                raise ValueError(
                    f"Generation {generation:02d} score table lacks candidate_id, and "
                    "apex_candidate_id does not match the frozen candidate IDs."
                )
        else:
            raise ValueError(
                f"Generation {generation:02d} score table lacks candidate_id."
            )

    scored["candidate_id"] = scored["candidate_id"].astype(str)
    if scored["candidate_id"].duplicated().any():
        duplicates = int(scored["candidate_id"].duplicated().sum())
        raise ValueError(
            f"Generation {generation:02d} score table has {duplicates:,} duplicate candidate IDs."
        )

    candidate_ids = candidates["candidate_id"].tolist()
    candidate_set = set(candidate_ids)
    score_set = set(scored["candidate_id"])
    if candidate_set != score_set:
        missing_scores = len(candidate_set - score_set)
        unexpected_scores = len(score_set - candidate_set)
        raise ValueError(
            f"Generation {generation:02d} candidate/APEX ID mismatch: "
            f"missing_scores={missing_scores:,}, unexpected_scores={unexpected_scores:,}."
        )

    scored = scored.set_index("candidate_id", drop=False).loc[candidate_ids].reset_index(drop=True)

    score_sequence_column = None
    for column in ("sequence", "sequence_clean", "PeptideSequence"):
        if column in scored.columns:
            score_sequence_column = column
            break
    if score_sequence_column is not None:
        score_sequences = scored[score_sequence_column].map(clean_sequence).to_numpy()
        candidate_sequences = candidates["sequence"].to_numpy()
        if not np.array_equal(score_sequences, candidate_sequences):
            mismatches = int(np.sum(score_sequences != candidate_sequences))
            raise ValueError(
                f"Generation {generation:02d} candidate/APEX sequence mismatch in "
                f"{mismatches:,} rows."
            )

    # The APEX table is the wide base because it carries all pathogen-model columns.
    # Frozen candidate metadata are copied in only when absent, while canonical ID and
    # sequence are always forced to the validated candidate-table values.
    merged = scored.copy()
    for column in candidates.columns:
        if column not in merged.columns:
            merged[column] = candidates[column].to_numpy()

    merged["candidate_id"] = candidates["candidate_id"].to_numpy()
    merged["sequence"] = candidates["sequence"].to_numpy()
    merged["generation_source"] = generation

    if "generation" in candidates.columns:
        candidate_generation = pd.to_numeric(candidates["generation"], errors="coerce")
        if candidate_generation.isna().any() or not candidate_generation.eq(generation).all():
            raise ValueError(
                f"Generation {generation:02d} candidate table contains inconsistent generation labels."
            )
        merged["generation"] = generation

    return merged


def main() -> None:
    args = parse_args()
    if args.start_generation < 1:
        raise ValueError("--start-generation must be at least 1.")
    if args.end_generation < args.start_generation:
        raise ValueError("--end-generation must be >= --start-generation.")

    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    qc_pass_path = output_dir / "v4c_million_scored_qc_pass.csv"
    qc_fail_path = output_dir / "v4c_million_scored_qc_fail.csv"
    audit_path = output_dir / "v4c_generation_merge_audit.csv"
    manifest_path = output_dir / "v4c_million_merge_qc_manifest.json"
    output_paths = [qc_pass_path, qc_fail_path, audit_path, manifest_path]
    validate_output_paths(output_paths, overwrite=args.overwrite)

    merged_generations: list[pd.DataFrame] = []
    audits: list[dict[str, object]] = []
    source_files: list[dict[str, object]] = []

    for generation in range(args.start_generation, args.end_generation + 1):
        pad = f"{generation:02d}"
        directory = results_root / f"generation_{pad}"
        candidate_path = directory / f"generation_{pad}_candidates_pre_apex.csv"
        score_path = directory / f"generation_{pad}_apex_scores.csv"

        for path in (candidate_path, score_path):
            if not path.exists() or path.stat().st_size == 0:
                raise FileNotFoundError(f"Missing or empty frozen V4C input: {path}")

        print(f"[V4C-FUNNEL] Reading Generation {pad} candidates")
        candidates = pd.read_csv(candidate_path, low_memory=False)
        print(f"[V4C-FUNNEL] Reading Generation {pad} APEX scores")
        scored = pd.read_csv(score_path, low_memory=False)

        if args.expected_per_generation > 0:
            if len(candidates) != args.expected_per_generation:
                raise ValueError(
                    f"Generation {pad} candidate count {len(candidates):,} != "
                    f"expected {args.expected_per_generation:,}."
                )
            if len(scored) != args.expected_per_generation:
                raise ValueError(
                    f"Generation {pad} score count {len(scored):,} != "
                    f"expected {args.expected_per_generation:,}."
                )

        merged = align_scored_to_candidates(candidates, scored, generation)

        median_mic = numeric_series(merged, "APEX_median_MIC")
        mean_mic = numeric_series(merged, "APEX_mean_MIC")
        worst_mic = numeric_series(merged, "APEX_worst_MIC")
        if median_mic.isna().any() or mean_mic.isna().any() or worst_mic.isna().any():
            raise ValueError(
                f"Generation {pad} contains missing/non-numeric APEX summary MIC values."
            )

        audits.append(
            {
                "generation": generation,
                "candidate_rows": int(len(candidates)),
                "apex_rows": int(len(scored)),
                "merged_rows": int(len(merged)),
                "unique_candidate_ids": int(merged["candidate_id"].nunique()),
                "unique_sequences": int(merged["sequence"].nunique()),
                "best_APEX_median_MIC": float(median_mic.min()),
                "median_APEX_median_MIC": float(median_mic.median()),
                "best_APEX_worst_MIC": float(worst_mic.min()),
            }
        )
        source_files.extend(
            [
                {
                    "generation": generation,
                    "role": "candidates_pre_apex",
                    "path": str(candidate_path),
                    "size_bytes": candidate_path.stat().st_size,
                    "sha256": sha256_file(candidate_path),
                },
                {
                    "generation": generation,
                    "role": "apex_scores",
                    "path": str(score_path),
                    "size_bytes": score_path.stat().st_size,
                    "sha256": sha256_file(score_path),
                },
            ]
        )
        merged_generations.append(merged)
        print(f"[V4C-FUNNEL] Generation {pad} alignment passed for {len(merged):,} rows")

    print("[V4C-FUNNEL] Concatenating validated generations")
    all_scored = pd.concat(merged_generations, ignore_index=True, sort=False)
    del merged_generations

    if args.expected_total > 0 and len(all_scored) != args.expected_total:
        raise ValueError(
            f"Merged row count {len(all_scored):,} != expected {args.expected_total:,}."
        )
    if not all_scored["candidate_id"].astype(str).is_unique:
        raise ValueError("Global duplicate candidate IDs detected after merge.")

    all_scored["sequence_clean"] = all_scored["sequence"].map(clean_sequence)
    if not all_scored["sequence_clean"].is_unique:
        raise ValueError("Global duplicate peptide sequences detected after merge.")

    all_scored["criteria_canonical_sequence"] = all_scored["sequence_clean"].map(
        lambda sequence: bool(sequence) and all(residue in CANONICAL_AA for residue in sequence)
    )
    all_scored["criteria_length"] = all_scored["sequence_clean"].str.len().astype(np.int16)
    all_scored["criteria_charge"] = all_scored["sequence_clean"].map(sequence_charge).astype(np.int16)
    all_scored["criteria_hydrophobic_fraction"] = numeric_series(
        all_scored, "hydrophobic_fraction"
    ).astype(np.float32)

    all_scored["criteria_length_pass"] = all_scored["criteria_length"].between(10, 40)
    all_scored["criteria_charge_pass"] = all_scored["criteria_charge"].between(2, 12)
    all_scored["criteria_hydrophobicity_pass"] = all_scored[
        "criteria_hydrophobic_fraction"
    ].between(0.20, 0.70)
    all_scored["criteria_apex_summary_complete"] = (
        numeric_series(all_scored, "APEX_median_MIC").notna()
        & numeric_series(all_scored, "APEX_mean_MIC").notna()
        & numeric_series(all_scored, "APEX_worst_MIC").notna()
    )

    if "length" in all_scored.columns:
        supplied_length = pd.to_numeric(all_scored["length"], errors="coerce")
        all_scored["audit_length_matches_generator"] = supplied_length.eq(
            all_scored["criteria_length"]
        )
    else:
        all_scored["audit_length_matches_generator"] = True

    if "net_charge_KR_minus_DE" in all_scored.columns:
        supplied_charge = pd.to_numeric(
            all_scored["net_charge_KR_minus_DE"], errors="coerce"
        )
        all_scored["audit_charge_matches_generator"] = supplied_charge.eq(
            all_scored["criteria_charge"]
        )
    else:
        all_scored["audit_charge_matches_generator"] = True

    all_scored["v4c_final_funnel_qc_pass"] = (
        all_scored["criteria_canonical_sequence"]
        & all_scored["criteria_length_pass"]
        & all_scored["criteria_charge_pass"]
        & all_scored["criteria_hydrophobicity_pass"]
        & all_scored["criteria_apex_summary_complete"]
        & all_scored["audit_length_matches_generator"]
        & all_scored["audit_charge_matches_generator"]
    )

    qc_pass = all_scored[all_scored["v4c_final_funnel_qc_pass"]].copy()
    qc_fail = all_scored[~all_scored["v4c_final_funnel_qc_pass"]].copy()

    print(f"[V4C-FUNNEL] Writing QC-pass table: {len(qc_pass):,} rows")
    atomic_to_csv(qc_pass, qc_pass_path)
    print(f"[V4C-FUNNEL] Writing QC-fail table: {len(qc_fail):,} rows")
    atomic_to_csv(qc_fail, qc_fail_path)

    audit_frame = pd.DataFrame(audits)
    atomic_to_csv(audit_frame, audit_path)

    generation_counts = (
        qc_pass.groupby("generation_source", sort=True)
        .size()
        .astype(int)
        .to_dict()
    )
    manifest = {
        "schema_version": "1.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "AMP-JEPA-Hybrid V4C final funnel",
        "stage": "00_merged_qc",
        "raw_inputs_immutable": True,
        "generation_range": [args.start_generation, args.end_generation],
        "expected_per_generation": args.expected_per_generation,
        "expected_total": args.expected_total,
        "merged_rows": int(len(all_scored)),
        "unique_candidate_ids": int(all_scored["candidate_id"].nunique()),
        "unique_sequences": int(all_scored["sequence_clean"].nunique()),
        "qc_pass_rows": int(len(qc_pass)),
        "qc_fail_rows": int(len(qc_fail)),
        "qc_pass_by_generation": {str(k): int(v) for k, v in generation_counts.items()},
        "frozen_qc_thresholds": {
            "canonical_amino_acids_only": True,
            "minimum_length": 10,
            "maximum_length": 40,
            "minimum_charge_KR_minus_DE": 2,
            "maximum_charge_KR_minus_DE": 12,
            "minimum_hydrophobic_fraction": 0.20,
            "maximum_hydrophobic_fraction": 0.70,
            "apex_summary_fields_required": [
                "APEX_mean_MIC",
                "APEX_median_MIC",
                "APEX_worst_MIC",
            ],
        },
        "source_files": source_files,
        "outputs": {
            "qc_pass": {
                "path": str(qc_pass_path),
                "size_bytes": qc_pass_path.stat().st_size,
                "sha256": sha256_file(qc_pass_path),
            },
            "qc_fail": {
                "path": str(qc_fail_path),
                "size_bytes": qc_fail_path.stat().st_size,
                "sha256": sha256_file(qc_fail_path),
            },
            "generation_audit": {
                "path": str(audit_path),
                "size_bytes": audit_path.stat().st_size,
                "sha256": sha256_file(audit_path),
            },
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\nV4C MILLION-PEPTIDE MERGE/QC SUMMARY")
    print(json.dumps({
        "merged_rows": manifest["merged_rows"],
        "unique_candidate_ids": manifest["unique_candidate_ids"],
        "unique_sequences": manifest["unique_sequences"],
        "qc_pass_rows": manifest["qc_pass_rows"],
        "qc_fail_rows": manifest["qc_fail_rows"],
        "qc_pass_path": str(qc_pass_path),
        "manifest": str(manifest_path),
    }, indent=2))

    if len(qc_pass) != len(all_scored):
        raise RuntimeError(
            f"Merge completed, but {len(qc_fail):,} rows failed frozen V4C QC. "
            f"Review {qc_fail_path} before novelty screening."
        )

    print("\nV4C MILLION-PEPTIDE MERGE AND FROZEN-QC AUDIT PASSED")


if __name__ == "__main__":
    main()
