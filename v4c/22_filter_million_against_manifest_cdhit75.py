#!/usr/bin/env python3
"""Screen the V4C million-peptide QC-pass corpus against a reference at 75% identity.

This is a scale-aware V4C adapter around the frozen V4B CD-HIT methodology:

    cd-hit-2d -i reference.fa -i2 candidates.fa -c 0.75 -n 2

Only compact novelty flags are written because the complete APEX-scored metadata
remain in the immutable merged master table. Candidates that CD-HIT does not
process (typically very short peptides) are explicitly labeled
``unprocessed_by_cdhit`` and are never counted as novelty passes.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

CANONICAL_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")


def load_v4b_filter_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "v4b" / "16_filter_100k_against_manifest_cdhit75.py"
    if not module_path.exists():
        raise FileNotFoundError(module_path)
    spec = importlib.util.spec_from_file_location("v4b_manifest75_filter", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates",
        default=(
            "v4c/results/final_funnel/00_merged_qc/"
            "v4c_million_scored_qc_pass.csv"
        ),
    )
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--identity", type=float, default=0.75)
    parser.add_argument("--word-length", type=int, default=2)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--memory-mb", type=int, default=0)
    parser.add_argument("--reference-sequence-col", default=None)
    parser.add_argument("--reference-id-col", default=None)
    parser.add_argument("--cdhit-bin", default="cd-hit-2d")
    parser.add_argument("--expected-candidates", type=int, default=1000000)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def atomic_to_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def bool_series(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values
    return values.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def load_candidates(path: Path, helper) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(path)

    header = pd.read_csv(path, nrows=0)
    if "candidate_id" not in header.columns:
        raise ValueError("Candidate master table must contain candidate_id.")
    sequence_col = helper.choose_sequence_column(header)

    usecols = ["candidate_id", sequence_col]
    for column in (
        "generation_source",
        "generation",
        "criteria_length",
        "v4c_final_funnel_qc_pass",
    ):
        if column in header.columns and column not in usecols:
            usecols.append(column)

    frame = pd.read_csv(path, usecols=usecols, low_memory=False)
    frame["candidate_id"] = frame["candidate_id"].astype(str)
    frame["sequence_clean"] = frame[sequence_col].map(helper.clean_sequence)

    if "v4c_final_funnel_qc_pass" in frame.columns:
        passed = bool_series(frame["v4c_final_funnel_qc_pass"])
        if not passed.all():
            raise ValueError(
                f"Candidate input contains {(~passed).sum():,} rows that did not pass frozen V4C QC."
            )

    valid = frame["sequence_clean"].map(
        lambda sequence: bool(sequence) and all(residue in CANONICAL_AA for residue in sequence)
    )
    if not valid.all():
        raise ValueError(f"Candidate input contains {(~valid).sum():,} noncanonical sequences.")
    if frame["candidate_id"].duplicated().any():
        raise ValueError("Candidate input contains duplicate candidate IDs.")
    if frame["sequence_clean"].duplicated().any():
        raise ValueError("Candidate input contains duplicate peptide sequences.")

    frame["sequence_length"] = frame["sequence_clean"].str.len()
    return frame.reset_index(drop=True)


def validate_outputs(paths: list[Path], overwrite: bool) -> None:
    existing = [path for path in paths if path.exists()]
    if existing and not overwrite:
        rendered = "\n".join(f"  {path}" for path in existing)
        raise FileExistsError(
            "Novelty outputs already exist. Review them or rerun with --overwrite:\n" + rendered
        )


def main() -> None:
    args = parse_args()
    if not 0.0 < args.identity <= 1.0:
        raise ValueError("--identity must be in (0, 1].")
    if args.word_length < 2:
        raise ValueError("--word-length must be at least 2.")

    helper = load_v4b_filter_module()
    candidate_path = Path(args.candidates)
    reference_path = Path(args.reference)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pct = int(round(args.identity * 100))
    flags_path = output_dir / f"v4c_1m_manifest{pct}_flags.csv"
    kept_path = output_dir / f"v4c_1m_manifest{pct}_kept_below_{pct}_identity.csv"
    removed_path = output_dir / f"v4c_1m_manifest{pct}_removed_ge_{pct}_identity.csv"
    unprocessed_path = output_dir / f"v4c_1m_manifest{pct}_unprocessed_by_cdhit.csv"
    summary_path = output_dir / f"v4c_manifest{pct}_filter_summary.json"
    generation_path = output_dir / f"v4c_manifest{pct}_status_by_generation.csv"
    length_path = output_dir / f"v4c_manifest{pct}_status_by_length.csv"
    candidate_fasta = output_dir / "v4c_1m_candidates_for_manifest75_filter.fasta"
    reference_fasta = output_dir / "reference_for_manifest75_filter.fasta"
    cdhit_kept_fasta = output_dir / f"v4c_candidates_not_{pct}pct_identical_to_reference.fasta"
    kept_fasta = output_dir / f"v4c_1m_manifest{pct}_kept_below_{pct}_identity.fasta"
    removed_fasta = output_dir / f"v4c_1m_manifest{pct}_removed_ge_{pct}_identity.fasta"
    unprocessed_fasta = output_dir / f"v4c_1m_manifest{pct}_unprocessed_by_cdhit.fasta"
    command_path = output_dir / "cdhit_manifest75_command.txt"

    validate_outputs(
        [flags_path, kept_path, removed_path, unprocessed_path, summary_path],
        overwrite=args.overwrite,
    )

    print("[V4C-NOVELTY] Loading compact candidate index")
    candidates = load_candidates(candidate_path, helper)
    if args.expected_candidates > 0 and len(candidates) != args.expected_candidates:
        raise ValueError(
            f"Candidate count {len(candidates):,} != expected {args.expected_candidates:,}."
        )

    print("[V4C-NOVELTY] Loading and deduplicating reference manifest")
    reference = helper.load_reference(
        reference_path,
        args.reference_sequence_col,
        args.reference_id_col,
    )
    if reference.empty:
        raise ValueError("Reference manifest contains no valid canonical sequences.")

    candidates["cdhit_candidate_id"] = [
        helper.safe_id(value, "cand", index + 1)
        for index, value in enumerate(candidates["candidate_id"])
    ]
    if candidates["cdhit_candidate_id"].duplicated().any():
        raise ValueError("Sanitized candidate IDs are not unique for CD-HIT.")

    reference["cdhit_reference_id"] = [
        helper.safe_id(value, "ref", index + 1)
        for index, value in enumerate(reference["reference_id"])
    ]
    if reference["cdhit_reference_id"].duplicated().any():
        # Reference labels may collide after sanitization; replace them with stable row IDs.
        reference["cdhit_reference_id"] = [f"V4CREF_{index + 1:09d}" for index in range(len(reference))]

    helper.write_fasta(candidates, candidate_fasta, "cdhit_candidate_id", "sequence_clean")
    helper.write_fasta(reference, reference_fasta, "cdhit_reference_id", "reference_sequence_clean")

    cmd = [
        args.cdhit_bin,
        "-i", str(reference_fasta),
        "-i2", str(candidate_fasta),
        "-o", str(cdhit_kept_fasta),
        "-c", str(args.identity),
        "-n", str(args.word_length),
        "-d", "0",
        "-T", str(args.threads),
        "-M", str(args.memory_mb),
    ]
    command_path.write_text(" ".join(cmd) + "\n", encoding="utf-8")

    if args.dry_run:
        print("[V4C-NOVELTY] Dry run complete. Prepared command:")
        print(" ".join(cmd))
        return

    if shutil.which(args.cdhit_bin) is None:
        raise SystemExit(
            f"Could not find {args.cdhit_bin!r} on PATH. Install with:\n"
            "  conda install -c bioconda cd-hit -y"
        )

    print("[V4C-NOVELTY] Running frozen V4B CD-HIT identity method")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

    kept_ids = helper.parse_fasta_ids(cdhit_kept_fasta)
    candidate_ids = set(candidates["cdhit_candidate_id"].astype(str))
    cluster_path = Path(str(cdhit_kept_fasta) + ".clstr")
    cluster_members = helper.parse_cdhit_cluster_members(cluster_path)
    clustered_candidate_ids: set[str] = set()
    if not cluster_members.empty:
        clustered_candidate_ids = set(
            cluster_members.loc[
                cluster_members["cdhit_id"].astype(str).isin(candidate_ids),
                "cdhit_id",
            ].astype(str)
        )

    processed_ids = kept_ids | clustered_candidate_ids
    removed_ids = clustered_candidate_ids - kept_ids
    unprocessed_ids = candidate_ids - processed_ids

    def status(candidate_id: str) -> str:
        if candidate_id in kept_ids:
            return "kept_below_threshold"
        if candidate_id in removed_ids:
            return "removed_manifest_ge_threshold"
        if candidate_id in unprocessed_ids:
            return "unprocessed_by_cdhit"
        return "unknown_cdhit_status"

    candidates["manifest75_status"] = candidates["cdhit_candidate_id"].map(status)
    candidates["manifest_identity_threshold"] = float(args.identity)
    candidates["manifest_reference_file"] = str(reference_path)

    hit_info = helper.infer_reference_for_removed(cluster_members, candidate_ids)
    if not hit_info.empty:
        candidates = candidates.merge(hit_info, on="cdhit_candidate_id", how="left")

    unknown = candidates["manifest75_status"].eq("unknown_cdhit_status")
    if unknown.any():
        raise RuntimeError(f"CD-HIT status could not be resolved for {unknown.sum():,} candidates.")

    kept = candidates[candidates["manifest75_status"].eq("kept_below_threshold")].copy()
    removed = candidates[candidates["manifest75_status"].eq("removed_manifest_ge_threshold")].copy()
    unprocessed = candidates[candidates["manifest75_status"].eq("unprocessed_by_cdhit")].copy()

    atomic_to_csv(candidates, flags_path)
    atomic_to_csv(kept, kept_path)
    atomic_to_csv(removed, removed_path)
    atomic_to_csv(unprocessed, unprocessed_path)
    helper.write_fasta(kept, kept_fasta, "candidate_id", "sequence_clean")
    helper.write_fasta(removed, removed_fasta, "candidate_id", "sequence_clean")
    helper.write_fasta(unprocessed, unprocessed_fasta, "candidate_id", "sequence_clean")

    generation_column = "generation_source" if "generation_source" in candidates.columns else "generation"
    by_generation = (
        candidates.groupby([generation_column, "manifest75_status"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    atomic_to_csv(by_generation, generation_path)

    by_length = (
        candidates.groupby(["sequence_length", "manifest75_status"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    atomic_to_csv(by_length, length_path)

    summary = {
        "schema_version": "1.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "AMP-JEPA-Hybrid V4C",
        "stage": "manifest75_novelty_screen",
        "identity_threshold": float(args.identity),
        "reference_file": str(reference_path),
        "reference_unique_sequences": int(len(reference)),
        "candidate_unique_sequences": int(len(candidates)),
        "cdhit_kept_below_threshold": int(len(kept)),
        "cdhit_removed_manifest_ge_threshold": int(len(removed)),
        "cdhit_unprocessed": int(len(unprocessed)),
        "fraction_kept_among_all_candidates": float(len(kept) / max(len(candidates), 1)),
        "fraction_removed_among_all_candidates": float(len(removed) / max(len(candidates), 1)),
        "fraction_unprocessed_among_all_candidates": float(len(unprocessed) / max(len(candidates), 1)),
        "fraction_removed_among_cdhit_processed": float(
            len(removed) / max(len(kept) + len(removed), 1)
        ),
        "method": "cd-hit-2d; frozen V4B settings",
        "important_note": (
            "unprocessed_by_cdhit candidates are not novelty passes and require "
            "short-peptide alignment follow-up"
        ),
        "command": " ".join(cmd),
        "outputs": {
            "all_flags": str(flags_path),
            "kept": str(kept_path),
            "removed": str(removed_path),
            "unprocessed": str(unprocessed_path),
            "status_by_generation": str(generation_path),
            "status_by_length": str(length_path),
            "cdhit_raw_kept_fasta": str(cdhit_kept_fasta),
            "cdhit_cluster_file": str(cluster_path),
            "command_file": str(command_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nV4C MANIFEST {pct}% IDENTITY FILTER SUMMARY")
    print(json.dumps(summary, indent=2))
    print("\nSTATUS BY GENERATION")
    print(by_generation.to_string(index=False))
    print("\nSTATUS BY LENGTH")
    print(by_length.to_string(index=False))


if __name__ == "__main__":
    main()
