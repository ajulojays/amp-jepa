#!/usr/bin/env python3
"""Import the frozen V4A candidate population as immutable V4B Generation 0.

The script validates and deduplicates peptide sequences, preserves V4A annotations,
assigns stable sequence-derived candidate IDs, and writes a provenance manifest.
It never modifies the V4A source table.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

CANONICAL_AA = set("ACDEFGHIKLMNPQRSTVWY")
BOOLEAN_COLUMNS = [
    "is_optimization_success",
    "is_g_rescue",
    "is_elite",
    "is_pareto",
    "is_elite_pareto",
    "is_potent_any_organism",
    "is_narrow_spectrum_specialist",
    "is_broad_spectrum",
]
V4B_RESERVED_COLUMNS = [
    "candidate_id",
    "generation",
    "parent_candidate_id",
    "lineage_depth",
    "v4b_source",
]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def clean_sequence(value: Any) -> str:
    return "".join(str(value).upper().split())


def valid_sequence(sequence: str, min_len: int, max_len: int) -> bool:
    return (
        min_len <= len(sequence) <= max_len
        and all(residue in CANONICAL_AA for residue in sequence)
    )


def stable_candidate_id(sequence: str) -> str:
    sequence_hash = hashlib.sha256(sequence.encode("utf-8")).hexdigest()[:16]
    return f"V4B_G00_{sequence_hash}"


def normalize_boolean(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)

    true_values = {"1", "true", "t", "yes", "y"}
    return (
        series.fillna(False)
        .map(lambda value: str(value).strip().lower() in true_values)
        .astype(bool)
    )


def preserve_reserved_source_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Preserve source metadata that collides with V4B-owned column names."""
    out = df.copy()
    renamed: dict[str, str] = {}

    for column in V4B_RESERVED_COLUMNS:
        if column not in out.columns:
            continue

        target = f"v4a_{column}"
        suffix = 1
        while target in out.columns:
            target = f"v4a_{column}_{suffix}"
            suffix += 1

        out.rename(columns={column: target}, inplace=True)
        renamed[column] = target

    return out, renamed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="v4/results/final_panel/v4a_candidate_groups_all.csv",
        help="Frozen V4A candidate table.",
    )
    parser.add_argument(
        "--outdir",
        default="v4b/results/generation_00",
        help="Generation 0 output directory.",
    )
    parser.add_argument("--min-len", type=int, default=8)
    parser.add_argument("--max-len", type=int, default=64)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of existing Generation 0 outputs.",
    )
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    outdir = Path(args.outdir)
    candidate_path = outdir / "generation_00_candidates.csv"
    manifest_path = outdir / "generation_manifest.json"
    rejected_path = outdir / "generation_00_rejected_rows.csv"

    if not input_path.exists():
        raise FileNotFoundError(f"V4A candidate table not found: {input_path}")

    existing = [path for path in (candidate_path, manifest_path) if path.exists()]
    if existing and not args.overwrite:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Generation 0 already exists ({names}). Use --overwrite only deliberately."
        )

    source = pd.read_csv(input_path, low_memory=False)
    if "sequence" not in source.columns:
        raise ValueError("The V4A table must contain a 'sequence' column.")

    source_rows = len(source)
    source, renamed_source_columns = preserve_reserved_source_columns(source)
    source["sequence_original"] = source["sequence"].astype(str)
    source["sequence"] = source["sequence"].map(clean_sequence)
    source["sequence_valid"] = source["sequence"].map(
        lambda sequence: valid_sequence(sequence, args.min_len, args.max_len)
    )

    invalid = source.loc[~source["sequence_valid"]].copy()
    invalid["rejection_reason"] = "invalid_sequence"

    accepted = source.loc[source["sequence_valid"]].copy()
    duplicate_mask = accepted.duplicated("sequence", keep="first")
    duplicate_rows = int(duplicate_mask.sum())
    duplicates = accepted.loc[duplicate_mask].copy()
    duplicates["rejection_reason"] = "duplicate_sequence"
    rejected = pd.concat([invalid, duplicates], ignore_index=True, sort=False)

    accepted = accepted.loc[~duplicate_mask].copy()
    accepted.drop(columns=["sequence_valid"], inplace=True)

    for column in BOOLEAN_COLUMNS:
        if column not in accepted.columns:
            accepted[column] = False
        accepted[column] = normalize_boolean(accepted[column])

    accepted.insert(0, "candidate_id", accepted["sequence"].map(stable_candidate_id))
    accepted.insert(1, "generation", 0)
    accepted.insert(2, "parent_candidate_id", "")
    accepted.insert(3, "lineage_depth", 0)
    accepted.insert(4, "v4b_source", "frozen_v4a")

    if accepted["candidate_id"].duplicated().any():
        raise RuntimeError("Stable candidate IDs are unexpectedly non-unique.")

    outdir.mkdir(parents=True, exist_ok=True)
    accepted.to_csv(candidate_path, index=False)
    if not rejected.empty:
        rejected.to_csv(rejected_path, index=False)
    elif rejected_path.exists():
        rejected_path.unlink()

    group_counts = {
        column: int(accepted[column].sum())
        for column in BOOLEAN_COLUMNS
        if column in accepted.columns
    }
    manifest = {
        "schema_version": "1.1",
        "v4b_stage": "generation_00_import",
        "generation": 0,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "immutable_baseline": True,
        "source": {
            "path": str(input_path),
            "sha256": sha256_file(input_path),
            "rows": int(source_rows),
            "renamed_reserved_columns": renamed_source_columns,
        },
        "validation": {
            "minimum_length": args.min_len,
            "maximum_length": args.max_len,
            "canonical_amino_acids_only": True,
            "accepted_unique_sequences": int(len(accepted)),
            "invalid_rows": int(len(invalid)),
            "duplicate_rows": duplicate_rows,
            "rejected_rows_total": int(len(rejected)),
        },
        "candidate_group_counts": group_counts,
        "outputs": {
            "candidates": str(candidate_path),
            "candidates_sha256": sha256_file(candidate_path),
            "rejected_rows": str(rejected_path) if rejected_path.exists() else None,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
