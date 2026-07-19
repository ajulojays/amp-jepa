#!/usr/bin/env python3
"""Select a scalable self-nonredundant V4C dual-novel MIC32 portfolio.

The V4B exact greedy global-edit screen is quadratic and is not practical for the
V4C pool of >200,000 candidates. This V4C stage therefore uses an iterative,
rank-aware CD-HIT reduction at 75% identity with coverage required over at least
75% of both sequences. Within every CD-HIT cluster, the strongest pre-ranked
candidate is retained, and the retained set is reclustered until stable. A final
certification run must return singleton clusters only.

This is a scalable self-redundancy screen under the documented CD-HIT criterion.
It does not claim exact equivalence to the V4B Levenshtein/global-edit metric;
that stricter metric should be applied later to the final experimental shortlist.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

CANONICAL_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=(
            "v4c/results/final_funnel/03_dual_novelty_mic32/"
            "v4c_dual_novel_MIC32_ranked.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="v4c/results/final_funnel/04_self_nonredundant_cdhit75",
    )
    parser.add_argument("--identity", type=float, default=0.75)
    parser.add_argument("--coverage-long", type=float, default=0.75)
    parser.add_argument("--coverage-short", type=float, default=0.75)
    parser.add_argument("--word-length", type=int, default=2)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--memory-mb", type=int, default=0)
    parser.add_argument("--max-iterations", type=int, default=8)
    parser.add_argument("--expected-input", type=int, default=213347)
    parser.add_argument("--cdhit-bin", default="cd-hit")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def clean_sequence(value: object) -> str:
    return re.sub(r"\s+", "", str(value).upper())


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def write_fasta(frame: pd.DataFrame, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for internal_id, sequence in zip(frame["self_cdhit_id"], frame["sequence_clean"]):
            handle.write(f">{internal_id}\n{sequence}\n")


def parse_clusters(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing/empty CD-HIT cluster file: {path}")

    rows: list[dict[str, object]] = []
    cluster_id: int | None = None
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">Cluster"):
                cluster_id = int(line.split()[-1])
                continue
            if cluster_id is None:
                raise ValueError(f"Malformed CD-HIT cluster file before first cluster: {path}")
            match = re.search(r">([^\.\s]+)\.\.\.", line)
            if not match:
                raise ValueError(f"Could not parse CD-HIT cluster member line: {line}")
            identity_match = re.search(r"at\s+([0-9.]+)%", line)
            rows.append(
                {
                    "cluster_id": cluster_id,
                    "self_cdhit_id": match.group(1),
                    "cdhit_is_representative": line.endswith("*"),
                    "cdhit_reported_identity": (
                        float(identity_match.group(1)) / 100.0
                        if identity_match
                        else np.nan
                    ),
                    "cdhit_cluster_line": line,
                }
            )

    clusters = pd.DataFrame(rows)
    if clusters.empty:
        raise ValueError(f"No cluster members parsed from {path}")
    return clusters


def run_cdhit(
    executable: str,
    input_fasta: Path,
    output_fasta: Path,
    identity: float,
    word_length: int,
    coverage_long: float,
    coverage_short: float,
    threads: int,
    memory_mb: int,
) -> list[str]:
    command = [
        executable,
        "-i", str(input_fasta),
        "-o", str(output_fasta),
        "-c", str(identity),
        "-n", str(word_length),
        "-G", "1",
        "-aL", str(coverage_long),
        "-aS", str(coverage_short),
        "-g", "1",
        "-d", "0",
        "-T", str(threads),
        "-M", str(memory_mb),
    ]
    print("[V4C-SELF] Running:", " ".join(command))
    subprocess.run(command, check=True)
    return command


def main() -> None:
    args = parse_args()
    for name, value in (
        ("identity", args.identity),
        ("coverage_long", args.coverage_long),
        ("coverage_short", args.coverage_short),
    ):
        if not 0.0 < value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in (0, 1].")
    if args.max_iterations < 1:
        raise ValueError("--max-iterations must be at least 1.")
    if shutil.which(args.cdhit_bin) is None:
        raise SystemExit(
            f"Could not find {args.cdhit_bin!r} on PATH. Install with:\n"
            "  conda install -c bioconda cd-hit -y"
        )

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    work_dir = output_dir / "cdhit_work"
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    selected_path = output_dir / "v4c_novel_MIC32_self_nonredundant_cdhit75.csv"
    selected_fasta = output_dir / "v4c_novel_MIC32_self_nonredundant_cdhit75.fasta"
    membership_path = output_dir / "v4c_novel_MIC32_cdhit75_membership.csv"
    rejected_path = output_dir / "v4c_novel_MIC32_rejected_cdhit75_compact.csv"
    iteration_path = output_dir / "v4c_self_nonredundancy_iteration_summary.csv"
    summary_path = output_dir / "v4c_self_nonredundancy_cdhit75_summary.json"
    outputs = [
        selected_path,
        selected_fasta,
        membership_path,
        rejected_path,
        iteration_path,
        summary_path,
    ]
    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Self-nonredundancy outputs already exist; review them or use --overwrite:\n"
            + "\n".join(str(path) for path in existing)
        )
    if not input_path.exists() or input_path.stat().st_size == 0:
        raise FileNotFoundError(input_path)

    print("[V4C-SELF] Loading ranked dual-novel MIC32 pool")
    frame = pd.read_csv(input_path, low_memory=False)
    required = {"candidate_id", "APEX_median_MIC"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")

    sequence_column = "sequence_clean" if "sequence_clean" in frame.columns else "sequence"
    if sequence_column not in frame.columns:
        raise ValueError("Input must contain sequence_clean or sequence.")

    frame["candidate_id"] = frame["candidate_id"].astype(str)
    frame["sequence_clean"] = frame[sequence_column].map(clean_sequence)
    frame["APEX_median_MIC"] = pd.to_numeric(frame["APEX_median_MIC"], errors="coerce")
    if frame["APEX_median_MIC"].isna().any():
        raise ValueError("Input contains missing/non-numeric APEX_median_MIC values.")
    if frame["candidate_id"].duplicated().any():
        raise ValueError("Input contains duplicate candidate IDs.")
    if frame["sequence_clean"].duplicated().any():
        raise ValueError("Input contains duplicate peptide sequences.")
    valid = frame["sequence_clean"].map(
        lambda sequence: bool(sequence) and all(residue in CANONICAL_AA for residue in sequence)
    )
    if not valid.all():
        raise ValueError(f"Input contains {(~valid).sum():,} invalid/noncanonical sequences.")
    if args.expected_input > 0 and len(frame) != args.expected_input:
        raise ValueError(
            f"Input count {len(frame):,} != expected {args.expected_input:,}."
        )

    if "v4c_pre_self_similarity_rank" in frame.columns:
        frame["v4c_pre_self_similarity_rank"] = pd.to_numeric(
            frame["v4c_pre_self_similarity_rank"], errors="raise"
        ).astype(int)
        frame = frame.sort_values("v4c_pre_self_similarity_rank").reset_index(drop=True)
    else:
        frame = frame.reset_index(drop=True)
        frame["v4c_pre_self_similarity_rank"] = np.arange(1, len(frame) + 1)

    frame["self_cdhit_id"] = [f"V4CSELF_{index + 1:09d}" for index in range(len(frame))]
    id_to_candidate = dict(zip(frame["self_cdhit_id"], frame["candidate_id"]))
    candidate_to_internal = dict(zip(frame["candidate_id"], frame["self_cdhit_id"]))
    rank_by_internal = dict(zip(frame["self_cdhit_id"], frame["v4c_pre_self_similarity_rank"]))

    # Propagate every original candidate to its current retained representative.
    original_to_current = dict(zip(frame["self_cdhit_id"], frame["self_cdhit_id"]))
    current = frame[["self_cdhit_id", "sequence_clean", "v4c_pre_self_similarity_rank"]].copy()
    iteration_rows: list[dict[str, object]] = []
    commands: list[str] = []

    for iteration in range(1, args.max_iterations + 1):
        input_fasta = work_dir / f"iteration_{iteration:02d}_input.fasta"
        output_fasta = work_dir / f"iteration_{iteration:02d}_clustered.fasta"
        write_fasta(current, input_fasta)
        command = run_cdhit(
            args.cdhit_bin,
            input_fasta,
            output_fasta,
            args.identity,
            args.word_length,
            args.coverage_long,
            args.coverage_short,
            args.threads,
            args.memory_mb,
        )
        commands.append(" ".join(command))

        clusters = parse_clusters(Path(str(output_fasta) + ".clstr"))
        current_ids = set(current["self_cdhit_id"])
        parsed_ids = set(clusters["self_cdhit_id"])
        if current_ids != parsed_ids:
            raise RuntimeError(
                f"Iteration {iteration}: CD-HIT did not account for all candidates; "
                f"missing={len(current_ids - parsed_ids):,}, unexpected={len(parsed_ids - current_ids):,}."
            )

        clusters["rank"] = clusters["self_cdhit_id"].map(rank_by_internal)
        winners = (
            clusters.sort_values(["cluster_id", "rank"])
            .drop_duplicates("cluster_id", keep="first")
            [["cluster_id", "self_cdhit_id", "rank"]]
            .rename(columns={"self_cdhit_id": "winner_self_cdhit_id", "rank": "winner_rank"})
        )
        clusters = clusters.merge(winners, on="cluster_id", how="left", validate="many_to_one")
        current_to_winner = dict(
            zip(clusters["self_cdhit_id"], clusters["winner_self_cdhit_id"])
        )
        original_to_current = {
            original: current_to_winner[current_rep]
            for original, current_rep in original_to_current.items()
        }

        winner_ids = winners.sort_values("winner_rank")["winner_self_cdhit_id"].tolist()
        previous_ids = current["self_cdhit_id"].tolist()
        next_current = (
            frame.set_index("self_cdhit_id")
            .loc[winner_ids, ["sequence_clean", "v4c_pre_self_similarity_rank"]]
            .reset_index()
            .sort_values("v4c_pre_self_similarity_rank")
            .reset_index(drop=True)
        )

        cluster_sizes = clusters.groupby("cluster_id").size()
        iteration_rows.append(
            {
                "iteration": iteration,
                "input_candidates": int(len(current)),
                "clusters": int(len(winners)),
                "retained_candidates": int(len(next_current)),
                "removed_this_iteration": int(len(current) - len(next_current)),
                "largest_cluster": int(cluster_sizes.max()),
                "multi_member_clusters": int((cluster_sizes > 1).sum()),
                "retained_set_unchanged": set(previous_ids) == set(winner_ids),
                "command": commands[-1],
            }
        )
        print(
            f"[V4C-SELF] Iteration {iteration}: {len(current):,} -> "
            f"{len(next_current):,}; multi-member clusters={(cluster_sizes > 1).sum():,}"
        )

        current = next_current
        if set(previous_ids) == set(winner_ids):
            break
    else:
        raise RuntimeError(
            f"Self-nonredundancy set did not stabilize within {args.max_iterations} iterations."
        )

    # Explicit final certification: every retained sequence must be a singleton cluster
    # under the exact same documented CD-HIT criterion.
    certification_input = work_dir / "certification_input.fasta"
    certification_output = work_dir / "certification_clustered.fasta"
    write_fasta(current, certification_input)
    certification_command = run_cdhit(
        args.cdhit_bin,
        certification_input,
        certification_output,
        args.identity,
        args.word_length,
        args.coverage_long,
        args.coverage_short,
        args.threads,
        args.memory_mb,
    )
    certification_clusters = parse_clusters(Path(str(certification_output) + ".clstr"))
    certification_sizes = certification_clusters.groupby("cluster_id").size()
    if not certification_sizes.eq(1).all():
        raise RuntimeError(
            "Final CD-HIT certification failed: retained candidates still form "
            f"{(certification_sizes > 1).sum():,} multi-member clusters."
        )
    if len(certification_clusters) != len(current):
        raise RuntimeError("Final CD-HIT certification row-count mismatch.")

    final_internal_ids = current.sort_values("v4c_pre_self_similarity_rank")["self_cdhit_id"].tolist()
    selected = (
        frame.set_index("self_cdhit_id")
        .loc[final_internal_ids]
        .sort_values("v4c_pre_self_similarity_rank")
        .reset_index()
    )
    selected["self_similarity_status"] = "selected_cdhit75_nonredundant"
    selected["v4c_self_nonredundant_rank"] = np.arange(1, len(selected) + 1)

    internal_to_final = original_to_current
    membership = frame[[
        "self_cdhit_id",
        "candidate_id",
        "v4c_pre_self_similarity_rank",
        "sequence_clean",
    ]].copy()
    membership["final_selected_self_cdhit_id"] = membership["self_cdhit_id"].map(internal_to_final)
    membership["final_selected_candidate_id"] = membership[
        "final_selected_self_cdhit_id"
    ].map(id_to_candidate)
    membership["selected"] = membership["self_cdhit_id"].eq(
        membership["final_selected_self_cdhit_id"]
    )
    membership["self_similarity_status"] = np.where(
        membership["selected"],
        "selected_cdhit75_nonredundant",
        "rejected_cdhit75_redundant",
    )
    selected_rank_map = dict(
        zip(selected["candidate_id"], selected["v4c_self_nonredundant_rank"])
    )
    membership["final_selected_rank"] = membership["final_selected_candidate_id"].map(
        selected_rank_map
    )

    rejected = membership.loc[
        ~membership["selected"],
        [
            "candidate_id",
            "sequence_clean",
            "v4c_pre_self_similarity_rank",
            "final_selected_candidate_id",
            "final_selected_rank",
            "self_similarity_status",
        ],
    ].copy()

    atomic_csv(selected.drop(columns="self_cdhit_id"), selected_path)
    with selected_fasta.open("w", encoding="utf-8") as handle:
        for _, row in selected.iterrows():
            handle.write(
                f">{row['candidate_id']}|rank={row['v4c_self_nonredundant_rank']}|"
                f"median_MIC={row['APEX_median_MIC']}\n{row['sequence_clean']}\n"
            )
    atomic_csv(membership.drop(columns="self_cdhit_id"), membership_path)
    atomic_csv(rejected, rejected_path)
    iteration_frame = pd.DataFrame(iteration_rows)
    atomic_csv(iteration_frame, iteration_path)

    summary = {
        "schema_version": "1.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "AMP-JEPA-Hybrid V4C",
        "stage": "self_nonredundancy_cdhit75",
        "input_file": str(input_path),
        "input_candidates": int(len(frame)),
        "selected_self_nonredundant": int(len(selected)),
        "rejected_redundant": int(len(frame) - len(selected)),
        "retained_fraction": float(len(selected) / max(len(frame), 1)),
        "identity_threshold": float(args.identity),
        "coverage_long": float(args.coverage_long),
        "coverage_short": float(args.coverage_short),
        "word_length": int(args.word_length),
        "method": (
            "iterative rank-aware CD-HIT global identity clustering; best pre-ranked "
            "member retained per cluster until stable"
        ),
        "important_note": (
            "This scalable CD-HIT criterion is not claimed to be exactly equivalent "
            "to V4B global Levenshtein identity. Apply exact global-edit/Needleman-"
            "Wunsch certification to the final experimental shortlist."
        ),
        "iterations": iteration_rows,
        "certification": {
            "command": " ".join(certification_command),
            "clusters": int(certification_clusters["cluster_id"].nunique()),
            "all_clusters_singleton": True,
        },
        "best_predicted_median_MIC_uM": float(selected["APEX_median_MIC"].min()),
        "outputs": {
            "selected": str(selected_path),
            "selected_fasta": str(selected_fasta),
            "membership": str(membership_path),
            "rejected_compact": str(rejected_path),
            "iteration_summary": str(iteration_path),
            "work_dir": str(work_dir),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nV4C SELF-NONREDUNDANCY SUMMARY")
    print(json.dumps(summary, indent=2))
    print("\nTOP SELECTED CANDIDATES")
    display_columns = [
        column
        for column in [
            "candidate_id",
            "generation_source",
            "sequence_clean",
            "APEX_median_MIC",
            "APEX_worst_MIC",
            "APEX_mean_MIC",
            "organisms_MIC_le_64",
            "criteria_length",
            "criteria_charge",
            "criteria_hydrophobic_fraction",
            "v4c_self_nonredundant_rank",
        ]
        if column in selected.columns
    ]
    print(selected[display_columns].head(40).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
