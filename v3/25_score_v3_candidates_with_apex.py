#!/usr/bin/env python3
"""Score AMP-JEPA-Hybrid v3 candidates with the APEX MIC ensemble.

This is the v3 equivalent of the older AMP-JEPA discovery scoring step.
It performs real APEX ensemble inference for v3-generated peptides, then adds
cross-organism MIC summaries so v3 candidates can be compared against bundled
APEX/ApexOracle rows.

Typical use:
    python v3/25_score_v3_candidates_with_apex.py \
      --candidates v3/results/top_panel_v3.csv \
      --output-dir v3/results/apex_scored_v3

Environment:
    APEX_ROOT defaults to /home/julojays/apex unless --apex-root is supplied.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))


BENCHMARK_PEPTIDES = {
    "LL-37": "LLGDFFRKSKEKIGKEFKRIVQRIKDFLRNLVPRTES",
    "Magainin 2": "GIGKFLHSAKKFGKAFVGEIMNS",
    "Cecropin A": "KWKLFKKIEKVGQNIRDGIIKAGPAVAVVGQATQIAK",
    "Melittin": "GIGAVLKVLTTGLPALISWIKRKRQQ",
    "Protegrin-1": "RGGRLCYCRRRFCVCVGR",
}

METADATA_COLUMNS = {
    "APEX_rank",
    "candidate_id",
    "apex_candidate_id",
    "PeptideSequence",
    "sequence",
    "Sequence",
    "record_type",
    "comparison_note",
    "v3_rank",
    "length",
    "net_charge",
    "net_charge_KR_minus_DE",
    "hydrophobic_fraction",
    "max_train_identity",
    "novelty_score",
    "developability_score",
    "v3_rank_score",
    "passes_v3_filters",
    "strategy",
    "source_a",
    "source_b",
    "alpha",
    "generation_round",
    "ended_with_end_token",
    "valid_basic",
    "duplicate_generated",
    "mean_hydropathy",
    "entropy",
    "maximum_residue_fraction",
    "longest_homopolymer",
    "nearest_apd_similarity",
    "nearest_apd_sequence",
    "pre_apex_score",
    "screen_rank",
    "nearest_selected_similarity",
}

SUMMARY_PREFIXES = (
    "APEX_",
    "mean_",
    "median_",
    "min_",
    "max_",
    "n_pathogens_",
    "organisms_MIC_",
    "fraction_MIC_",
    "key_organisms_",
)

AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def clean_sequence(value: object) -> str:
    sequence = str(value).strip().upper()
    return "".join(residue for residue in sequence if residue in AMINO_ACIDS)


def ensure_dataframe(predictions) -> pd.DataFrame:
    if isinstance(predictions, pd.DataFrame):
        return predictions.copy()
    return pd.DataFrame(predictions)


def prepare_candidate_table(path: Path, limit: int | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Candidate file not found: {path}")

    df = pd.read_csv(path)

    if "sequence" not in df.columns:
        for candidate in ["Sequence", "PeptideSequence", "peptide", "Peptide"]:
            if candidate in df.columns:
                df = df.rename(columns={candidate: "sequence"})
                break

    if "sequence" not in df.columns:
        raise ValueError("Candidate CSV needs a sequence, Sequence, or PeptideSequence column.")

    df["sequence"] = df["sequence"].map(clean_sequence)
    df = df[df["sequence"].str.len() > 0].copy()
    df = df.drop_duplicates(subset=["sequence"]).reset_index(drop=True)

    if "candidate_id" not in df.columns:
        df.insert(0, "candidate_id", [f"v3_candidate_{i+1:06d}" for i in range(len(df))])

    if "apex_candidate_id" not in df.columns:
        df.insert(0, "apex_candidate_id", [f"V3_APEX_{i+1:04d}" for i in range(len(df))])

    if limit is not None and limit > 0:
        df = df.head(limit).copy()

    return df.reset_index(drop=True)


def identify_mic_columns(prediction_dataframe: pd.DataFrame) -> List[str]:
    mic_columns = []

    for column in prediction_dataframe.columns:
        if column in METADATA_COLUMNS:
            continue
        if column.startswith(SUMMARY_PREFIXES):
            continue

        converted = pd.to_numeric(prediction_dataframe[column], errors="coerce")
        if converted.notna().any():
            mic_columns.append(column)

    return mic_columns


def is_gram_negative(column: str) -> bool:
    text = column.lower()
    markers = [
        "e. coli",
        "escherichia",
        "p. aeruginosa",
        "pseudomonas",
        "k. pneumoniae",
        "klebsiella",
        "a. baumannii",
        "acinetobacter",
    ]
    return any(marker in text for marker in markers)


def is_gram_positive(column: str) -> bool:
    text = column.lower()
    markers = [
        "s. aureus",
        "staphylococcus",
        "e. faecalis",
        "e. faecium",
        "enterococcus",
        "l. monocytogenes",
        "listeria",
        "b. subtilis",
        "bacillus",
        "streptococcus",
    ]
    return any(marker in text for marker in markers)


def add_mic_summaries(df: pd.DataFrame, mic_columns: Iterable[str]) -> pd.DataFrame:
    output = df.copy()
    mic_columns = [column for column in mic_columns if column in output.columns]

    if not mic_columns:
        return output

    numeric = output[mic_columns].apply(pd.to_numeric, errors="coerce")

    output["APEX_mean_MIC"] = numeric.mean(axis=1, skipna=True)
    output["APEX_median_MIC"] = numeric.median(axis=1, skipna=True)
    output["APEX_best_MIC"] = numeric.min(axis=1, skipna=True)
    output["APEX_worst_MIC"] = numeric.max(axis=1, skipna=True)
    output["APEX_MIC_std"] = numeric.std(axis=1, skipna=True)

    # Comparator-compatible aliases.
    output["mean_all_pred_MIC"] = output["APEX_mean_MIC"]
    output["median_pred_MIC"] = output["APEX_median_MIC"]
    output["median_all_pred_MIC"] = output["APEX_median_MIC"]
    output["min_pred_MIC"] = output["APEX_best_MIC"]
    output["max_pred_MIC"] = output["APEX_worst_MIC"]

    for threshold in [20, 32, 64, 80, 128]:
        output[f"organisms_MIC_le_{threshold}"] = numeric.le(threshold).sum(axis=1)
        output[f"fraction_MIC_le_{threshold}"] = numeric.le(threshold).mean(axis=1)
        output[f"n_pathogens_pred_MIC_le_{threshold}"] = numeric.le(threshold).sum(axis=1)

    gn_cols = [column for column in mic_columns if is_gram_negative(column)]
    gp_cols = [column for column in mic_columns if is_gram_positive(column)]

    if gn_cols:
        gn = output[gn_cols].apply(pd.to_numeric, errors="coerce")
        output["mean_GN_pred_MIC"] = gn.mean(axis=1, skipna=True)
        output["median_GN_pred_MIC"] = gn.median(axis=1, skipna=True)
        output["min_GN_pred_MIC"] = gn.min(axis=1, skipna=True)

    if gp_cols:
        gp = output[gp_cols].apply(pd.to_numeric, errors="coerce")
        output["mean_GP_pred_MIC"] = gp.mean(axis=1, skipna=True)
        output["median_GP_pred_MIC"] = gp.median(axis=1, skipna=True)
        output["min_GP_pred_MIC"] = gp.min(axis=1, skipna=True)

    return output


def load_apex(apex_root: Path, pattern: str, device: torch.device):
    sys.path.insert(0, str(apex_root))

    try:
        from apex_utils.apex_direct_ensemble import load_apex_ensemble, predict_mic_ensemble
    except Exception as exc:  # pragma: no cover - depends on local APEX checkout
        raise RuntimeError(
            "Could not import apex_utils.apex_direct_ensemble. "
            "Set --apex-root or APEX_ROOT to your local APEX checkout."
        ) from exc

    model_dir = apex_root / "trained_models"
    if not model_dir.exists():
        raise FileNotFoundError(f"APEX model directory not found: {model_dir}")

    models = load_apex_ensemble(
        model_dir=str(model_dir),
        pattern=pattern,
        device=device,
    )

    if len(models) == 0:
        raise RuntimeError(f"No APEX models matched pattern {pattern!r} in {model_dir}")

    return models, predict_mic_ensemble


def score_table_with_apex(
    metadata: pd.DataFrame,
    apex_models,
    predict_mic_ensemble,
    device: torch.device,
) -> tuple[pd.DataFrame, List[str]]:
    sequences = metadata["sequence"].tolist()

    predictions = predict_mic_ensemble(
        apex_models,
        sequences,
        device=device,
    )

    predictions = ensure_dataframe(predictions).reset_index(drop=True)

    for column in ["sequence", "Sequence", "peptide", "Peptide", "PeptideSequence"]:
        if column in predictions.columns:
            predictions = predictions.drop(columns=[column])

    mic_columns = identify_mic_columns(predictions)
    if not mic_columns:
        raise RuntimeError("No numeric APEX organism prediction columns were detected.")

    scored = pd.concat([metadata.reset_index(drop=True), predictions], axis=1)
    scored = add_mic_summaries(scored, mic_columns)

    return scored, mic_columns


def score_benchmarks(apex_models, predict_mic_ensemble, device: torch.device) -> pd.DataFrame:
    names = list(BENCHMARK_PEPTIDES.keys())
    sequences = list(BENCHMARK_PEPTIDES.values())

    metadata = pd.DataFrame(
        {
            "apex_candidate_id": names,
            "candidate_id": names,
            "sequence": sequences,
            "record_type": "benchmark",
        }
    )

    scored, _ = score_table_with_apex(metadata, apex_models, predict_mic_ensemble, device)
    return scored


def standardize_oracle(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
    if "sequence" not in df.columns:
        for candidate in ["PeptideSequence", "Sequence", "peptide_sequence"]:
            if candidate in df.columns:
                df = df.rename(columns={candidate: "sequence"})
                break

    if "sequence" not in df.columns:
        return pd.DataFrame()

    df["sequence"] = df["sequence"].map(clean_sequence)
    if "apex_candidate_id" not in df.columns:
        if "candidate_id" in df.columns:
            df["apex_candidate_id"] = df["candidate_id"].astype(str)
        elif "Unnamed: 0" in df.columns:
            df["apex_candidate_id"] = df["Unnamed: 0"].astype(str)
        else:
            df["apex_candidate_id"] = [f"APEX_ORACLE_{i+1:04d}" for i in range(len(df))]

    df["candidate_id"] = df["apex_candidate_id"]
    df["record_type"] = "apex_oracle"

    # Harmonize existing oracle summaries with v3-scored summaries.
    if "APEX_mean_MIC" not in df.columns and "mean_all_pred_MIC" in df.columns:
        df["APEX_mean_MIC"] = pd.to_numeric(df["mean_all_pred_MIC"], errors="coerce")
    if "APEX_median_MIC" not in df.columns:
        if "median_pred_MIC" in df.columns:
            df["APEX_median_MIC"] = pd.to_numeric(df["median_pred_MIC"], errors="coerce")
        elif "median_all_pred_MIC" in df.columns:
            df["APEX_median_MIC"] = pd.to_numeric(df["median_all_pred_MIC"], errors="coerce")
    if "APEX_best_MIC" not in df.columns and "min_pred_MIC" in df.columns:
        df["APEX_best_MIC"] = pd.to_numeric(df["min_pred_MIC"], errors="coerce")

    return df


def write_fasta(df: pd.DataFrame, path: Path, top_n: int) -> None:
    top = df.head(top_n).copy()
    with path.open("w", encoding="utf-8") as handle:
        for _, row in top.iterrows():
            handle.write(
                f">{row['apex_candidate_id']}|APEX_median_MIC={row.get('APEX_median_MIC', np.nan):.2f}|APEX_mean_MIC={row.get('APEX_mean_MIC', np.nan):.2f}\n"
            )
            handle.write(f"{row['sequence']}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", default="v3/results/top_panel_v3.csv")
    parser.add_argument("--output-dir", default="v3/results/apex_scored_v3")
    parser.add_argument("--oracle", default="v3/data/external/apex_oracle_ranked_summary.csv")
    parser.add_argument("--apex-root", default=os.environ.get("APEX_ROOT", "/home/julojays/apex"))
    parser.add_argument("--apex-pattern", default="trained_all_model_*_ensemble_*")
    parser.add_argument("--limit", type=int, default=0, help="Use 0 to score all rows in --candidates.")
    parser.add_argument("--top-fasta-count", type=int, default=10)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    candidate_path = resolve_path(args.candidates)
    output_dir = resolve_path(args.output_dir)
    oracle_path = resolve_path(args.oracle)
    apex_root = resolve_path(args.apex_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = prepare_candidate_table(candidate_path, limit=args.limit if args.limit > 0 else None)
    print(f"Loaded {len(candidates)} unique v3 candidates for APEX scoring.")

    apex_models, predict_mic_ensemble = load_apex(apex_root, args.apex_pattern, device)
    print(f"Loaded {len(apex_models)} APEX models from {apex_root / 'trained_models'}")

    print("Scoring v3 candidates with APEX...")
    scored_candidates, mic_columns = score_table_with_apex(
        candidates,
        apex_models,
        predict_mic_ensemble,
        device,
    )
    scored_candidates["record_type"] = "v3_generated"

    scored_candidates = scored_candidates.sort_values(
        by=["APEX_median_MIC", "APEX_worst_MIC", "APEX_mean_MIC"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    scored_candidates.insert(0, "APEX_rank", range(1, len(scored_candidates) + 1))

    print("Scoring benchmark AMPs with APEX...")
    scored_benchmarks = score_benchmarks(apex_models, predict_mic_ensemble, device)

    candidate_output = output_dir / "apex_scored_v3_candidates.csv"
    benchmark_output = output_dir / "apex_scored_v3_benchmarks.csv"
    combined_output = output_dir / "apex_scored_v3_combined.csv"
    fasta_output = output_dir / "apex_top_v3_candidates.fasta"
    summary_output = output_dir / "apex_scoring_summary.json"
    oracle_compare_output = output_dir / "apex_scored_v3_vs_oracle.csv"

    scored_candidates.to_csv(candidate_output, index=False)
    scored_benchmarks.to_csv(benchmark_output, index=False)

    combined = pd.concat(
        [
            scored_benchmarks.assign(record_type="benchmark"),
            scored_candidates.assign(record_type="v3_generated"),
        ],
        ignore_index=True,
        sort=False,
    )
    combined.to_csv(combined_output, index=False)
    write_fasta(scored_candidates, fasta_output, args.top_fasta_count)

    oracle = standardize_oracle(oracle_path)
    if not oracle.empty:
        oracle_compare = pd.concat(
            [
                oracle.assign(record_type="apex_oracle"),
                scored_candidates.assign(record_type="v3_generated"),
            ],
            ignore_index=True,
            sort=False,
        )
        sort_cols = [column for column in ["APEX_median_MIC", "APEX_worst_MIC", "APEX_mean_MIC"] if column in oracle_compare.columns]
        if sort_cols:
            oracle_compare = oracle_compare.sort_values(sort_cols, ascending=True, na_position="last")
        oracle_compare.to_csv(oracle_compare_output, index=False)

    summary = {
        "candidate_input": str(candidate_path),
        "number_candidates_scored": int(len(scored_candidates)),
        "number_apex_models": int(len(apex_models)),
        "number_organisms_scored": int(len(mic_columns)),
        "organism_columns": mic_columns,
        "mean_of_candidate_mean_MIC": float(scored_candidates["APEX_mean_MIC"].mean()),
        "median_of_candidate_median_MIC": float(scored_candidates["APEX_median_MIC"].median()),
        "best_candidate_median_MIC": float(scored_candidates["APEX_median_MIC"].min()),
        "best_candidate_sequence": str(scored_candidates.iloc[0]["sequence"]),
    }

    for threshold in [20, 32, 64, 80, 128]:
        median_hits = int((scored_candidates["APEX_median_MIC"] <= threshold).sum())
        worst_hits = int((scored_candidates["APEX_worst_MIC"] <= threshold).sum())
        summary[f"candidates_median_MIC_le_{threshold}"] = median_hits
        summary[f"fraction_median_MIC_le_{threshold}"] = median_hits / max(len(scored_candidates), 1)
        summary[f"candidates_worst_MIC_le_{threshold}"] = worst_hits
        summary[f"fraction_worst_MIC_le_{threshold}"] = worst_hits / max(len(scored_candidates), 1)

    summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 120)
    print("TOP V3 CANDIDATES BY APEX PREDICTED MIC")
    print("=" * 120)
    display_cols = [
        "APEX_rank",
        "apex_candidate_id",
        "candidate_id",
        "sequence",
        "length",
        "net_charge_KR_minus_DE",
        "hydrophobic_fraction",
        "max_train_identity",
        "v3_rank_score",
        "APEX_mean_MIC",
        "APEX_median_MIC",
        "APEX_best_MIC",
        "APEX_worst_MIC",
        "organisms_MIC_le_64",
    ]
    display_cols = [column for column in display_cols if column in scored_candidates.columns]
    print(scored_candidates[display_cols].head(20).round(3).to_string(index=False))

    print("\nOutput files:")
    print(f"  {candidate_output}")
    print(f"  {benchmark_output}")
    print(f"  {combined_output}")
    print(f"  {oracle_compare_output}")
    print(f"  {fasta_output}")
    print(f"  {summary_output}")
    print("\nAPEX MIC scoring for v3 completed successfully.")


if __name__ == "__main__":
    main()
