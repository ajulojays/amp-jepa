#!/usr/bin/env python3
"""Score v3-generated AMP-JEPA candidates with the APEX MIC ensemble.

This restores the old working AMP-JEPA discovery loop:

    AMP-JEPA/v3 candidates -> APEX ensemble -> predicted MIC summaries

The pasted APEX/ApexOracle MIC table is kept as a benchmark only. It is not
used to assign MIC values to v3 candidates.
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
APEX_ROOT = Path(os.environ.get("APEX_ROOT", "/home/julojays/apex"))

sys.path.insert(0, str(APEX_ROOT))

try:
    from apex_utils.apex_direct_ensemble import load_apex_ensemble, predict_mic_ensemble
except Exception as exc:  # pragma: no cover - helpful runtime error for local use
    raise SystemExit(
        "[ERROR] Could not import APEX utilities. Set APEX_ROOT to your APEX repo, "
        "for example: export APEX_ROOT=/home/julojays/apex\n"
        f"Original import error: {exc}"
    )


METADATA_EXCLUDE = {
    "APEX_rank",
    "v3_rank",
    "candidate_id",
    "apex_candidate_id",
    "record_type",
    "comparison_note",
    "sequence",
    "Sequence",
    "PeptideSequence",
    "peptide",
    "Peptide",
    "strategy",
    "source_a",
    "source_b",
    "alpha",
    "generation_round",
    "length",
    "ended_with_end_token",
    "valid_basic",
    "duplicate_generated",
    "net_charge",
    "net_charge_KR_minus_DE",
    "hydrophobic_fraction",
    "mean_hydropathy",
    "entropy",
    "maximum_residue_fraction",
    "longest_homopolymer",
    "max_train_identity",
    "nearest_apd_similarity",
    "nearest_apd_sequence",
    "novelty_score",
    "developability_score",
    "pre_apex_score",
    "v3_rank_score",
    "passes_v3_filters",
    "screen_rank",
    "nearest_selected_similarity",
    "min_pred_MIC",
    "median_pred_MIC",
    "median_all_pred_MIC",
    "mean_all_pred_MIC",
    "mean_GN_pred_MIC",
    "mean_GP_pred_MIC",
    "min_GN_pred_MIC",
    "median_GN_pred_MIC",
    "min_GP_pred_MIC",
    "median_GP_pred_MIC",
    "n_pathogens_pred_MIC_le_20",
    "n_pathogens_pred_MIC_le_32",
    "n_pathogens_pred_MIC_le_64",
    "n_pathogens_pred_MIC_le_80",
    "n_pathogens_pred_MIC_le_128",
    "APEX_mean_MIC",
    "APEX_median_MIC",
    "APEX_best_MIC",
    "APEX_worst_MIC",
    "APEX_MIC_std",
}

SUMMARY_PREFIXES = (
    "APEX_",
    "organisms_MIC_",
    "fraction_MIC_",
    "key_organisms_",
    "n_pathogens_",
)


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def clean_sequence(value: object) -> str:
    allowed = set("ACDEFGHIKLMNPQRSTVWY")
    seq = str(value).strip().upper()
    return "".join(res for res in seq if res in allowed)


def ensure_dataframe(predictions) -> pd.DataFrame:
    if isinstance(predictions, pd.DataFrame):
        return predictions.copy()
    return pd.DataFrame(predictions)


def identify_numeric_mic_columns(df: pd.DataFrame) -> List[str]:
    mic_columns: List[str] = []
    for col in df.columns:
        if col in METADATA_EXCLUDE:
            continue
        if any(str(col).startswith(prefix) for prefix in SUMMARY_PREFIXES):
            continue
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().any():
            mic_columns.append(col)
    return mic_columns


def add_apex_summary_metrics(df: pd.DataFrame, mic_columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    mic_columns = list(mic_columns)
    if not mic_columns:
        return out

    numeric_predictions = out[mic_columns].apply(pd.to_numeric, errors="coerce")

    out["APEX_mean_MIC"] = numeric_predictions.mean(axis=1, skipna=True)
    out["APEX_median_MIC"] = numeric_predictions.median(axis=1, skipna=True)
    out["APEX_best_MIC"] = numeric_predictions.min(axis=1, skipna=True)
    out["APEX_worst_MIC"] = numeric_predictions.max(axis=1, skipna=True)
    out["APEX_MIC_std"] = numeric_predictions.std(axis=1, skipna=True)

    for threshold in [20, 32, 64, 80, 128]:
        out[f"organisms_MIC_le_{threshold}"] = numeric_predictions.le(threshold).sum(axis=1)
        out[f"fraction_MIC_le_{threshold}"] = numeric_predictions.le(threshold).mean(axis=1)

    # Compatibility with the comparator vocabulary used earlier in v3.
    out["min_pred_MIC"] = out["APEX_best_MIC"]
    out["median_pred_MIC"] = out["APEX_median_MIC"]
    out["mean_all_pred_MIC"] = out["APEX_mean_MIC"]
    out["median_all_pred_MIC"] = out["APEX_median_MIC"]
    out["n_pathogens_pred_MIC_le_32"] = out["organisms_MIC_le_32"]
    out["n_pathogens_pred_MIC_le_64"] = out["organisms_MIC_le_64"]
    out["n_pathogens_pred_MIC_le_128"] = out["organisms_MIC_le_128"]

    return out


def read_candidate_table(path: Path, max_candidates: int | None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Candidate table not found: {path}")

    df = pd.read_csv(path)
    if "sequence" not in df.columns:
        raise ValueError("Candidate table must contain a sequence column")

    df = df.copy()
    df["sequence"] = df["sequence"].map(clean_sequence)
    df = df[df["sequence"].str.len() > 0].drop_duplicates("sequence").reset_index(drop=True)

    if "candidate_id" not in df.columns:
        df.insert(0, "candidate_id", [f"v3_candidate_{i + 1:06d}" for i in range(len(df))])

    if max_candidates is not None and max_candidates > 0:
        df = df.head(max_candidates).copy()

    return df.reset_index(drop=True)


def read_benchmark_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Benchmark APEX table not found: {path}")

    df = pd.read_csv(path).copy()

    if "PeptideSequence" in df.columns:
        seq_col = "PeptideSequence"
    elif "sequence" in df.columns:
        seq_col = "sequence"
    elif "Sequence" in df.columns:
        seq_col = "Sequence"
    else:
        raise ValueError("Benchmark table needs PeptideSequence, sequence, or Sequence column")

    df["sequence"] = df[seq_col].map(clean_sequence)
    df = df[df["sequence"].str.len() > 0].drop_duplicates("sequence").reset_index(drop=True)

    if "candidate_id" not in df.columns:
        if "Unnamed: 0" in df.columns:
            df.insert(0, "candidate_id", df["Unnamed: 0"].astype(str))
        else:
            df.insert(0, "candidate_id", [f"apex_benchmark_{i + 1:04d}" for i in range(len(df))])

    mic_columns = identify_numeric_mic_columns(df)
    df = add_apex_summary_metrics(df, mic_columns)
    df["record_type"] = "apex_benchmark_pasted"

    if "apex_rank" in df.columns:
        df = df.drop(columns=["apex_rank"])
    sort_col = "mean_GN_pred_MIC" if "mean_GN_pred_MIC" in df.columns else "APEX_mean_MIC"
    if sort_col in df.columns:
        df = df.sort_values(sort_col, ascending=True).reset_index(drop=True)
    df.insert(0, "apex_rank", range(1, len(df) + 1))

    return df


def load_apex_models(model_dir: Path, pattern: str, device: torch.device):
    if not model_dir.exists():
        raise FileNotFoundError(f"APEX model directory not found: {model_dir}")

    print(f"[INFO] Loading APEX ensemble from {model_dir}")
    models = load_apex_ensemble(model_dir=str(model_dir), pattern=pattern, device=device)
    if len(models) == 0:
        raise RuntimeError("No APEX models were loaded")
    print(f"[INFO] Loaded {len(models)} APEX models")
    return models


def score_v3_candidates(
    candidates: pd.DataFrame,
    apex_models,
    device: torch.device,
) -> tuple[pd.DataFrame, list[str]]:
    sequences = candidates["sequence"].tolist()
    print(f"[INFO] Scoring {len(sequences)} v3 candidates with APEX")
    predictions = predict_mic_ensemble(apex_models, sequences, device=device)
    predictions = ensure_dataframe(predictions).reset_index(drop=True)

    for col in ["sequence", "Sequence", "PeptideSequence", "peptide", "Peptide"]:
        if col in predictions.columns:
            predictions = predictions.drop(columns=[col])

    scored = pd.concat([candidates.reset_index(drop=True), predictions], axis=1)
    mic_columns = identify_numeric_mic_columns(predictions)
    if not mic_columns:
        raise RuntimeError("No numeric APEX organism prediction columns were detected")

    print(f"[INFO] Detected {len(mic_columns)} APEX organism prediction columns")
    scored = add_apex_summary_metrics(scored, mic_columns)
    scored = scored.sort_values(
        by=["APEX_median_MIC", "APEX_worst_MIC", "APEX_mean_MIC"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    scored.insert(0, "APEX_rank", range(1, len(scored) + 1))
    scored["record_type"] = "v3_generated_apex_scored"
    return scored, mic_columns


def write_fasta(df: pd.DataFrame, path: Path, n: int) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for _, row in df.head(n).iterrows():
            candidate_id = row.get("candidate_id", row.get("apex_candidate_id", "candidate"))
            handle.write(
                f">{candidate_id}|APEX_median_MIC={row.get('APEX_median_MIC', np.nan):.3f}"
                f"|APEX_mean_MIC={row.get('APEX_mean_MIC', np.nan):.3f}\n"
            )
            handle.write(f"{row['sequence']}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", default="v3/results/top_panel_v3.csv")
    parser.add_argument("--benchmark-apex", default="v3/data/external/apex_oracle_ranked_summary.csv")
    parser.add_argument("--output-dir", default="v3/results/apex_scored_v3")
    parser.add_argument("--apex-root", default=str(APEX_ROOT))
    parser.add_argument("--apex-model-dir", default="")
    parser.add_argument("--apex-pattern", default="trained_all_model_*_ensemble_*")
    parser.add_argument("--max-candidates", type=int, default=50)
    parser.add_argument("--top-fasta-count", type=int, default=10)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    apex_root = Path(args.apex_root)
    model_dir = Path(args.apex_model_dir) if args.apex_model_dir else apex_root / "trained_models"
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = read_candidate_table(resolve_path(args.candidates), args.max_candidates)
    benchmarks = read_benchmark_table(resolve_path(args.benchmark_apex))
    apex_models = load_apex_models(model_dir=model_dir, pattern=args.apex_pattern, device=device)

    scored_candidates, mic_columns = score_v3_candidates(candidates, apex_models, device)

    candidate_output = output_dir / "v3_apex_scored_candidates.csv"
    benchmark_output = output_dir / "apex_pasted_benchmark_summary.csv"
    combined_output = output_dir / "v3_vs_apex_benchmark_mic_comparison.csv"
    top_fasta_output = output_dir / "v3_apex_top_candidates.fasta"
    summary_output = output_dir / "v3_apex_scoring_summary.json"

    scored_candidates.to_csv(candidate_output, index=False)
    benchmarks.to_csv(benchmark_output, index=False)

    shared_columns = sorted(set(scored_candidates.columns).union(set(benchmarks.columns)))
    combined = pd.concat(
        [
            scored_candidates.reindex(columns=shared_columns),
            benchmarks.reindex(columns=shared_columns),
        ],
        ignore_index=True,
        sort=False,
    )
    combined = combined.sort_values(
        by=["APEX_median_MIC", "APEX_worst_MIC", "APEX_mean_MIC"],
        ascending=[True, True, True],
        na_position="last",
    ).reset_index(drop=True)
    combined.insert(0, "combined_MIC_rank", range(1, len(combined) + 1))
    combined.to_csv(combined_output, index=False)

    write_fasta(scored_candidates, top_fasta_output, args.top_fasta_count)

    summary = {
        "candidate_input": str(resolve_path(args.candidates)),
        "benchmark_input": str(resolve_path(args.benchmark_apex)),
        "number_v3_candidates_scored": int(len(scored_candidates)),
        "number_pasted_apex_benchmarks": int(len(benchmarks)),
        "number_apex_models": int(len(apex_models)),
        "number_apex_organism_columns": int(len(mic_columns)),
        "median_of_v3_APEX_median_MIC": float(scored_candidates["APEX_median_MIC"].median()),
        "best_v3_APEX_median_MIC": float(scored_candidates["APEX_median_MIC"].min()),
        "best_v3_APEX_mean_MIC": float(scored_candidates["APEX_mean_MIC"].min()),
        "best_benchmark_APEX_median_MIC": float(benchmarks["APEX_median_MIC"].min()) if "APEX_median_MIC" in benchmarks else None,
    }
    for threshold in [20, 32, 64, 80, 128]:
        summary[f"v3_candidates_median_MIC_le_{threshold}"] = int((scored_candidates["APEX_median_MIC"] <= threshold).sum())
        summary[f"v3_candidates_worst_MIC_le_{threshold}"] = int((scored_candidates["APEX_worst_MIC"] <= threshold).sum())

    summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    display_cols = [
        "combined_MIC_rank",
        "record_type",
        "APEX_rank",
        "apex_rank",
        "candidate_id",
        "sequence",
        "APEX_best_MIC",
        "APEX_median_MIC",
        "APEX_mean_MIC",
        "APEX_worst_MIC",
        "organisms_MIC_le_64",
        "v3_rank",
        "v3_rank_score",
    ]
    display_cols = [c for c in display_cols if c in combined.columns]

    print("\n" + "=" * 120)
    print("V3 CANDIDATES SCORED BY APEX + PASTED APEX TABLE AS BENCHMARK")
    print("=" * 120)
    print(combined[display_cols].head(25).round(3).to_string(index=False))

    print("\nOutput files:")
    print(f"  {candidate_output}")
    print(f"  {benchmark_output}")
    print(f"  {combined_output}")
    print(f"  {top_fasta_output}")
    print(f"  {summary_output}")
    print("\n[DONE] APEX MIC scoring completed.")


if __name__ == "__main__":
    main()
