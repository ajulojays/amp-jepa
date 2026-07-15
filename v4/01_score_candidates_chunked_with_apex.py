#!/usr/bin/env python3
"""Chunked APEX scoring for AMP-JEPA-Hybrid V4A.

The original v3 APEX scorer is correct, but it sends the whole candidate table
through the APEX ensemble at once. For V4A full-scale pools, that can overflow
smaller GPUs. This wrapper reuses the proven v3 scorer functions while scoring
candidate chunks sequentially.

Input:
    CSV with a sequence/Sequence/PeptideSequence column.

Outputs:
    apex_scored_v3_candidates.csv
    apex_scoring_summary.json
    apex_top_v3_candidates.fasta

The output filename intentionally matches the v3 scorer so downstream V4A steps
can use the same paths.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import List

import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
V3_SCORER_PATH = PROJECT_ROOT / "v3" / "25_score_v3_candidates_with_apex.py"


def load_v3_scorer():
    spec = importlib.util.spec_from_file_location("v3_apex_scorer", V3_SCORER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import v3 scorer from {V3_SCORER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["v3_apex_scorer"] = module
    spec.loader.exec_module(module)
    return module


def write_fasta(df: pd.DataFrame, path: Path, top_n: int) -> None:
    top = df.head(top_n).copy()
    with path.open("w", encoding="utf-8") as handle:
        for _, row in top.iterrows():
            handle.write(
                f">{row.get('apex_candidate_id', row.get('candidate_id', 'V4A'))}"
                f"|APEX_median_MIC={row.get('APEX_median_MIC', float('nan')):.2f}"
                f"|APEX_mean_MIC={row.get('APEX_mean_MIC', float('nan')):.2f}\n"
            )
            handle.write(f"{row['sequence']}\n")


def score_with_adaptive_batches(
    scorer,
    candidates: pd.DataFrame,
    apex_models,
    predict_mic_ensemble,
    device: torch.device,
    initial_batch_size: int,
    min_batch_size: int,
) -> tuple[pd.DataFrame, List[str], list[dict]]:
    pieces: list[pd.DataFrame] = []
    all_mic_columns: List[str] = []
    logs: list[dict] = []

    n = len(candidates)
    start = 0
    batch_size = max(1, int(initial_batch_size))
    min_batch_size = max(1, int(min_batch_size))

    while start < n:
        current_batch_size = min(batch_size, n - start)
        batch = candidates.iloc[start : start + current_batch_size].copy().reset_index(drop=True)

        try:
            print(f"Scoring rows {start + 1:,}-{start + current_batch_size:,} / {n:,} with batch_size={current_batch_size}")
            scored, mic_columns = scorer.score_table_with_apex(
                batch,
                apex_models,
                predict_mic_ensemble,
                device,
            )
            pieces.append(scored)
            all_mic_columns = sorted(set(all_mic_columns).union(mic_columns))
            logs.append({"start": start, "end": start + current_batch_size, "batch_size": current_batch_size, "status": "ok"})
            start += current_batch_size

            if device.type == "cuda":
                torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if current_batch_size <= min_batch_size:
                raise
            new_batch_size = max(min_batch_size, current_batch_size // 2)
            print(
                f"[WARN] CUDA OOM at batch_size={current_batch_size}. "
                f"Retrying same position with batch_size={new_batch_size}."
            )
            logs.append({"start": start, "end": start + current_batch_size, "batch_size": current_batch_size, "status": "oom_retry"})
            batch_size = new_batch_size

    if not pieces:
        raise RuntimeError("No batches were scored.")

    scored_all = pd.concat(pieces, ignore_index=True, sort=False)
    return scored_all, all_mic_columns, logs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--apex-root", default=os.environ.get("APEX_ROOT", "/home/julojays/apex"))
    parser.add_argument("--apex-pattern", default="trained_all_model_*_ensemble_*")
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("APEX_BATCH_SIZE", "64")))
    parser.add_argument("--min-batch-size", type=int, default=int(os.environ.get("APEX_MIN_BATCH_SIZE", "8")))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--top-fasta-count", type=int, default=50)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    args = parser.parse_args()

    scorer = load_v3_scorer()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    candidate_path = scorer.resolve_path(args.candidates)
    output_dir = scorer.resolve_path(args.output_dir)
    apex_root = scorer.resolve_path(args.apex_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = scorer.prepare_candidate_table(candidate_path, limit=args.limit if args.limit > 0 else None)
    print(f"Loaded {len(candidates):,} unique candidates for chunked APEX scoring.")
    print(f"Using device: {device}")
    print(f"Initial APEX batch size: {args.batch_size}")

    apex_models, predict_mic_ensemble = scorer.load_apex(apex_root, args.apex_pattern, device)
    print(f"Loaded {len(apex_models)} APEX models from {apex_root / 'trained_models'}")

    scored_candidates, mic_columns, batch_logs = score_with_adaptive_batches(
        scorer=scorer,
        candidates=candidates,
        apex_models=apex_models,
        predict_mic_ensemble=predict_mic_ensemble,
        device=device,
        initial_batch_size=args.batch_size,
        min_batch_size=args.min_batch_size,
    )

    scored_candidates["record_type"] = "v4a_generated_or_variant"
    sort_cols = [c for c in ["APEX_median_MIC", "APEX_worst_MIC", "APEX_mean_MIC"] if c in scored_candidates.columns]
    if sort_cols:
        scored_candidates = scored_candidates.sort_values(sort_cols, ascending=True, na_position="last").reset_index(drop=True)
    scored_candidates.insert(0, "APEX_rank", range(1, len(scored_candidates) + 1))

    candidate_output = output_dir / "apex_scored_v3_candidates.csv"
    fasta_output = output_dir / "apex_top_v3_candidates.fasta"
    summary_output = output_dir / "apex_scoring_summary.json"
    batch_log_output = output_dir / "apex_batch_logs.json"

    scored_candidates.to_csv(candidate_output, index=False)
    write_fasta(scored_candidates, fasta_output, args.top_fasta_count)

    summary = {
        "candidate_input": str(candidate_path),
        "number_candidates_scored": int(len(scored_candidates)),
        "number_apex_models": int(len(apex_models)),
        "number_organisms_scored": int(len(mic_columns)),
        "organism_columns": mic_columns,
        "device": str(device),
        "initial_batch_size": int(args.batch_size),
        "min_batch_size": int(args.min_batch_size),
        "batches_attempted": len(batch_logs),
        "mean_of_candidate_mean_MIC": float(scored_candidates["APEX_mean_MIC"].mean()) if "APEX_mean_MIC" in scored_candidates else None,
        "median_of_candidate_median_MIC": float(scored_candidates["APEX_median_MIC"].median()) if "APEX_median_MIC" in scored_candidates else None,
        "best_candidate_median_MIC": float(scored_candidates["APEX_median_MIC"].min()) if "APEX_median_MIC" in scored_candidates else None,
        "best_candidate_sequence": str(scored_candidates.iloc[0]["sequence"]) if len(scored_candidates) else None,
    }

    if "APEX_median_MIC" in scored_candidates and "APEX_worst_MIC" in scored_candidates:
        for threshold in [20, 32, 64, 80, 128]:
            median_hits = int((scored_candidates["APEX_median_MIC"] <= threshold).sum())
            worst_hits = int((scored_candidates["APEX_worst_MIC"] <= threshold).sum())
            summary[f"candidates_median_MIC_le_{threshold}"] = median_hits
            summary[f"fraction_median_MIC_le_{threshold}"] = median_hits / max(len(scored_candidates), 1)
            summary[f"candidates_worst_MIC_le_{threshold}"] = worst_hits
            summary[f"fraction_worst_MIC_le_{threshold}"] = worst_hits / max(len(scored_candidates), 1)

    summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    batch_log_output.write_text(json.dumps(batch_logs, indent=2), encoding="utf-8")

    print("\n" + "=" * 120)
    print("TOP CHUNKED APEX-SCORED V4A CANDIDATES")
    print("=" * 120)
    display_cols = [
        "APEX_rank", "candidate_id", "sequence", "length",
        "net_charge_KR_minus_DE", "hydrophobic_fraction", "v4a_class",
        "APEX_mean_MIC", "APEX_median_MIC", "APEX_worst_MIC", "organisms_MIC_le_64",
    ]
    display_cols = [c for c in display_cols if c in scored_candidates.columns]
    print(scored_candidates[display_cols].head(20).round(3).to_string(index=False))

    print("\nOutput files:")
    print(f"  {candidate_output}")
    print(f"  {fasta_output}")
    print(f"  {summary_output}")
    print(f"  {batch_log_output}")
    print("\nChunked APEX scoring completed successfully.")


if __name__ == "__main__":
    main()
