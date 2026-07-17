#!/usr/bin/env python3
"""Build a final stratified AMP-JEPA V4B panel from elite/Pareto buckets.

Input is the bucket union produced by:
    v4b/11_select_elite_pareto_panels.py

Default quota for a 96-member panel:
    24 hydrophobicity-balanced / Pareto-narrow developability
    20 APEX elite
    16 worst-case robust
    16 broad-spectrum
     8 Pareto-broad
     8 generation-diverse
     4 potency-rescue high-hydrophobicity

The script avoids duplicate sequences and backfills from the global union if a
quota bucket is exhausted. This is a selection/triage layer only; it does not
change scored candidate files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def split_buckets(value: object) -> set[str]:
    if pd.isna(value):
        return set()
    return {x.strip() for x in str(value).split(";") if x.strip()}


def clean_sequence(seq: object) -> str:
    return "".join(str(seq).upper().split())


def sort_panel(df: pd.DataFrame) -> pd.DataFrame:
    sort_cols = []
    ascending = []
    for col, asc in [
        ("v4b_elite_composite_score", False),
        ("APEX_median_MIC", True),
        ("APEX_worst_MIC", True),
        ("organisms_MIC_le_64", False),
        ("score_hydro_balance", False),
    ]:
        if col in df.columns:
            sort_cols.append(col)
            ascending.append(asc)
    if not sort_cols:
        return df.copy()
    return df.sort_values(sort_cols, ascending=ascending, na_position="last").copy()


def pick_from_bucket(
    pool: pd.DataFrame,
    selected_sequences: set[str],
    bucket_names: set[str],
    quota: int,
    label: str,
) -> pd.DataFrame:
    if quota <= 0 or pool.empty:
        return pool.head(0).copy()

    mask = pool["bucket_set"].map(lambda s: bool(s & bucket_names))
    candidates = pool.loc[mask & ~pool["sequence_clean"].isin(selected_sequences)].copy()
    candidates = sort_panel(candidates)
    picked = candidates.head(quota).copy()
    if not picked.empty:
        picked["final_panel_stratum"] = label
        selected_sequences.update(picked["sequence_clean"].astype(str))
    return picked


def write_fasta(df: pd.DataFrame, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for _, row in df.iterrows():
            cid = row.get("candidate_id", "candidate")
            seq = clean_sequence(row.get("sequence", ""))
            stratum = row.get("final_panel_stratum", "selected")
            buckets = row.get("selection_bucket_union", "")
            gen = row.get("generation_source", row.get("generation", "NA"))
            med = row.get("APEX_median_MIC", "NA")
            worst = row.get("APEX_worst_MIC", "NA")
            hydro = row.get("criteria_hydrophobic_fraction", row.get("hydrophobic_fraction", "NA"))
            charge = row.get("criteria_charge", row.get("net_charge_KR_minus_DE", "NA"))
            handle.write(
                f">{cid}|stratum={stratum}|buckets={buckets}|G={gen}|median_MIC={med}|worst_MIC={worst}|hydro={hydro}|charge={charge}\n"
            )
            handle.write(seq + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="v4b/results/elite_pareto_selection/v4b_elite_pareto_bucket_union.csv")
    parser.add_argument("--output-dir", default="v4b/results/final_stratified_panel")
    parser.add_argument("--panel-size", type=int, default=96)
    parser.add_argument("--balanced", type=int, default=None)
    parser.add_argument("--elite", type=int, default=None)
    parser.add_argument("--worst-case", type=int, default=None)
    parser.add_argument("--broad-spectrum", type=int, default=None)
    parser.add_argument("--pareto-broad", type=int, default=None)
    parser.add_argument("--generation-diverse", type=int, default=None)
    parser.add_argument("--potency-rescue", type=int, default=None)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    pool = pd.read_csv(input_path, low_memory=False)
    if "sequence" not in pool.columns:
        raise ValueError("Input lacks sequence column.")
    if "selection_bucket_union" not in pool.columns:
        raise ValueError("Input lacks selection_bucket_union column. Run v4b/11_select_elite_pareto_panels.py first.")

    pool["sequence_clean"] = pool["sequence"].map(clean_sequence)
    pool["bucket_set"] = pool["selection_bucket_union"].map(split_buckets)
    pool = sort_panel(pool.drop_duplicates("sequence_clean", keep="first").reset_index(drop=True))

    # Default quotas are scaled from a 96-member panel.
    scale = args.panel_size / 96.0
    quotas = {
        "balanced_developability": args.balanced if args.balanced is not None else round(24 * scale),
        "apex_elite": args.elite if args.elite is not None else round(20 * scale),
        "worst_case_robust": args.worst_case if args.worst_case is not None else round(16 * scale),
        "broad_spectrum": args.broad_spectrum if args.broad_spectrum is not None else round(16 * scale),
        "pareto_broad": args.pareto_broad if args.pareto_broad is not None else round(8 * scale),
        "generation_diverse": args.generation_diverse if args.generation_diverse is not None else round(8 * scale),
        "potency_rescue_high_hydro": args.potency_rescue if args.potency_rescue is not None else round(4 * scale),
    }

    # Correct rounding drift by adjusting the largest bucket.
    diff = args.panel_size - sum(quotas.values())
    if diff != 0:
        quotas["balanced_developability"] += diff

    selected_sequences: set[str] = set()
    pieces: list[pd.DataFrame] = []

    selection_plan = [
        (
            "balanced_developability",
            {"pareto_narrow_developability", "hydrophobicity_balanced"},
            quotas["balanced_developability"],
        ),
        ("apex_elite", {"apex_elite"}, quotas["apex_elite"]),
        ("worst_case_robust", {"worst_case_robust"}, quotas["worst_case_robust"]),
        ("broad_spectrum", {"broad_spectrum_MIC_le_64"}, quotas["broad_spectrum"]),
        ("pareto_broad", {"pareto_broad_initial_pass"}, quotas["pareto_broad"]),
        ("generation_diverse", {"generation_diverse"}, quotas["generation_diverse"]),
        ("potency_rescue_high_hydro", {"potency_rescue_high_hydro"}, quotas["potency_rescue_high_hydro"]),
    ]

    for label, bucket_names, quota in selection_plan:
        picked = pick_from_bucket(pool, selected_sequences, bucket_names, quota, label)
        pieces.append(picked)

    panel = pd.concat(pieces, ignore_index=True, sort=False) if pieces else pool.head(0).copy()

    # Backfill if overlaps exhausted some quotas.
    if len(panel) < args.panel_size:
        backfill = pool.loc[~pool["sequence_clean"].isin(selected_sequences)].copy()
        backfill = sort_panel(backfill).head(args.panel_size - len(panel)).copy()
        if not backfill.empty:
            backfill["final_panel_stratum"] = "global_backfill"
            panel = pd.concat([panel, backfill], ignore_index=True, sort=False)

    panel = sort_panel(panel).drop_duplicates("sequence_clean", keep="first").head(args.panel_size).reset_index(drop=True)
    panel.insert(0, "final_panel_rank", range(1, len(panel) + 1))

    # Lean version for ordering synthesis/validation discussions.
    key_cols = [
        "final_panel_rank",
        "candidate_id",
        "final_panel_stratum",
        "selection_bucket_union",
        "generation_source",
        "sequence",
        "criteria_length",
        "criteria_charge",
        "criteria_hydrophobic_fraction",
        "APEX_mean_MIC",
        "APEX_median_MIC",
        "APEX_worst_MIC",
        "organisms_MIC_le_64",
        "v4b_elite_composite_score",
    ]
    key_cols = [c for c in key_cols if c in panel.columns]
    lean = panel[key_cols].copy()

    full_path = output_dir / f"v4b_final_stratified_panel_{args.panel_size}.csv"
    lean_path = output_dir / f"v4b_final_stratified_panel_{args.panel_size}_lean.csv"
    fasta_path = output_dir / f"v4b_final_stratified_panel_{args.panel_size}.fasta"
    summary_path = output_dir / f"v4b_final_stratified_panel_{args.panel_size}_summary.json"

    panel.to_csv(full_path, index=False)
    lean.to_csv(lean_path, index=False)
    write_fasta(panel, fasta_path)

    summary = {
        "input": str(input_path),
        "panel_size_requested": int(args.panel_size),
        "panel_size_created": int(len(panel)),
        "quotas": quotas,
        "stratum_counts": panel["final_panel_stratum"].value_counts().to_dict(),
        "bucket_union_counts_in_panel": panel["selection_bucket_union"].value_counts().head(25).to_dict(),
        "hydrophobicity_summary": panel["criteria_hydrophobic_fraction"].describe().to_dict() if "criteria_hydrophobic_fraction" in panel else {},
        "charge_summary": panel["criteria_charge"].describe().to_dict() if "criteria_charge" in panel else {},
        "length_summary": panel["criteria_length"].describe().to_dict() if "criteria_length" in panel else {},
        "outputs": {
            "full_panel": str(full_path),
            "lean_panel": str(lean_path),
            "fasta": str(fasta_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    print("\nFINAL STRATIFIED PANEL SUMMARY")
    print(json.dumps(summary, indent=2, default=str))
    print("\nTOP PANEL CANDIDATES")
    print(lean.head(40).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
