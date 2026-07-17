#!/usr/bin/env python3
"""Select a nonredundant Novel+MIC32 V4B candidate set.

This script starts from the De la Fuente-style preliminary pool:

    Tier 1-4 + Layer 1 novelty pass + APEX_median_MIC <= 32

and applies the third rule: selected peptides should be <75% similar to each
other. It uses a fast global edit-identity precheck by default:

    similarity = 1 - Levenshtein_distance(seq1, seq2) / max(len(seq1), len(seq2))

This is a conservative practical screen for short AMP candidates and is meant to
produce a stable nonredundant pool for downstream EMBOSS needleall/Needleman-
Wunsch certification.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

CANONICAL = set("ACDEFGHIKLMNPQRSTVWY")
TIER_ORDER = {
    "Tier1_core_frontier": 1,
    "Tier2_balanced_elite": 2,
    "Tier3_robust_potency_reserve": 3,
    "Tier4_exploration_reserve": 4,
}


def clean_sequence(x: object) -> str:
    return re.sub(r"\s+", "", str(x).upper())


def choose_sequence_column(df: pd.DataFrame) -> str:
    for col in ["sequence_clean", "sequence", "Sequence", "peptide_sequence", "seq"]:
        if col in df.columns:
            return col
    raise ValueError(f"Could not find sequence column. Columns: {list(df.columns)}")


def numeric(df: pd.DataFrame, col: str, default: float) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.full(len(df), default), index=df.index)
    s = pd.to_numeric(df[col], errors="coerce")
    if s.notna().sum() == 0:
        return pd.Series(np.full(len(df), default), index=df.index)
    return s.fillna(float(s.median()))


def edit_distance(a: str, b: str) -> int:
    """Memory-light Levenshtein distance for short peptides."""
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    # len(a) >= len(b); keep the shorter row in memory.
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        prev_diag = i - 1
        for j, cb in enumerate(b, 1):
            old = previous[j]
            cost = 0 if ca == cb else 1
            current.append(min(
                previous[j] + 1,      # deletion
                current[j - 1] + 1,   # insertion
                prev_diag + cost,     # substitution/match
            ))
            prev_diag = old
        previous = current
    return previous[-1]


def global_edit_identity(a: str, b: str) -> float:
    """Global edit identity on [0,1], using max sequence length as denominator."""
    a = clean_sequence(a)
    b = clean_sequence(b)
    denom = max(len(a), len(b), 1)
    # Fast upper bound: even with perfect containment, identity cannot exceed min/max.
    if min(len(a), len(b)) / denom < 0.0:
        return 0.0
    d = edit_distance(a, b)
    return max(0.0, 1.0 - (d / denom))


def hydrophobic_balance_score(values: Iterable[float], target: float = 0.52, half_width: float = 0.18) -> np.ndarray:
    x = pd.to_numeric(pd.Series(values), errors="coerce").astype(float).to_numpy()
    x = np.nan_to_num(x, nan=target, posinf=target, neginf=target)
    return np.clip(1.0 - (np.abs(x - target) / max(half_width, 1e-9)), 0.0, 1.0)


def build_ranked_input(df: pd.DataFrame, mic_cutoff: float) -> pd.DataFrame:
    seq_col = choose_sequence_column(df)
    out = df.copy()
    out["sequence_clean"] = out[seq_col].map(clean_sequence)
    out = out[out["sequence_clean"].map(lambda s: bool(s) and all(a in CANONICAL for a in s))].copy()
    out["sequence_length"] = out["sequence_clean"].str.len()

    out["APEX_median_MIC"] = numeric(out, "APEX_median_MIC", 9999.0)
    out["APEX_worst_MIC"] = numeric(out, "APEX_worst_MIC", 9999.0)
    out["APEX_mean_MIC"] = numeric(out, "APEX_mean_MIC", 9999.0)
    out["organisms_MIC_le_64"] = numeric(out, "organisms_MIC_le_64", 0.0)

    if "criteria_hydrophobic_fraction" in out.columns:
        out["hydro_balance_score"] = hydrophobic_balance_score(out["criteria_hydrophobic_fraction"])
    else:
        out["hydro_balance_score"] = 0.0

    if "lead_tier" in out.columns:
        out["tier_priority"] = out["lead_tier"].map(TIER_ORDER).fillna(99).astype(int)
    else:
        out["tier_priority"] = 99

    if "layer1_manifest75_novelty_class" in out.columns:
        out = out[out["layer1_manifest75_novelty_class"].eq("passes_broad_and_qc_75")].copy()

    out = out[out["APEX_median_MIC"] <= float(mic_cutoff)].copy()

    # Keep one row per sequence, choosing the strongest record.
    out = out.sort_values(
        ["tier_priority", "APEX_median_MIC", "APEX_worst_MIC", "APEX_mean_MIC", "organisms_MIC_le_64", "hydro_balance_score"],
        ascending=[True, True, True, True, False, False],
        na_position="last",
    )
    out = out.drop_duplicates("sequence_clean", keep="first").reset_index(drop=True)
    out["pre_self_similarity_rank"] = np.arange(1, len(out) + 1)
    return out


def greedy_nonredundant(df: pd.DataFrame, threshold: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    selected_rows: list[pd.Series] = []
    selected_seqs: list[str] = []
    rejected_rows: list[dict] = []
    pair_rows: list[dict] = []

    for _, row in df.iterrows():
        seq = str(row["sequence_clean"])
        cid = str(row.get("candidate_id", f"candidate_{len(selected_rows)+len(rejected_rows)+1}"))
        best_sim = -1.0
        best_cid = None
        best_seq = None
        rejected = False

        for keep_row, keep_seq in zip(selected_rows, selected_seqs):
            denom = max(len(seq), len(keep_seq), 1)
            # Upper bound: if lengths alone make >=threshold impossible, skip exact distance.
            if min(len(seq), len(keep_seq)) / denom < threshold:
                sim = min(len(seq), len(keep_seq)) / denom
            else:
                sim = global_edit_identity(seq, keep_seq)

            if sim > best_sim:
                best_sim = sim
                best_cid = str(keep_row.get("candidate_id", "selected"))
                best_seq = keep_seq

            if sim >= threshold:
                rejected = True
                break

        if not selected_rows:
            best_sim = np.nan

        if rejected:
            rejected_row = row.to_dict()
            rejected_row.update({
                "self_similarity_status": "rejected_self_ge_threshold",
                "nearest_selected_candidate_id": best_cid,
                "nearest_selected_sequence": best_seq,
                "nearest_selected_similarity": best_sim,
            })
            rejected_rows.append(rejected_row)
            pair_rows.append({
                "candidate_id": cid,
                "sequence_clean": seq,
                "nearest_selected_candidate_id": best_cid,
                "nearest_selected_sequence": best_seq,
                "nearest_selected_similarity": best_sim,
                "status": "rejected_self_ge_threshold",
            })
        else:
            selected_rows.append(row)
            selected_seqs.append(seq)
            pair_rows.append({
                "candidate_id": cid,
                "sequence_clean": seq,
                "nearest_selected_candidate_id": best_cid,
                "nearest_selected_sequence": best_seq,
                "nearest_selected_similarity": best_sim,
                "status": "selected_self_lt_threshold",
            })

    selected = pd.DataFrame([r.to_dict() for r in selected_rows])
    rejected = pd.DataFrame(rejected_rows)
    nearest = pd.DataFrame(pair_rows)

    if not selected.empty:
        selected["self_similarity_status"] = "selected_self_lt_threshold"
        selected["self_similarity_rank"] = np.arange(1, len(selected) + 1)
    return selected, rejected, nearest


def write_fasta(df: pd.DataFrame, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for _, row in df.iterrows():
            cid = row.get("candidate_id", "candidate")
            seq = row.get("sequence_clean", row.get("sequence", ""))
            tier = row.get("lead_tier", "NA")
            med = row.get("APEX_median_MIC", "NA")
            worst = row.get("APEX_worst_MIC", "NA")
            handle.write(f">{cid}|tier={tier}|median_MIC={med}|worst_MIC={worst}\n{seq}\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="v4b/results/novelty_layer1_manifest75_tiered/v4b_tier1_4_novel_MIC32_candidates.csv")
    p.add_argument("--output-dir", default="v4b/results/novelty_nonredundant_mic32")
    p.add_argument("--mic-cutoff", type=float, default=32.0)
    p.add_argument("--self-similarity-threshold", type=float, default=0.75)
    p.add_argument("--write-fasta", action="store_true")
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(args.input, low_memory=False)
    ranked = build_ranked_input(raw, mic_cutoff=args.mic_cutoff)
    selected, rejected, nearest = greedy_nonredundant(ranked, threshold=args.self_similarity_threshold)

    ranked.to_csv(output_dir / "v4b_novel_MIC32_ranked_before_self_similarity.csv", index=False)
    selected.to_csv(output_dir / "v4b_novel_MIC32_self_nonredundant_lt75.csv", index=False)
    rejected.to_csv(output_dir / "v4b_novel_MIC32_rejected_self_ge75.csv", index=False)
    nearest.to_csv(output_dir / "v4b_novel_MIC32_nearest_selected_similarity.csv", index=False)

    if "lead_tier" in selected.columns:
        tier_summary = selected.groupby("lead_tier").size().reset_index(name="selected_self_nonredundant_count")
        tier_summary.to_csv(output_dir / "v4b_novel_MIC32_self_nonredundant_summary_by_tier.csv", index=False)
    else:
        tier_summary = pd.DataFrame()

    if args.write_fasta:
        write_fasta(selected, output_dir / "v4b_novel_MIC32_self_nonredundant_lt75.fasta")
        write_fasta(rejected, output_dir / "v4b_novel_MIC32_rejected_self_ge75.fasta")

    payload = {
        "input": str(args.input),
        "mic_cutoff": float(args.mic_cutoff),
        "self_similarity_threshold": float(args.self_similarity_threshold),
        "similarity_method": "global_edit_identity = 1 - levenshtein_distance/max_length",
        "note": "Use this as a fast self-redundancy screen before final EMBOSS needleall certification.",
        "input_rows": int(len(raw)),
        "ranked_novel_mic32_unique_sequences": int(len(ranked)),
        "selected_self_nonredundant_lt_threshold": int(len(selected)),
        "rejected_self_ge_threshold": int(len(rejected)),
        "outputs": {
            "ranked_input": str(output_dir / "v4b_novel_MIC32_ranked_before_self_similarity.csv"),
            "selected": str(output_dir / "v4b_novel_MIC32_self_nonredundant_lt75.csv"),
            "rejected": str(output_dir / "v4b_novel_MIC32_rejected_self_ge75.csv"),
            "nearest_similarity": str(output_dir / "v4b_novel_MIC32_nearest_selected_similarity.csv"),
        },
    }
    (output_dir / "v4b_novel_MIC32_self_nonredundant_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\nSELF-SIMILARITY NONREDUNDANT SELECTION SUMMARY")
    print(json.dumps(payload, indent=2))
    if not tier_summary.empty:
        print("\nSELECTED BY TIER")
        print(tier_summary.to_string(index=False))
    if not selected.empty:
        show = [
            "candidate_id", "lead_tier", "generation_source", "sequence_clean",
            "APEX_median_MIC", "APEX_worst_MIC", "APEX_mean_MIC", "organisms_MIC_le_64",
            "criteria_length", "criteria_charge", "criteria_hydrophobic_fraction", "self_similarity_rank",
        ]
        show = [c for c in show if c in selected.columns]
        print("\nTOP SELECTED SELF-NONREDUNDANT CANDIDATES")
        print(selected[show].head(40).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
