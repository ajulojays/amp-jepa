#!/usr/bin/env python3
"""Compare optimized V4A variants against their parent candidates.

Input is the APEX-scored optimized variant table produced by the existing v3 APEX
scorer. The variant metadata is preserved because the scorer keeps candidate
metadata columns.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parent-landscape", default="v4/results/landscape/candidate_landscape.csv")
    ap.add_argument("--scored-variants", default="v4/results/optimization/apex_optimized_scoring/apex_scored_v3_candidates.csv")
    ap.add_argument("--output", default="v4/results/optimization/optimized_variants_with_gains.csv")
    ap.add_argument("--g-success-output", default="v4/results/rescue/g_rescue_successes.csv")
    args = ap.parse_args()

    parents = pd.read_csv(args.parent_landscape, low_memory=False)
    variants = pd.read_csv(args.scored_variants, low_memory=False)

    need = ["sequence", "APEX_median_MIC", "APEX_mean_MIC", "APEX_worst_MIC", "organisms_MIC_le_64", "v4a_landscape_score", "v4a_class"]
    for c in need:
        if c not in parents.columns:
            parents[c] = np.nan
    parent_small = parents[need].rename(columns={
        "sequence": "parent_sequence",
        "APEX_median_MIC": "parent_APEX_median_MIC",
        "APEX_mean_MIC": "parent_APEX_mean_MIC",
        "APEX_worst_MIC": "parent_APEX_worst_MIC",
        "organisms_MIC_le_64": "parent_organisms_MIC_le_64",
        "v4a_landscape_score": "parent_v4a_landscape_score",
        "v4a_class": "parent_v4a_class_from_landscape",
    })

    if "parent_sequence" not in variants.columns:
        raise ValueError("Scored variants need parent_sequence metadata. Re-run 03 optimizer then APEX scorer.")

    out = variants.merge(parent_small, on="parent_sequence", how="left")

    for c in [
        "APEX_median_MIC", "APEX_mean_MIC", "APEX_worst_MIC", "organisms_MIC_le_64",
        "parent_APEX_median_MIC", "parent_APEX_mean_MIC", "parent_APEX_worst_MIC", "parent_organisms_MIC_le_64"
    ]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out["delta_median_MIC"] = out["parent_APEX_median_MIC"] - out["APEX_median_MIC"]
    out["delta_mean_MIC"] = out["parent_APEX_mean_MIC"] - out["APEX_mean_MIC"]
    out["delta_worst_MIC"] = out["parent_APEX_worst_MIC"] - out["APEX_worst_MIC"]
    out["delta_organisms_MIC_le_64"] = out["organisms_MIC_le_64"] - out["parent_organisms_MIC_le_64"]

    out["improved_median"] = out["delta_median_MIC"] > 0
    out["improved_worst"] = out["delta_worst_MIC"] > 0
    out["improved_breadth"] = out["delta_organisms_MIC_le_64"] > 0
    out["v4a_optimization_success"] = (
        out["improved_median"].fillna(False)
        | out["improved_worst"].fillna(False)
        | out["improved_breadth"].fillna(False)
    )

    out["v4a_gain_score"] = (
        out["delta_median_MIC"].fillna(0).clip(lower=-100, upper=100) / 100
        + out["delta_worst_MIC"].fillna(0).clip(lower=-300, upper=300) / 300
        + out["delta_organisms_MIC_le_64"].fillna(0).clip(lower=-10, upper=10) / 10
    )

    out = out.sort_values(
        ["v4a_optimization_success", "v4a_gain_score", "APEX_median_MIC", "APEX_worst_MIC"],
        ascending=[False, False, True, True]
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    g_success = out[(out.get("parent_v4a_class", out.get("parent_v4a_class_from_landscape", "")) == "G_rescue") & out["v4a_optimization_success"]].copy()
    g_path = Path(args.g_success_output)
    g_path.parent.mkdir(parents=True, exist_ok=True)
    g_success.to_csv(g_path, index=False)

    summary = {
        "scored_variants": int(len(out)),
        "optimization_successes": int(out["v4a_optimization_success"].sum()),
        "g_rescue_successes": int(len(g_success)),
        "best_delta_median_MIC": float(out["delta_median_MIC"].max(skipna=True)) if len(out) else None,
        "best_delta_worst_MIC": float(out["delta_worst_MIC"].max(skipna=True)) if len(out) else None,
        "best_delta_organisms_MIC_le_64": float(out["delta_organisms_MIC_le_64"].max(skipna=True)) if len(out) else None,
        "output": str(out_path),
        "g_success_output": str(g_path),
    }
    (out_path.parent / "optimization_gains_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
