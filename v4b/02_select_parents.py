#!/usr/bin/env python3
"""Select fitness-diverse parents for any V4B generation."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


def percentile_score(values: pd.Series, higher_is_better: bool) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().sum() == 0:
        return pd.Series(np.nan, index=values.index, dtype=float)
    ranks = numeric.rank(method="average", pct=True)
    if higher_is_better:
        return ranks
    return 1.0 - ranks + (1.0 / max(len(values), 1))


def infer_fitness(frame: pd.DataFrame) -> tuple[pd.Series, list[dict]]:
    """Construct a robust percentile composite from available APEX/V4 columns."""
    candidates = [
        ("evolution_fitness", True, 1.20),
        ("APEX_rank", False, 1.00),
        ("v4a_seed_rank", False, 0.80),
        ("v3_rank_score", True, 0.60),
        ("v3_rank_score_pre_apex", True, 0.40),
        ("v4a_score", True, 1.00),
        ("composite_score", True, 1.00),
        ("apex_score", True, 1.00),
        ("score", True, 0.50),
        ("mean_mic", False, 0.70),
        ("median_mic", False, 0.90),
        ("worst_mic", False, 0.40),
        ("organisms_le_64", True, 0.70),
        ("org_le_64", True, 0.70),
        ("novelty_score", True, 0.30),
        ("developability_score", True, 0.30),
    ]
    lowered = {column.lower(): column for column in frame.columns}
    components: list[pd.Series] = []
    weights: list[float] = []
    used: list[dict] = []
    for requested, higher, weight in candidates:
        actual = lowered.get(requested.lower())
        if actual is None:
            continue
        component = percentile_score(frame[actual], higher_is_better=higher)
        if component.notna().sum() == 0:
            continue
        components.append(component.fillna(component.median()))
        weights.append(weight)
        used.append({"column": actual, "higher_is_better": higher, "weight": weight})
    if not components:
        raise ValueError(
            "No usable fitness columns found. The source population must contain APEX/V4 fitness metrics."
        )
    matrix = np.column_stack([series.to_numpy(dtype=np.float64) for series in components])
    weight_array = np.asarray(weights, dtype=np.float64)
    score = (matrix * weight_array).sum(axis=1) / weight_array.sum()
    return pd.Series(score, index=frame.index, name="v4b_parent_fitness"), used


def standardize_latent(mu: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = mu.mean(axis=0, keepdims=True)
    scale = mu.std(axis=0, keepdims=True)
    scale[scale < 1e-8] = 1.0
    standardized = (mu - center) / scale
    norms = np.linalg.norm(standardized, axis=1, keepdims=True)
    unit = standardized / np.clip(norms, 1e-12, None)
    return standardized, unit


def greedy_farthest(points: np.ndarray, count: int, seed_index: int = 0) -> list[int]:
    if count <= 0 or len(points) == 0:
        return []
    count = min(count, len(points))
    selected = [int(seed_index)]
    min_distance = np.full(len(points), np.inf, dtype=np.float64)
    for _ in range(1, count):
        anchor = points[selected[-1]]
        distance = 1.0 - points @ anchor
        min_distance = np.minimum(min_distance, distance)
        min_distance[selected] = -np.inf
        selected.append(int(np.argmax(min_distance)))
    return selected


def allocate_counts(total: int, fractions: dict[str, float]) -> dict[str, int]:
    raw = {name: total * fraction for name, fraction in fractions.items()}
    counts = {name: int(np.floor(value)) for name, value in raw.items()}
    remainder = total - sum(counts.values())
    order = sorted(raw, key=lambda name: raw[name] - counts[name], reverse=True)
    for name in order[:remainder]:
        counts[name] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation", type=int, required=True, help="Target generation, from 1 onward.")
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--latents", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--n-parents", type=int, default=512)
    parser.add_argument("--candidate-pool-fraction", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.generation < 1:
        raise ValueError("--generation must be at least 1.")

    metadata_path = Path(args.metadata)
    latent_path = Path(args.latents)
    outdir = Path(args.outdir)
    prefix = f"generation_{args.generation:02d}"
    parents_path = outdir / f"{prefix}_parents.csv"
    summary_path = outdir / "parent_selection_summary.json"

    if (parents_path.exists() or summary_path.exists()) and not args.overwrite:
        raise FileExistsError(f"Generation {args.generation} parent outputs already exist.")

    frame = pd.read_csv(metadata_path, low_memory=False)
    latent = np.load(latent_path)
    mu = latent["mu"].astype(np.float32, copy=False)
    latent_ids = latent["candidate_id"].astype(str)

    if len(frame) != len(mu):
        raise ValueError("Metadata and latent vector row counts differ.")
    if not np.array_equal(frame["candidate_id"].astype(str).to_numpy(), latent_ids):
        raise ValueError("Metadata and latent candidate IDs are not aligned.")
    if not 1 <= args.n_parents <= len(frame):
        raise ValueError("--n-parents must be between 1 and the source population size.")
    if not 0 < args.candidate_pool_fraction <= 1:
        raise ValueError("--candidate-pool-fraction must be in (0, 1].")

    rng = np.random.default_rng(args.seed)
    fitness, used_columns = infer_fitness(frame)
    frame = frame.copy()
    frame["v4b_parent_fitness"] = fitness
    standardized, unit = standardize_latent(mu)
    centroid_distance = np.linalg.norm(standardized, axis=1)
    frame["latent_frontier_score"] = centroid_distance

    fractions = {
        "elite": 0.25,
        "diverse_fitness": 0.45,
        "frontier": 0.20,
        "random_control": 0.10,
    }
    counts = allocate_counts(args.n_parents, fractions)
    chosen: list[int] = []
    strata: dict[int, str] = {}

    fitness_order = np.argsort(-fitness.to_numpy())
    for idx_value in fitness_order[: counts["elite"]]:
        idx = int(idx_value)
        chosen.append(idx)
        strata[idx] = "elite"

    pool_n = max(
        counts["diverse_fitness"],
        int(round(len(frame) * args.candidate_pool_fraction)),
    )
    pool = [int(idx) for idx in fitness_order[:pool_n] if int(idx) not in strata]
    if pool and counts["diverse_fitness"]:
        local = greedy_farthest(unit[pool], counts["diverse_fitness"], seed_index=0)
        for pos in local:
            idx = pool[pos]
            chosen.append(idx)
            strata[idx] = "diverse_fitness"

    frontier_order = np.argsort(-centroid_distance)
    frontier_added = 0
    for idx_value in frontier_order:
        idx = int(idx_value)
        if idx in strata:
            continue
        chosen.append(idx)
        strata[idx] = "frontier"
        frontier_added += 1
        if frontier_added >= counts["frontier"]:
            break

    remaining = np.asarray([idx for idx in range(len(frame)) if idx not in strata], dtype=int)
    random_n = min(counts["random_control"], len(remaining))
    if random_n:
        for idx_value in rng.choice(remaining, size=random_n, replace=False):
            idx = int(idx_value)
            chosen.append(idx)
            strata[idx] = "random_control"

    if len(chosen) < args.n_parents:
        for idx_value in fitness_order:
            idx = int(idx_value)
            if idx not in strata:
                chosen.append(idx)
                strata[idx] = "fitness_fill"
            if len(chosen) == args.n_parents:
                break

    selected = frame.iloc[chosen].copy().reset_index(drop=True)
    selected["parent_selection_stratum"] = [strata[idx] for idx in chosen]
    selected["parent_selection_order"] = np.arange(1, len(selected) + 1)
    selected["parent_latent_row"] = chosen
    selected["generation_selected_for"] = args.generation

    outdir.mkdir(parents=True, exist_ok=True)
    selected.to_csv(parents_path, index=False)
    summary = {
        "schema_version": "1.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_generation": args.generation - 1,
        "target_generation": args.generation,
        "population_size": int(len(frame)),
        "parent_count": int(len(selected)),
        "latent_dimension": int(mu.shape[1]),
        "seed": int(args.seed),
        "requested_fractions": fractions,
        "actual_stratum_counts": selected["parent_selection_stratum"].value_counts().to_dict(),
        "fitness_columns": used_columns,
        "fitness_summary": selected["v4b_parent_fitness"].describe().to_dict(),
        "outputs": {"parents": str(parents_path)},
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
