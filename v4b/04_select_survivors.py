#!/usr/bin/env python3
"""Merge APEX scores and select a fitness-diverse survivor population."""

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


def infer_direction(column: str) -> bool | None:
    """Return True for higher-is-better, False for lower-is-better, or None."""
    name = column.lower()
    if any(token in name for token in ("hemol", "cytotox", "toxicity")):
        return None
    if "mic" in name or "rank" in name or "loss" in name:
        return False
    if any(token in name for token in ("score", "activity", "organisms", "org_le", "novelty", "developability")):
        return True
    return None


def infer_fitness(frame: pd.DataFrame) -> tuple[pd.Series, list[dict]]:
    preferred = [
        ("evolution_fitness", True, 1.20),
        ("APEX_rank", False, 1.00),
        ("apex_APEX_rank", False, 1.00),
        ("apex_score", True, 1.00),
        ("composite_score", True, 1.00),
        ("v4a_score", True, 0.90),
        ("mean_mic", False, 0.70),
        ("median_mic", False, 0.90),
        ("worst_mic", False, 0.40),
        ("apex_mean_mic", False, 0.70),
        ("apex_median_mic", False, 0.90),
        ("apex_worst_mic", False, 0.40),
        ("organisms_le_64", True, 0.70),
        ("org_le_64", True, 0.70),
        ("apex_organisms_le_64", True, 0.70),
        ("apex_org_le_64", True, 0.70),
        ("v3_rank_score", True, 0.35),
        ("v3_rank_score_pre_apex", True, 0.25),
        ("novelty_score", True, 0.25),
        ("developability_score", True, 0.25),
    ]
    lowered = {column.lower(): column for column in frame.columns}
    used_columns: set[str] = set()
    components: list[pd.Series] = []
    weights: list[float] = []
    used: list[dict] = []

    for requested, higher, weight in preferred:
        actual = lowered.get(requested.lower())
        if actual is None or actual in used_columns:
            continue
        component = percentile_score(frame[actual], higher_is_better=higher)
        if component.notna().sum() == 0:
            continue
        components.append(component.fillna(component.median()))
        weights.append(weight)
        used.append({"column": actual, "higher_is_better": higher, "weight": weight})
        used_columns.add(actual)

    for column in frame.columns:
        if column in used_columns:
            continue
        direction = infer_direction(column)
        if direction is None:
            continue
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.notna().sum() < max(10, int(0.20 * len(frame))):
            continue
        component = percentile_score(numeric, higher_is_better=direction)
        components.append(component.fillna(component.median()))
        weights.append(0.20)
        used.append({"column": column, "higher_is_better": direction, "weight": 0.20})
        used_columns.add(column)

    if not components:
        raise ValueError("No usable APEX or fitness columns were found after score merging.")
    matrix = np.column_stack([series.to_numpy(dtype=np.float64) for series in components])
    weight_array = np.asarray(weights, dtype=np.float64)
    score = (matrix * weight_array).sum(axis=1) / weight_array.sum()
    return pd.Series(score, index=frame.index, name="evolution_fitness"), used


def merge_apex_scores(offspring: pd.DataFrame, scored: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    if "candidate_id" in offspring.columns and "candidate_id" in scored.columns:
        key = "candidate_id"
    elif "sequence" in offspring.columns and "sequence" in scored.columns:
        key = "sequence"
    else:
        raise ValueError("APEX output must share candidate_id or sequence with the offspring table.")

    scored = scored.copy()
    if key == "candidate_id":
        scored[key] = scored[key].astype(str)
        offspring = offspring.copy()
        offspring[key] = offspring[key].astype(str)
    else:
        scored[key] = scored[key].astype(str).str.upper().str.replace(r"\s+", "", regex=True)
        offspring = offspring.copy()
        offspring[key] = offspring[key].astype(str).str.upper().str.replace(r"\s+", "", regex=True)

    if scored[key].duplicated().any():
        scored = scored.drop_duplicates(key, keep="first")

    rename: dict[str, str] = {}
    score_columns: list[str] = []
    for column in scored.columns:
        if column == key:
            continue
        target = column if column not in offspring.columns else f"apex_{column}"
        rename[column] = target
        score_columns.append(target)
    scored = scored.rename(columns=rename)
    merged = offspring.merge(scored[[key] + score_columns], on=key, how="left", validate="one_to_one")

    matched = merged[score_columns].notna().any(axis=1) if score_columns else pd.Series(False, index=merged.index)
    if not score_columns or not matched.all():
        missing = int((~matched).sum())
        raise ValueError(
            f"APEX score merge failed for {missing:,} offspring rows. "
            "The scorer must return one row per input candidate."
        )
    return merged, score_columns


def load_latent_map(path: Path, value_key: str) -> tuple[dict[str, np.ndarray], int]:
    archive = np.load(path)
    ids = archive["candidate_id"].astype(str)
    values = archive[value_key].astype(np.float32, copy=False)
    if len(ids) != len(values):
        raise ValueError(f"Latent ID/value count mismatch in {path}")
    return {candidate_id: values[i] for i, candidate_id in enumerate(ids)}, int(values.shape[1])


def standardize_latent(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = values.mean(axis=0, keepdims=True)
    scale = values.std(axis=0, keepdims=True)
    scale[scale < 1e-8] = 1.0
    standardized = (values - center) / scale
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
    parser.add_argument("--generation", type=int, required=True)
    parser.add_argument("--source-population", required=True)
    parser.add_argument("--source-latents", required=True)
    parser.add_argument("--offspring", required=True)
    parser.add_argument("--offspring-scored", required=True)
    parser.add_argument("--offspring-latents", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--n-survivors", type=int, default=2500)
    parser.add_argument("--candidate-pool-fraction", type=float, default=0.60)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.generation < 1:
        raise ValueError("--generation must be at least 1.")

    source_path = Path(args.source_population)
    source_latent_path = Path(args.source_latents)
    offspring_path = Path(args.offspring)
    scored_path = Path(args.offspring_scored)
    offspring_latent_path = Path(args.offspring_latents)
    outdir = Path(args.outdir)
    prefix = f"generation_{args.generation:02d}"
    merged_scored_path = outdir / f"{prefix}_scored_offspring.csv"
    survivors_path = outdir / f"{prefix}_survivors.csv"
    selection_latents_path = outdir / f"{prefix}_selection_latents.npz"
    summary_path = outdir / "survivor_selection_summary.json"

    required = [source_path, source_latent_path, offspring_path, scored_path, offspring_latent_path]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")
    if any(path.exists() for path in (merged_scored_path, survivors_path, selection_latents_path, summary_path)) and not args.overwrite:
        raise FileExistsError(f"Generation {args.generation} survivor outputs already exist.")

    source = pd.read_csv(source_path, low_memory=False)
    offspring = pd.read_csv(offspring_path, low_memory=False)
    scored = pd.read_csv(scored_path, low_memory=False)
    offspring_scored, score_columns = merge_apex_scores(offspring, scored)
    outdir.mkdir(parents=True, exist_ok=True)
    offspring_scored.to_csv(merged_scored_path, index=False)

    source = source.copy()
    offspring_scored = offspring_scored.copy()
    source["population_origin"] = "carryover"
    offspring_scored["population_origin"] = "offspring"
    combined = pd.concat([source, offspring_scored], ignore_index=True, sort=False)
    if "candidate_id" not in combined.columns or "sequence" not in combined.columns:
        raise ValueError("Both source and offspring populations require candidate_id and sequence.")
    combined["candidate_id"] = combined["candidate_id"].astype(str)
    combined["sequence"] = combined["sequence"].astype(str).str.upper().str.replace(r"\s+", "", regex=True)
    combined = combined.drop_duplicates("candidate_id", keep="last").reset_index(drop=True)

    source_latents, source_dim = load_latent_map(source_latent_path, "mu")
    offspring_latents, offspring_dim = load_latent_map(offspring_latent_path, "z")
    if source_dim != offspring_dim:
        raise ValueError("Source and offspring latent dimensions differ.")

    latent_rows: list[np.ndarray] = []
    missing_latents: list[str] = []
    for candidate_id in combined["candidate_id"]:
        if candidate_id in offspring_latents:
            latent_rows.append(offspring_latents[candidate_id])
        elif candidate_id in source_latents:
            latent_rows.append(source_latents[candidate_id])
        else:
            missing_latents.append(candidate_id)
    if missing_latents:
        raise ValueError(f"Missing latent vectors for {len(missing_latents):,} combined candidates.")
    latent_matrix = np.stack(latent_rows).astype(np.float32, copy=False)

    fitness, used_columns = infer_fitness(combined)
    combined["evolution_fitness"] = fitness
    combined["_latent_row"] = np.arange(len(combined), dtype=int)
    combined = combined.sort_values("evolution_fitness", ascending=False)
    combined = combined.drop_duplicates("sequence", keep="first").reset_index(drop=True)
    latent_matrix = latent_matrix[combined["_latent_row"].to_numpy(dtype=int)]
    combined["_latent_row"] = np.arange(len(combined), dtype=int)

    if not 1 <= args.n_survivors <= len(combined):
        raise ValueError("--n-survivors must be between 1 and the combined candidate pool size.")

    standardized, unit = standardize_latent(latent_matrix)
    frontier_score = np.linalg.norm(standardized, axis=1)
    combined["latent_frontier_score"] = frontier_score

    fractions = {
        "elite": 0.40,
        "diverse_fitness": 0.40,
        "frontier": 0.15,
        "random_control": 0.05,
    }
    counts = allocate_counts(args.n_survivors, fractions)
    rng = np.random.default_rng(args.seed)
    chosen: list[int] = []
    strata: dict[int, str] = {}

    fitness_order = np.argsort(-combined["evolution_fitness"].to_numpy())
    for idx_value in fitness_order[: counts["elite"]]:
        idx = int(idx_value)
        chosen.append(idx)
        strata[idx] = "elite"

    pool_n = max(counts["diverse_fitness"], int(round(len(combined) * args.candidate_pool_fraction)))
    pool = [int(idx) for idx in fitness_order[:pool_n] if int(idx) not in strata]
    if pool and counts["diverse_fitness"]:
        local = greedy_farthest(unit[pool], counts["diverse_fitness"], seed_index=0)
        for pos in local:
            idx = pool[pos]
            chosen.append(idx)
            strata[idx] = "diverse_fitness"

    frontier_added = 0
    for idx_value in np.argsort(-frontier_score):
        idx = int(idx_value)
        if idx in strata:
            continue
        chosen.append(idx)
        strata[idx] = "frontier"
        frontier_added += 1
        if frontier_added >= counts["frontier"]:
            break

    remaining = np.asarray([idx for idx in range(len(combined)) if idx not in strata], dtype=int)
    random_n = min(counts["random_control"], len(remaining))
    if random_n:
        for idx_value in rng.choice(remaining, size=random_n, replace=False):
            idx = int(idx_value)
            chosen.append(idx)
            strata[idx] = "random_control"

    if len(chosen) < args.n_survivors:
        for idx_value in fitness_order:
            idx = int(idx_value)
            if idx not in strata:
                chosen.append(idx)
                strata[idx] = "fitness_fill"
            if len(chosen) == args.n_survivors:
                break

    survivors = combined.iloc[chosen].copy().reset_index(drop=True)
    selected_latents = latent_matrix[np.asarray(chosen, dtype=int)]
    survivors["population_generation"] = args.generation
    survivors["survivor_selection_stratum"] = [strata[idx] for idx in chosen]
    survivors["survivor_selection_order"] = np.arange(1, len(survivors) + 1)
    survivors["survived_from_previous_population"] = survivors["population_origin"].eq("carryover")
    survivors = survivors.drop(columns=["_latent_row"], errors="ignore")
    survivors.to_csv(survivors_path, index=False)
    np.savez_compressed(
        selection_latents_path,
        candidate_id=survivors["candidate_id"].astype(str).to_numpy(),
        z=selected_latents,
        generation=np.full(len(survivors), args.generation, dtype=np.int16),
    )

    summary = {
        "schema_version": "1.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "generation": args.generation,
        "source_population_size": int(len(source)),
        "offspring_scored": int(len(offspring_scored)),
        "combined_unique_sequences": int(len(combined)),
        "survivor_count": int(len(survivors)),
        "carryover_survivors": int(survivors["survived_from_previous_population"].sum()),
        "new_offspring_survivors": int((~survivors["survived_from_previous_population"]).sum()),
        "latent_dimension": int(selected_latents.shape[1]),
        "seed": int(args.seed),
        "apex_score_columns": score_columns,
        "fitness_columns": used_columns,
        "requested_fractions": fractions,
        "actual_stratum_counts": survivors["survivor_selection_stratum"].value_counts().to_dict(),
        "fitness_summary": survivors["evolution_fitness"].describe().to_dict(),
        "outputs": {
            "scored_offspring": str(merged_scored_path),
            "survivors": str(survivors_path),
            "selection_latents": str(selection_latents_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
