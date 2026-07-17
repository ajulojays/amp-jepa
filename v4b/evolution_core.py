#!/usr/bin/env python3
"""Shared utilities for V4B evolutionary AMP-JEPA optimization.

These helpers intentionally avoid project-specific APEX assumptions. They infer a
usable higher-is-better fitness signal from scored candidate tables, while still
allowing an explicit score column when one is known.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

CANONICAL_AA = set("ACDEFGHIKLMNPQRSTVWY")
POSITIVE = set("KR")
NEGATIVE = set("DE")
HYDROPHOBIC = set("AILMFWVY")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean_sequence(seq: object) -> str:
    return "".join(str(seq).upper().split())


def basic_features(seq: str) -> dict[str, float]:
    seq = clean_sequence(seq)
    n = max(len(seq), 1)
    charge = sum(aa in POSITIVE for aa in seq) - sum(aa in NEGATIVE for aa in seq)
    hydro = sum(aa in HYDROPHOBIC for aa in seq) / n
    return {
        "length": float(len(seq)),
        "net_charge_KR_minus_DE": float(charge),
        "hydrophobic_fraction": float(hydro),
    }


def is_valid_sequence(seq: str, min_len: int = 8, max_len: int = 64) -> bool:
    seq = clean_sequence(seq)
    return min_len <= len(seq) <= max_len and all(aa in CANONICAL_AA for aa in seq)


def write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def choose_latent_key(npz: np.lib.npyio.NpzFile, requested: str = "auto") -> str:
    if requested != "auto":
        if requested not in npz.files:
            raise KeyError(f"Requested latent key '{requested}' not in archive. Available: {npz.files}")
        return requested
    for key in ("mu", "z", "latent", "latents"):
        if key in npz.files:
            return key
    candidates = [key for key in npz.files if key != "candidate_id" and np.asarray(npz[key]).ndim == 2]
    if not candidates:
        raise KeyError(f"Could not infer latent vector key. Available arrays: {npz.files}")
    return candidates[0]


def load_latents(path: str | Path, latent_key: str = "auto") -> tuple[np.ndarray, np.ndarray, str]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    # Some existing V4B latent archives saved candidate_id from pandas as an
    # object/string array. NumPy refuses to load those with allow_pickle=False.
    # These archives are generated locally by this pipeline, so enabling pickle
    # here is acceptable and lets failed runs resume without regenerating APEX
    # outputs.
    npz = np.load(path, allow_pickle=True)

    if "candidate_id" not in npz.files:
        raise KeyError(f"Latent archive lacks candidate_id array: {path}")
    key = choose_latent_key(npz, latent_key)
    ids = np.asarray(npz["candidate_id"]).astype(str)
    z = np.asarray(npz[key], dtype=np.float32)
    if z.ndim != 2:
        raise ValueError(f"Latent array '{key}' must be 2D, got shape {z.shape}")
    if len(ids) != z.shape[0]:
        raise ValueError(f"candidate_id length {len(ids)} does not match {key} rows {z.shape[0]}")
    return ids, z, key


def align_latents(df: pd.DataFrame, latent_ids: np.ndarray, z: np.ndarray) -> np.ndarray:
    if "candidate_id" not in df.columns:
        raise ValueError("Table lacks candidate_id column.")
    id_to_row = {cid: i for i, cid in enumerate(latent_ids.astype(str))}
    rows: list[int] = []
    missing: list[str] = []
    for cid in df["candidate_id"].astype(str):
        row = id_to_row.get(cid)
        if row is None:
            missing.append(cid)
        else:
            rows.append(row)
    if missing:
        preview = ", ".join(missing[:5])
        raise ValueError(f"{len(missing)} candidate IDs are missing from latent archive. Examples: {preview}")
    return z[np.asarray(rows, dtype=int)]


def normalized(values: Sequence[float]) -> np.ndarray:
    x = pd.to_numeric(pd.Series(values), errors="coerce").astype(float).to_numpy()
    finite = np.isfinite(x)
    if not finite.any():
        return np.zeros_like(x, dtype=np.float32)
    fill = float(np.nanmedian(x[finite]))
    x[~finite] = fill
    lo = float(np.min(x))
    hi = float(np.max(x))
    if hi - lo < 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo)).astype(np.float32)


def _numeric_series(df: pd.DataFrame, col: str) -> pd.Series | None:
    if col not in df.columns:
        return None
    s = pd.to_numeric(df[col], errors="coerce")
    if s.notna().sum() == 0:
        return None
    return s


def _filled_numeric(df: pd.DataFrame, col: str) -> pd.Series | None:
    s = _numeric_series(df, col)
    if s is None:
        return None
    return s.fillna(s.median())


def infer_fitness(df: pd.DataFrame, explicit_col: str | None = None) -> tuple[np.ndarray, str, str]:
    """Return higher-is-better fitness, source column/expression, and mode.

    Priority:
    1. Explicit user-provided score column, assumed higher-is-better.
    2. APEX MIC composite, where lower MIC is better.
    3. Known higher-is-better score/fitness columns.
    4. Any numeric score/fitness column.
    5. Generic MIC-like columns converted to negative log1p(MIC).
    6. V3 pre-APEX score fallback.
    """
    if explicit_col:
        s = _numeric_series(df, explicit_col)
        if s is None:
            raise ValueError(f"Explicit fitness column is absent or nonnumeric: {explicit_col}")
        return s.fillna(s.median()).to_numpy(dtype=np.float32), explicit_col, "explicit_higher_is_better"

    apex_median = _filled_numeric(df, "APEX_median_MIC")
    if apex_median is not None:
        score = -np.log1p(apex_median.clip(lower=0).to_numpy(dtype=np.float32))
        sources = ["APEX_median_MIC"]

        apex_worst = _filled_numeric(df, "APEX_worst_MIC")
        if apex_worst is not None:
            score = score + 0.35 * (-np.log1p(apex_worst.clip(lower=0).to_numpy(dtype=np.float32)))
            sources.append("APEX_worst_MIC")

        org64 = _filled_numeric(df, "organisms_MIC_le_64")
        if org64 is not None:
            score = score + 0.25 * normalized(org64.to_numpy(dtype=np.float32))
            sources.append("organisms_MIC_le_64")

        return score.astype(np.float32), "+".join(sources), "apex_mic_composite_lower_is_better"

    high_priority = [
        "v4b_survival_score",
        "generation_survival_score",
        "v4b_fitness",
        "apex_composite_score",
        "apex_rank_score",
        "apex_score",
        "composite_score",
        "fitness",
        "score",
    ]
    for col in high_priority:
        s = _numeric_series(df, col)
        if s is not None:
            return s.fillna(s.median()).to_numpy(dtype=np.float32), col, "higher_is_better"

    numeric_cols: list[str] = []
    for col in df.columns:
        lc = col.lower()
        if "candidate_id" in lc or lc.endswith("_id"):
            continue
        if any(bad in lc for bad in ("tox", "hemo", "cytotox")):
            continue
        if "score" in lc or "fitness" in lc:
            s = _numeric_series(df, col)
            if s is not None:
                numeric_cols.append(col)
    if numeric_cols:
        numeric_cols.sort(key=lambda c: ("apex" not in c.lower(), "v4b" not in c.lower(), c))
        col = numeric_cols[0]
        s = _numeric_series(df, col)
        assert s is not None
        return s.fillna(s.median()).to_numpy(dtype=np.float32), col, "inferred_higher_is_better"

    mic_cols: list[str] = []
    for col in df.columns:
        lc = col.lower()
        if "mic" in lc and not any(bad in lc for bad in ("id", "count", "n_")):
            s = _numeric_series(df, col)
            if s is not None:
                mic_cols.append(col)
    if mic_cols:
        mic_table = df[mic_cols].apply(pd.to_numeric, errors="coerce")
        mic = mic_table.median(axis=1, skipna=True)
        mic = mic.fillna(mic.median()).clip(lower=0)
        score = -np.log1p(mic.to_numpy(dtype=np.float32))
        return score.astype(np.float32), "+".join(mic_cols), "mic_lower_is_better"

    for col in ["v3_rank_score_pre_apex", "v3_rank_score"]:
        s = _numeric_series(df, col)
        if s is not None:
            return s.fillna(s.median()).to_numpy(dtype=np.float32), col, "pre_apex_fallback_higher_is_better"

    raise ValueError(
        "Could not infer a fitness column. Provide --fitness-column or include an APEX score/MIC column."
    )


def standardize_latents(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float32)
    mu = np.nanmean(z, axis=0, keepdims=True)
    sd = np.nanstd(z, axis=0, keepdims=True)
    sd[sd < 1e-6] = 1.0
    out = (z - mu) / sd
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def split_counts(total: int, fractions: dict[str, float]) -> dict[str, int]:
    raw = {k: max(0.0, float(v)) * total for k, v in fractions.items()}
    counts = {k: int(np.floor(v)) for k, v in raw.items()}
    remainder = total - sum(counts.values())
    order = sorted(raw, key=lambda k: raw[k] - counts[k], reverse=True)
    for k in order[: max(0, remainder)]:
        counts[k] += 1
    return counts


def farthest_select(
    z_std: np.ndarray,
    candidates: Iterable[int],
    already_selected: Iterable[int],
    k: int,
    score_norm: np.ndarray | None = None,
    seed: int = 0,
) -> list[int]:
    rng = np.random.default_rng(seed)
    remaining = np.array(sorted(set(int(i) for i in candidates) - set(int(i) for i in already_selected)), dtype=int)
    if k <= 0 or len(remaining) == 0:
        return []
    selected: list[int] = []
    anchors = np.array(list(set(int(i) for i in already_selected)), dtype=int)
    if len(anchors) > 0:
        min_dist = np.full(len(remaining), np.inf, dtype=np.float32)
        for start in range(0, len(anchors), 128):
            a = anchors[start : start + 128]
            d = ((z_std[remaining, None, :] - z_std[a][None, :, :]) ** 2).sum(axis=2)
            min_dist = np.minimum(min_dist, d.min(axis=1))
    else:
        centroid = z_std[remaining].mean(axis=0, keepdims=True)
        min_dist = ((z_std[remaining] - centroid) ** 2).sum(axis=1)

    while len(selected) < k and len(remaining) > 0:
        metric = min_dist.copy()
        if score_norm is not None:
            metric = metric * (0.75 + 0.25 * score_norm[remaining])
        if not np.isfinite(metric).any():
            pick_pos = int(rng.integers(0, len(remaining)))
        else:
            pick_pos = int(np.nanargmax(metric))
        pick = int(remaining[pick_pos])
        selected.append(pick)

        keep = np.ones(len(remaining), dtype=bool)
        keep[pick_pos] = False
        remaining = remaining[keep]
        min_dist = min_dist[keep]
        if len(remaining) == 0:
            break
        d_new = ((z_std[remaining] - z_std[pick]) ** 2).sum(axis=1)
        min_dist = np.minimum(min_dist, d_new)
    return selected


def stratified_select(
    df: pd.DataFrame,
    z: np.ndarray,
    n: int,
    fractions: dict[str, float],
    fitness: np.ndarray,
    seed: int = 0,
    label_prefix: str = "selection",
) -> pd.DataFrame:
    if n <= 0:
        raise ValueError("Selection size must be positive.")
    n = min(n, len(df))
    rng = np.random.default_rng(seed)
    score_norm = normalized(fitness)
    z_std = standardize_latents(z)
    counts = split_counts(n, fractions)

    selected: list[int] = []
    strata: dict[int, str] = {}

    elite_k = counts.get("elite", 0)
    if elite_k > 0:
        elite = np.argsort(-fitness)[:elite_k].astype(int).tolist()
        selected.extend(elite)
        for idx in elite:
            strata[idx] = f"{label_prefix}_fitness_elite"

    diverse_k = counts.get("diverse", 0)
    if diverse_k > 0:
        cutoff = np.quantile(fitness, 0.50)
        pool = np.where(fitness >= cutoff)[0].tolist()
        diverse = farthest_select(z_std, pool, selected, diverse_k, score_norm=score_norm, seed=seed + 1)
        selected.extend(diverse)
        for idx in diverse:
            strata[idx] = f"{label_prefix}_latent_diverse"

    frontier_k = counts.get("frontier", 0)
    if frontier_k > 0:
        centroid = z_std.mean(axis=0, keepdims=True)
        dist = ((z_std - centroid) ** 2).sum(axis=1)
        metric = dist * (0.60 + 0.40 * score_norm)
        remaining = np.array(sorted(set(range(len(df))) - set(selected)), dtype=int)
        if len(remaining) > 0:
            frontier = remaining[np.argsort(-metric[remaining])[:frontier_k]].astype(int).tolist()
            selected.extend(frontier)
            for idx in frontier:
                strata[idx] = f"{label_prefix}_latent_frontier"

    random_k = n - len(set(selected))
    if random_k > 0:
        remaining = np.array(sorted(set(range(len(df))) - set(selected)), dtype=int)
        if len(remaining) > 0:
            take = min(random_k, len(remaining))
            random_pick = rng.choice(remaining, size=take, replace=False).astype(int).tolist()
            selected.extend(random_pick)
            for idx in random_pick:
                strata[idx] = f"{label_prefix}_random_control"

    selected_unique: list[int] = []
    seen: set[int] = set()
    for idx in selected:
        if idx not in seen:
            selected_unique.append(idx)
            seen.add(idx)
        if len(selected_unique) >= n:
            break

    out = df.iloc[selected_unique].copy()
    out[f"{label_prefix}_stratum"] = [strata.get(idx, f"{label_prefix}_unassigned") for idx in selected_unique]
    out[f"{label_prefix}_rank"] = np.arange(1, len(out) + 1)
    out[f"{label_prefix}_score"] = fitness[selected_unique].astype(float)
    out[f"{label_prefix}_score_norm"] = score_norm[selected_unique].astype(float)
    return out
