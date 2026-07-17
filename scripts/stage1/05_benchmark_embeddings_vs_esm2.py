#!/usr/bin/env python3
"""Stage 1E: benchmark AMP-JEPA embeddings.

This script always reports unsupervised embedding diagnostics. If the metadata file
contains a label column, it also runs a lightweight supervised benchmark using a
logistic-regression classifier.

For a real paper-grade comparison, run the same script on AMP-JEPA embeddings and
on ESM2 embeddings exported with identical peptide IDs and split definitions.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd


def load_npz(path: Path) -> np.ndarray:
    payload = np.load(path, allow_pickle=True)
    if "embeddings" not in payload:
        raise SystemExit(f"[ERROR] NPZ missing 'embeddings': {path}")
    return payload["embeddings"]


def pairwise_cosine_sample(x: np.ndarray, sample_size: int = 2000, seed: int = 42) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    n = x.shape[0]
    if n < 2:
        return {"cosine_mean": np.nan, "cosine_std": np.nan}
    idx = rng.choice(n, size=min(n, sample_size), replace=False)
    z = x[idx].astype(np.float64)
    z = z / np.clip(np.linalg.norm(z, axis=1, keepdims=True), 1e-12, None)
    sims = z @ z.T
    upper = sims[np.triu_indices_from(sims, k=1)]
    return {"cosine_mean": float(np.mean(upper)), "cosine_std": float(np.std(upper))}


def supervised_benchmark(x: np.ndarray, meta: pd.DataFrame, label_col: str) -> Dict[str, float]:
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder, StandardScaler
        from sklearn.pipeline import make_pipeline
    except ImportError:
        return {"supervised_error": "scikit-learn not installed"}

    y_raw = meta[label_col].astype(str).values
    keep = pd.Series(y_raw).notna().values & (pd.Series(y_raw).str.lower() != "nan").values
    x = x[keep]
    y_raw = y_raw[keep]
    if len(set(y_raw)) < 2 or len(y_raw) < 20:
        return {"supervised_error": "Need at least two labels and >=20 labelled rows"}

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    stratify = y if min(np.bincount(y)) >= 2 else None
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=stratify)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    out = {
        "supervised_label_col": label_col,
        "supervised_n": int(len(y)),
        "supervised_n_classes": int(len(le.classes_)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
    }
    if len(le.classes_) == 2 and hasattr(clf[-1], "predict_proba"):
        proba = clf.predict_proba(x_test)[:, 1]
        out["roc_auc"] = float(roc_auc_score(y_test, proba))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-npz", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output", default="results/stage1/embedding_benchmark_summary.csv")
    parser.add_argument("--label-col", default=None, help="Optional label column in metadata")
    args = parser.parse_args()

    emb_path = Path(args.embedding_npz)
    meta_path = Path(args.metadata)
    if not emb_path.exists():
        raise SystemExit(f"[ERROR] Missing embedding NPZ: {emb_path}")
    if not meta_path.exists():
        raise SystemExit(f"[ERROR] Missing metadata: {meta_path}")

    x = load_npz(emb_path)
    meta = pd.read_csv(meta_path)
    if len(meta) != x.shape[0]:
        raise SystemExit(f"[ERROR] Metadata rows ({len(meta)}) != embeddings ({x.shape[0]})")

    rows = []
    diagnostics = {
        "embedding_file": str(emb_path),
        "n_sequences": int(x.shape[0]),
        "embedding_dim": int(x.shape[1]) if x.ndim == 2 else 0,
        "embedding_mean": float(np.mean(x)) if x.size else np.nan,
        "embedding_std": float(np.std(x)) if x.size else np.nan,
        "embedding_l2_mean": float(np.mean(np.linalg.norm(x, axis=1))) if x.size else np.nan,
        **pairwise_cosine_sample(x),
    }
    rows.append(diagnostics)

    label_col = args.label_col
    if label_col is None:
        for candidate in ["label", "activity", "is_amp", "evidence_tier"]:
            if candidate in meta.columns and meta[candidate].nunique(dropna=True) > 1:
                label_col = candidate
                break

    if label_col and label_col in meta.columns:
        sup = supervised_benchmark(x, meta, label_col)
        rows.append({"embedding_file": str(emb_path), **sup})
    else:
        rows.append({"embedding_file": str(emb_path), "supervised_note": "No usable label column found. Diagnostics only."})

    out = pd.DataFrame(rows)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(out.to_string(index=False))
    print(f"[DONE] Wrote {out_path}")


if __name__ == "__main__":
    main()
