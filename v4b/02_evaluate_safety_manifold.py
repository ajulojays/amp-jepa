#!/usr/bin/env python3
"""Evaluate whether V3 latent μ contains hemolysis/cytotoxicity signal."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HYDRO = set("AILMFWYV")
AROM = set("FWY")


def physicochemical_features(sequences: pd.Series) -> np.ndarray:
    rows = []
    for seq in sequences.astype(str):
        n = max(len(seq), 1)
        rows.append(
            [
                len(seq),
                seq.count("K") + seq.count("R") - seq.count("D") - seq.count("E"),
                sum(a in HYDRO for a in seq) / n,
                (seq.count("G") + seq.count("P")) / n,
                sum(a in AROM for a in seq) / n,
                seq.count("C") / n,
                seq.count("W") / n,
            ]
        )
    return np.asarray(rows, dtype=np.float32)


def evaluate_endpoint(
    metadata: pd.DataFrame,
    latent: np.ndarray,
    endpoint: str,
    outdir: Path,
    folds: int,
    random_state: int,
) -> list[dict]:
    label_col = f"{endpoint}_label"
    if label_col not in metadata.columns:
        return []

    mask = metadata[label_col].notna().to_numpy()
    y = pd.to_numeric(metadata.loc[mask, label_col], errors="coerce").to_numpy(dtype=int)
    if len(y) == 0 or len(np.unique(y)) < 2:
        return []

    min_class = int(np.bincount(y).min())
    n_splits = min(folds, min_class)
    if n_splits < 2:
        return []

    X_latent = latent[mask]
    X_phys = physicochemical_features(metadata.loc[mask, "sequence"])
    X_combined = np.concatenate([X_latent, X_phys], axis=1)

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rows: list[dict] = []

    for feature_set, X in {
        "latent_mu": X_latent,
        "physicochemical": X_phys,
        "latent_plus_physicochemical": X_combined,
    }.items():
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "logreg",
                    LogisticRegression(
                        max_iter=5000,
                        class_weight="balanced",
                        solver="liblinear",
                        random_state=random_state,
                    ),
                ),
            ]
        )
        prob = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
        pred = (prob >= 0.5).astype(int)
        rows.append(
            {
                "endpoint": endpoint,
                "feature_set": feature_set,
                "n_labeled": int(len(y)),
                "n_positive": int((y == 1).sum()),
                "n_negative": int((y == 0).sum()),
                "cv_folds": int(n_splits),
                "auroc": float(roc_auc_score(y, prob)),
                "average_precision": float(average_precision_score(y, prob)),
                "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
                "accuracy": float(accuracy_score(y, pred)),
            }
        )

    pca = PCA(n_components=2, random_state=random_state)
    coords = pca.fit_transform(StandardScaler().fit_transform(X_latent))
    pca_df = metadata.loc[mask, ["sequence", label_col]].copy().reset_index(drop=True)
    pca_df["PC1"] = coords[:, 0]
    pca_df["PC2"] = coords[:, 1]
    pca_df.to_csv(outdir / f"{endpoint}_pca_coordinates.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 6))
    for label, group in pca_df.groupby(label_col):
        ax.scatter(group["PC1"], group["PC2"], s=18, alpha=0.65, label=f"label={int(label)}")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    ax.set_title(f"V3 latent μ safety manifold: {endpoint}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / f"{endpoint}_pca.png", dpi=300)
    plt.close(fig)

    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metadata", default="v4b/results/safety_manifold_pilot/latent_metadata.csv")
    ap.add_argument("--latent", default="v4b/results/safety_manifold_pilot/latent_mu.npy")
    ap.add_argument("--outdir", default="v4b/results/safety_manifold_pilot")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--go-min-labeled", type=int, default=200)
    ap.add_argument("--go-min-positive", type=int, default=40)
    ap.add_argument("--go-min-negative", type=int, default=40)
    ap.add_argument("--go-min-auroc", type=float, default=0.70)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(args.metadata, low_memory=False)
    latent = np.load(args.latent)
    if len(metadata) != len(latent):
        raise SystemExit(f"Row mismatch: metadata={len(metadata)} latent={len(latent)}")

    rows: list[dict] = []
    for endpoint in ["hemolysis", "cytotoxicity"]:
        rows.extend(
            evaluate_endpoint(
                metadata,
                latent,
                endpoint,
                outdir,
                folds=args.folds,
                random_state=args.random_state,
            )
        )

    metrics = pd.DataFrame(rows)
    metrics_path = outdir / "safety_manifold_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    decisions = {}
    for endpoint in ["hemolysis", "cytotoxicity"]:
        sub = metrics[metrics["endpoint"] == endpoint] if len(metrics) else pd.DataFrame()
        if sub.empty:
            decisions[endpoint] = {
                "decision": "NO-GO",
                "reason": "Insufficient binary labels or only one class present",
            }
            continue
        best = sub.sort_values("auroc", ascending=False).iloc[0]
        enough = (
            best["n_labeled"] >= args.go_min_labeled
            and best["n_positive"] >= args.go_min_positive
            and best["n_negative"] >= args.go_min_negative
        )
        signal = best["auroc"] >= args.go_min_auroc
        if enough and signal:
            decision = "GO"
            reason = "Adequate labels and cross-validated latent/combined signal"
        elif not enough:
            decision = "REVISE"
            reason = "Signal estimated, but label count or class balance is insufficient"
        else:
            decision = "REVISE"
            reason = "Adequate labels but AUROC is below the initial safety-head threshold"
        decisions[endpoint] = {
            "decision": decision,
            "reason": reason,
            "best_feature_set": str(best["feature_set"]),
            "best_auroc": float(best["auroc"]),
            "average_precision": float(best["average_precision"]),
            "n_labeled": int(best["n_labeled"]),
            "n_positive": int(best["n_positive"]),
            "n_negative": int(best["n_negative"]),
        }

    summary = {
        "phase": "V4B Phase 0 safety manifold pilot",
        "decisions": decisions,
        "thresholds": {
            "minimum_labeled": args.go_min_labeled,
            "minimum_positive": args.go_min_positive,
            "minimum_negative": args.go_min_negative,
            "minimum_cv_auroc": args.go_min_auroc,
        },
        "metrics_file": str(metrics_path),
        "experimental_validation": False,
    }
    summary_path = outdir / "safety_manifold_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
