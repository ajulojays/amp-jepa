#!/usr/bin/env python3
"""Rank AMP-JEPA-Hybrid v3 candidates by novelty and AMP-like constraints."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ampjepa_hybrid_v3 import candidate_score, clean_sequence, max_identity_to_training


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidates", default="v3/results/raw_candidates_v3.csv")
    p.add_argument("--corpus", default="v3/data/processed/peptide_corpus_v3.csv")
    p.add_argument("--output", default="v3/results/ranked_candidates_v3.csv")
    p.add_argument("--sample-train-limit", type=int, default=5000)
    args = p.parse_args()

    cand = pd.read_csv(args.candidates)
    corpus = pd.read_csv(args.corpus)
    if "sequence" not in cand.columns or "sequence" not in corpus.columns:
        raise SystemExit("[ERROR] both candidates and corpus need a sequence column")

    train_seqs = corpus["sequence"].dropna().map(clean_sequence).tolist()
    rows = []
    for _, row in cand.iterrows():
        seq = clean_sequence(row["sequence"])
        max_id = max_identity_to_training(seq, train_seqs, sample_limit=args.sample_train_limit)
        scored = candidate_score(seq, max_train_identity=max_id)
        rows.append({**row.to_dict(), "sequence": seq, **scored})

    out = pd.DataFrame(rows)
    out = out.drop_duplicates("sequence")
    out = out.sort_values(["passes_v3_filters", "v3_rank_score", "novelty_score"], ascending=[False, False, False])
    out.insert(0, "v3_rank", range(1, len(out) + 1))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(out.head(25).to_string(index=False))
    print(f"[DONE] Wrote {out_path}")


if __name__ == "__main__":
    main()
