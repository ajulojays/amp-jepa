#!/usr/bin/env python3
"""Stage 1B: build biologically motivated JEPA context/target pairs.

Each row defines one masked context view and one target span from the original
peptide. The downstream JEPA trainer predicts target-encoder latent vectors for
that span from the masked context encoder output.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

CHARGED = set("KRHDE")
HYDROPHOBIC = set("AILMFWVY")


def info(message: str) -> None:
    print(f"[INFO] {message}")


def clamp_span(start: int, length: int, seq_len: int) -> Tuple[int, int]:
    if seq_len <= 0:
        return 0, 0
    start = max(0, min(start, seq_len - 1))
    length = max(1, min(length, seq_len - start))
    return start, length


def masked_context(seq: str, start: int, length: int, mask_char: str = "X") -> str:
    return seq[:start] + (mask_char * length) + seq[start + length :]


def longest_cluster(seq: str, alphabet: set[str]) -> Tuple[int, int]:
    best = (0, 1)
    current_start = None
    current_len = 0
    for i, aa in enumerate(seq):
        if aa in alphabet:
            if current_start is None:
                current_start = i
                current_len = 1
            else:
                current_len += 1
            if current_len > best[1]:
                best = (current_start, current_len)
        else:
            current_start = None
            current_len = 0
    return best


def make_masks(seq: str, pairs_per_sequence: int, rng: random.Random) -> List[Tuple[str, int, int]]:
    seq_len = len(seq)
    if seq_len == 0:
        return []

    span_len = max(1, min(seq_len, round(seq_len * 0.18)))
    masks: List[Tuple[str, int, int]] = []

    # Generic local context masks.
    for _ in range(max(1, pairs_per_sequence // 3)):
        length = rng.randint(1, max(1, min(seq_len, span_len + 2)))
        start = rng.randint(0, max(0, seq_len - length))
        masks.append(("random_contiguous", start, length))

    # Terminal biology: many AMPs are sensitive to N/C-terminal edits.
    n_len = max(1, min(seq_len, round(seq_len * 0.20)))
    masks.append(("n_terminal", *clamp_span(0, n_len, seq_len)))
    masks.append(("c_terminal", *clamp_span(seq_len - n_len, n_len, seq_len)))

    # Charge cluster: cationic patches are central to many AMP mechanisms.
    start, length = longest_cluster(seq, CHARGED)
    masks.append(("charge_cluster", *clamp_span(start, length, seq_len)))

    # Hydrophobic patch: crude proxy for amphipathic/membrane-interacting face.
    start, length = longest_cluster(seq, HYDROPHOBIC)
    masks.append(("hydrophobic_patch", *clamp_span(start, length, seq_len)))

    # Alpha-helix face proxy: mask every 3rd/4th nearby residue by using a short contiguous proxy window.
    if seq_len >= 9:
        start = rng.randint(0, seq_len - min(9, seq_len))
        masks.append(("helix_face_proxy", *clamp_span(start, min(9, seq_len - start), seq_len)))

    # De-duplicate and trim to requested count.
    seen = set()
    unique = []
    for mask_type, start, length in masks:
        key = (mask_type, start, length)
        if key not in seen:
            unique.append(key)
            seen.add(key)
    while len(unique) < pairs_per_sequence:
        length = rng.randint(1, max(1, min(seq_len, span_len + 2)))
        start = rng.randint(0, max(0, seq_len - length))
        key = ("random_contiguous_extra", start, length)
        if key not in seen:
            unique.append(key)
            seen.add(key)
    return unique[:pairs_per_sequence]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="data/processed/stage1/peptide_corpus.csv")
    parser.add_argument("--output", default="data/processed/stage1/jepa_pairs.csv")
    parser.add_argument("--pairs-per-sequence", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        raise SystemExit(f"[ERROR] Missing corpus: {corpus_path}")

    corpus = pd.read_csv(corpus_path)
    required = {"peptide_id", "sequence"}
    missing = required - set(corpus.columns)
    if missing:
        raise SystemExit(f"[ERROR] Corpus missing required columns: {sorted(missing)}")

    rng = random.Random(args.seed)
    rows = []
    for _, row in corpus.iterrows():
        seq = str(row["sequence"])
        for j, (mask_type, start, length) in enumerate(make_masks(seq, args.pairs_per_sequence, rng), start=1):
            rows.append(
                {
                    "pair_id": f"{row['peptide_id']}_pair_{j:02d}",
                    "peptide_id": row["peptide_id"],
                    "sequence": seq,
                    "sequence_length": len(seq),
                    "mask_type": mask_type,
                    "mask_start": int(start),
                    "mask_len": int(length),
                    "target_sequence": seq[start : start + length],
                    "context_sequence": masked_context(seq, start, length),
                }
            )

    out = pd.DataFrame(rows)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    info(f"Input peptides: {len(corpus):,}")
    info(f"JEPA pairs: {len(out):,}")
    info("Mask counts:")
    print(out["mask_type"].value_counts().to_string())
    print(f"[DONE] Wrote {out_path}")


if __name__ == "__main__":
    main()
