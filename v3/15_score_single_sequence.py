#!/usr/bin/env python3
"""Score one peptide sequence with v3 heuristic filters."""

from __future__ import annotations

import argparse

from ampjepa_hybrid_v3 import candidate_score, clean_sequence


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sequence", required=True)
    args = p.parse_args()
    seq = clean_sequence(args.sequence)
    scored = candidate_score(seq)
    print(f"sequence={seq}")
    for key, value in scored.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
