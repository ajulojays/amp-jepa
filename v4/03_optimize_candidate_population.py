#!/usr/bin/env python3
"""Generate V4A optimized variants from the full candidate landscape.

This is an APEX-only, sequence-level optimizer. It works on all candidate classes,
including G-Rescue candidates, by diagnosing failure modes and applying simple
biologically motivated sequence edits. It does not perform wet-lab guidance.

Output is an APEX-ready candidate CSV with one `sequence` column.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd

AA = "ACDEFGHIKLMNPQRSTVWY"
HYDRO = "AILMFWYV"
POLAR = "STNQGA"
POS = "KR"
AROM = "FWY"


def clean_sequence(x: object) -> str:
    return "".join(ch for ch in str(x).strip().upper() if ch in set(AA))


def props(seq: str) -> dict:
    n = len(seq)
    return {
        "length": n,
        "net_charge_KR_minus_DE": seq.count("K") + seq.count("R") - seq.count("D") - seq.count("E"),
        "hydrophobic_fraction": sum(ch in HYDRO for ch in seq) / max(n, 1),
        "cysteine_count": seq.count("C"),
        "tryptophan_count": seq.count("W"),
        "aromatic_fraction": sum(ch in AROM for ch in seq) / max(n, 1),
    }


def valid(seq: str, min_len=5, max_len=64) -> bool:
    if not (min_len <= len(seq) <= max_len):
        return False
    if any(ch not in AA for ch in seq):
        return False
    p = props(seq)
    if p["hydrophobic_fraction"] > 0.80 or p["hydrophobic_fraction"] < 0.10:
        return False
    if p["cysteine_count"] > 4 or p["tryptophan_count"] > 5:
        return False
    return True


def substitute(seq: str, rng: random.Random, positions, choices: str) -> str:
    if not seq or not positions:
        return seq
    pos = rng.choice(list(positions))
    choices = [c for c in choices if c != seq[pos]]
    if not choices:
        return seq
    return seq[:pos] + rng.choice(choices) + seq[pos + 1:]


def conservative_mutation(seq: str, rng: random.Random) -> str:
    groups = ["ILV", "KR", "STNQ", "FWY", "AG", "DE"]
    idxs = list(range(len(seq)))
    rng.shuffle(idxs)
    for i in idxs:
        for g in groups:
            if seq[i] in g and len(g) > 1:
                return substitute(seq, rng, [i], g)
    return seq


def rescue_variant(seq: str, modes: str, rng: random.Random) -> tuple[str, str]:
    modes = modes or ""
    n = len(seq)

    if "too_short" in modes and n < 35:
        left = rng.choice(["K", "R", "G", "A", "L"])
        right = rng.choice(["K", "R", "G", "A", "L"])
        return left + seq + right, "terminal_extension"

    if "too_long" in modes and n > 8:
        trim_left = rng.random() < 0.5
        k = rng.choice([1, 2, 3])
        return (seq[k:] if trim_left else seq[:-k]), "terminal_trim"

    if "low_charge" in modes:
        candidates = [i for i, ch in enumerate(seq) if ch not in POS and ch not in "DE"]
        return substitute(seq, rng, candidates, POS), "charge_rescue_KR"

    if "excess_charge" in modes:
        candidates = [i for i, ch in enumerate(seq) if ch in POS]
        return substitute(seq, rng, candidates, "ASNQG"), "charge_softening"

    if "excess_hydrophobicity" in modes:
        candidates = [i for i, ch in enumerate(seq) if ch in HYDRO]
        return substitute(seq, rng, candidates, "ASKNQG"), "hydrophobicity_reduction"

    if "low_hydrophobicity" in modes:
        candidates = [i for i, ch in enumerate(seq) if ch not in HYDRO and ch not in POS]
        return substitute(seq, rng, candidates, "AILV"), "hydrophobicity_addition"

    if "excess_aromatic" in modes:
        candidates = [i for i, ch in enumerate(seq) if ch in AROM]
        return substitute(seq, rng, candidates, "AILVKS"), "aromatic_reduction"

    if "excess_cysteine" in modes:
        candidates = [i for i, ch in enumerate(seq) if ch == "C"]
        return substitute(seq, rng, candidates, "ASG"), "cysteine_control"

    if "weak_worst_case" in modes:
        # Gentle amphipathic/charge tuning rather than large redesign.
        if rng.random() < 0.5:
            candidates = [i for i, ch in enumerate(seq) if ch not in POS and ch not in "DE"]
            return substitute(seq, rng, candidates, POS), "worst_case_charge_tuning"
        candidates = [i for i, ch in enumerate(seq) if ch in HYDRO]
        return substitute(seq, rng, candidates, "ASTNQ"), "worst_case_balance_tuning"

    if "weak_breadth" in modes:
        candidates = [i for i, ch in enumerate(seq) if ch not in POS and ch not in "DE"]
        return substitute(seq, rng, candidates, "KRALIV"), "breadth_tuning"

    if "low_complexity" in modes:
        counts = {ch: seq.count(ch) for ch in set(seq)}
        high = max(counts, key=counts.get)
        candidates = [i for i, ch in enumerate(seq) if ch == high]
        return substitute(seq, rng, candidates, "ASTNQKRLIV"), "complexity_rescue"

    return conservative_mutation(seq, rng), "conservative_mutation"


def make_variants(row, rng: random.Random, n_variants: int) -> list[dict]:
    seq = clean_sequence(row.sequence)
    out = []
    seen = set()
    modes = str(getattr(row, "failure_modes", ""))

    for i in range(n_variants * 3):
        if len(out) >= n_variants:
            break
        new, op = rescue_variant(seq, modes, rng)
        # Occasional second gentle edit for exploration.
        if rng.random() < 0.25 and len(new) >= 8:
            new = conservative_mutation(new, rng)
            op += "+conservative"
        new = clean_sequence(new)
        if new == seq or new in seen or not valid(new):
            continue
        seen.add(new)
        p = props(new)
        out.append({
            "candidate_id": f"V4A_{str(row.v4a_class).split('_')[0]}_{len(out)+1:03d}_{getattr(row, 'candidate_id', 'parent')}",
            "sequence": new,
            "parent_sequence": seq,
            "parent_candidate_id": getattr(row, "candidate_id", ""),
            "parent_v4a_class": getattr(row, "v4a_class", ""),
            "parent_failure_modes": modes,
            "optimization_operator": op,
            **p,
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--landscape", default="v4/results/landscape/candidate_landscape.csv")
    ap.add_argument("--output", default="v4/results/optimization/optimized_variants.csv")
    ap.add_argument("--g-output", default="v4/results/rescue/g_rescue_variants.csv")
    ap.add_argument("--max-parents", type=int, default=1200)
    ap.add_argument("--max-variants", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    df = pd.read_csv(args.landscape, low_memory=False)
    df["sequence"] = df["sequence"].map(clean_sequence)

    # Adaptive parent budget: keep all classes involved but prioritize classes with high score or rescue value.
    class_order = ["A_elite", "B_near_pass", "C_high_novelty", "D_broad_spectrum", "E_worst_case_robust", "F_developable", "G_rescue"]
    budgets = {
        "A_elite": 200,
        "B_near_pass": 250,
        "C_high_novelty": 200,
        "D_broad_spectrum": 175,
        "E_worst_case_robust": 150,
        "F_developable": 150,
        "G_rescue": 300,
    }

    parents = []
    for cls in class_order:
        sub = df[df["v4a_class"].eq(cls)].copy()
        if "v4a_landscape_score" in sub.columns:
            sub = sub.sort_values("v4a_landscape_score", ascending=False)
        parents.append(sub.head(budgets.get(cls, 100)))
    parents = pd.concat(parents, ignore_index=True).drop_duplicates("sequence").head(args.max_parents)

    variants = []
    per_class_variants = {
        "A_elite": 5,
        "B_near_pass": 7,
        "C_high_novelty": 7,
        "D_broad_spectrum": 6,
        "E_worst_case_robust": 6,
        "F_developable": 6,
        "G_rescue": 8,
    }
    for row in parents.itertuples(index=False):
        cls = getattr(row, "v4a_class", "G_rescue")
        variants.extend(make_variants(row, rng, per_class_variants.get(cls, 5)))
        if len(variants) >= args.max_variants:
            break

    out = pd.DataFrame(variants).drop_duplicates("sequence").head(args.max_variants)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    g = out[out["parent_v4a_class"].eq("G_rescue")].copy()
    g_path = Path(args.g_output)
    g_path.parent.mkdir(parents=True, exist_ok=True)
    g.to_csv(g_path, index=False)

    summary = {
        "landscape": args.landscape,
        "parents_used": int(len(parents)),
        "variants_created": int(len(out)),
        "g_rescue_variants": int(len(g)),
        "output": str(out_path),
        "g_output": str(g_path),
    }
    summary_path = out_path.parent / "optimized_variants_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
