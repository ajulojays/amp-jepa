#!/usr/bin/env python3
"""Select diversity-preserving survivors after scoring a V4B generation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from evolution_core import align_latents, infer_fitness, load_latents, stratified_select, utc_now, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scored-candidates", required=True)
    parser.add_argument("--candidate-latents", required=True)
    parser.add_argument("--generation", type=int, required=True)
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--n-survivors", type=int, default=2048)
    parser.add_argument("--fitness-column", default=None)
    parser.add_argument("--latent-key", default="auto")
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    g = int(args.generation)
    gtag = f"generation_{g:02d}"
    outdir = Path(args.outdir or f"v4b/results/{gtag}")
    survivors_path = outdir / f"{gtag}_survivors.csv"
    survivor_latents_path = outdir / f"{gtag}_survivor_latents.npz"
    summary_path = outdir / f"{gtag}_survivor_selection_summary.json"

    if any(path.exists() for path in (survivors_path, survivor_latents_path, summary_path)) and not args.overwrite:
        raise FileExistsError(f"Survivor outputs already exist for {gtag}. Use --overwrite deliberately.")

    candidates = pd.read_csv(args.scored_candidates, low_memory=False)
    if "candidate_id" not in candidates.columns or "sequence" not in candidates.columns:
        raise ValueError("Scored candidates must contain candidate_id and sequence columns.")
    candidates["candidate_id"] = candidates["candidate_id"].astype(str)

    latent_ids, latent_matrix, latent_key = load_latents(args.candidate_latents, args.latent_key)
    candidate_z = align_latents(candidates, latent_ids, latent_matrix)
    fitness, fitness_source, fitness_mode = infer_fitness(candidates, args.fitness_column)

    survivors = stratified_select(
        df=candidates,
        z=candidate_z,
        n=args.n_survivors,
        fractions={"elite": 0.35, "diverse": 0.40, "frontier": 0.15, "random": 0.10},
        fitness=fitness,
        seed=args.seed + g,
        label_prefix="survivor_selection",
    )
    survivors["fitness_source"] = fitness_source
    survivors["fitness_mode"] = fitness_mode
    survivors["source_generation"] = g

    selected_rows = [np.where(latent_ids.astype(str) == cid)[0][0] for cid in survivors["candidate_id"].astype(str)]
    survivor_z = latent_matrix[np.asarray(selected_rows, dtype=int)].astype(np.float32, copy=False)

    outdir.mkdir(parents=True, exist_ok=True)
    survivors.to_csv(survivors_path, index=False)
    np.savez_compressed(
        survivor_latents_path,
        candidate_id=survivors["candidate_id"].astype(str).to_numpy(),
        z=survivor_z,
        generation=np.full(len(survivors), g, dtype=np.int16),
    )

    summary = {
        "schema_version": "1.0",
        "created_utc": utc_now(),
        "generation": g,
        "scored_candidates": str(args.scored_candidates),
        "candidate_latents": str(args.candidate_latents),
        "latent_key": latent_key,
        "candidate_count": int(len(candidates)),
        "selected_survivors": int(len(survivors)),
        "fitness_source": fitness_source,
        "fitness_mode": fitness_mode,
        "survivor_stratum_counts": survivors["survivor_selection_stratum"].value_counts().to_dict(),
        "outputs": {
            "survivors": str(survivors_path),
            "survivor_latents": str(survivor_latents_path),
        },
    }
    write_json(summary_path, summary)
    print(summary)


if __name__ == "__main__":
    main()
