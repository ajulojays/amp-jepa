#!/usr/bin/env python3
"""Select parents for any V4B generation from a population table and latent archive."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from evolution_core import align_latents, infer_fitness, load_latents, stratified_select, utc_now, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population", required=True, help="Population CSV to sample parents from.")
    parser.add_argument("--latents", required=True, help="NPZ with candidate_id and mu/z latents for the population.")
    parser.add_argument("--generation", type=int, required=True, help="Generation being produced from these parents.")
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--n-parents", type=int, default=512)
    parser.add_argument("--fitness-column", default=None)
    parser.add_argument("--latent-key", default="auto")
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    g = int(args.generation)
    gtag = f"generation_{g:02d}"
    outdir = Path(args.outdir or f"v4b/results/{gtag}")
    parents_path = outdir / f"{gtag}_parents.csv"
    summary_path = outdir / f"{gtag}_parent_selection_summary.json"

    if (parents_path.exists() or summary_path.exists()) and not args.overwrite:
        raise FileExistsError(f"Parent outputs already exist for {gtag}. Use --overwrite deliberately.")

    population = pd.read_csv(args.population, low_memory=False)
    if "candidate_id" not in population.columns:
        raise ValueError("Population table must contain candidate_id.")
    if "sequence" not in population.columns:
        raise ValueError("Population table must contain sequence.")
    population["candidate_id"] = population["candidate_id"].astype(str)

    latent_ids, latent_matrix, latent_key = load_latents(args.latents, args.latent_key)
    aligned_z = align_latents(population, latent_ids, latent_matrix)
    fitness, fitness_source, fitness_mode = infer_fitness(population, args.fitness_column)

    parents = stratified_select(
        df=population,
        z=aligned_z,
        n=args.n_parents,
        fractions={"elite": 0.25, "diverse": 0.45, "frontier": 0.20, "random": 0.10},
        fitness=fitness,
        seed=args.seed + g,
        label_prefix="parent_selection",
    )
    parents.insert(0, "target_generation", g)
    parents["source_population"] = str(args.population)
    parents["source_latents"] = str(args.latents)
    parents["source_latent_key"] = latent_key
    parents["fitness_source"] = fitness_source
    parents["fitness_mode"] = fitness_mode

    outdir.mkdir(parents=True, exist_ok=True)
    parents.to_csv(parents_path, index=False)

    summary = {
        "schema_version": "1.0",
        "created_utc": utc_now(),
        "target_generation": g,
        "population": str(args.population),
        "latents": str(args.latents),
        "latent_key": latent_key,
        "population_size": int(len(population)),
        "selected_parents": int(len(parents)),
        "fitness_source": fitness_source,
        "fitness_mode": fitness_mode,
        "parent_stratum_counts": parents["parent_selection_stratum"].value_counts().to_dict(),
        "outputs": {"parents": str(parents_path)},
    }
    write_json(summary_path, summary)
    print(summary)


if __name__ == "__main__":
    main()
