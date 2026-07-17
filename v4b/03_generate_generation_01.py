#!/usr/bin/env python3
"""Generate V4B Generation 1 offspring from selected Generation 0 parents."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def load_v3_module(repo_root: Path):
    module_path = repo_root / "v3" / "ampjepa_hybrid_v3.py"
    if not module_path.exists():
        raise FileNotFoundError(f"V3 module not found: {module_path}")
    spec = importlib.util.spec_from_file_location("ampjepa_hybrid_v3", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module specification: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_device(requested: str) -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
    return torch.device(requested)


def stable_id(sequence: str) -> str:
    return "V4B_G01_" + hashlib.sha256(sequence.encode("utf-8")).hexdigest()[:16]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parents", default="v4b/results/generation_01/generation_01_parents.csv")
    parser.add_argument("--generation0", default="v4b/results/generation_00/generation_00_candidates.csv")
    parser.add_argument("--latents", default="v4b/results/generation_00/latent_vectors.npz")
    parser.add_argument("--checkpoint", default="v3/checkpoints/amp_jepa_hybrid_v3_qc_core.pt")
    parser.add_argument("--outdir", default="v4b/results/generation_01")
    parser.add_argument("--n-offspring", type=int, default=10000)
    parser.add_argument("--proposal-multiplier", type=float, default=4.0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--latent-sigma", type=float, default=0.35)
    parser.add_argument("--crossover-rate", type=float, default=0.20)
    parser.add_argument("--min-length", type=int, default=10)
    parser.add_argument("--max-length", type=int, default=40)
    parser.add_argument("--min-charge", type=int, default=2)
    parser.add_argument("--max-charge", type=int, default=12)
    parser.add_argument("--min-hydrophobic", type=float, default=0.20)
    parser.add_argument("--max-hydrophobic", type=float, default=0.70)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    module = load_v3_module(repo_root)
    device = resolve_device(args.device)

    parents_path = Path(args.parents)
    g0_path = Path(args.generation0)
    latent_path = Path(args.latents)
    checkpoint_path = Path(args.checkpoint)
    outdir = Path(args.outdir)
    candidates_path = outdir / "generation_01_candidates_pre_apex.csv"
    proposals_path = outdir / "generation_01_latent_proposals.npz"
    summary_path = outdir / "generation_01_generation_summary.json"

    if any(path.exists() for path in (candidates_path, proposals_path, summary_path)) and not args.overwrite:
        raise FileExistsError("Generation 1 offspring outputs already exist. Use --overwrite deliberately.")

    parents = pd.read_csv(parents_path, low_memory=False)
    g0 = pd.read_csv(g0_path, usecols=["candidate_id", "sequence"], low_memory=False)
    latent = np.load(latent_path)
    latent_mu = latent["mu"].astype(np.float32, copy=False)
    latent_ids = latent["candidate_id"].astype(str)
    id_to_row = {candidate_id: row for row, candidate_id in enumerate(latent_ids)}

    required = {"candidate_id", "sequence", "parent_selection_stratum"}
    missing = sorted(required - set(parents.columns))
    if missing:
        raise ValueError(f"Parent table is missing required columns: {missing}")
    parent_rows = []
    for candidate_id in parents["candidate_id"].astype(str):
        if candidate_id not in id_to_row:
            raise ValueError(f"Parent candidate ID is absent from latent archive: {candidate_id}")
        parent_rows.append(id_to_row[candidate_id])
    parent_mu = latent_mu[np.asarray(parent_rows, dtype=int)]

    model, config, _ = module.load_v3_checkpoint(str(checkpoint_path), map_location=device)
    model.to(device)
    model.eval()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    target_proposals = max(args.n_offspring, int(np.ceil(args.n_offspring * args.proposal_multiplier)))
    parent_count = len(parents)
    parent_indices = rng.integers(0, parent_count, size=target_proposals)
    partner_indices = rng.integers(0, parent_count, size=target_proposals)
    crossover_mask = rng.random(target_proposals) < args.crossover_rate
    alpha = rng.beta(2.0, 2.0, size=target_proposals).astype(np.float32)

    base = parent_mu[parent_indices].copy()
    if crossover_mask.any():
        mix = alpha[crossover_mask, None]
        base[crossover_mask] = (
            mix * parent_mu[parent_indices[crossover_mask]]
            + (1.0 - mix) * parent_mu[partner_indices[crossover_mask]]
        )

    latent_scale = latent_mu.std(axis=0, keepdims=True).astype(np.float32)
    latent_scale[latent_scale < 1e-6] = 1.0
    noise = rng.normal(size=base.shape).astype(np.float32)
    proposal_z = base + args.latent_sigma * latent_scale * noise

    existing_sequences = set(g0["sequence"].astype(str).str.upper().str.replace(r"\s+", "", regex=True))
    accepted: list[dict] = []
    accepted_sequences: set[str] = set()
    accepted_z: list[np.ndarray] = []

    with torch.inference_mode():
        for start in range(0, target_proposals, args.batch_size):
            stop = min(start + args.batch_size, target_proposals)
            z_tensor = torch.from_numpy(proposal_z[start:stop]).to(device)
            logits, length_norm = model.decode_logits(z_tensor)
            predicted_lengths = torch.round(length_norm * config.max_len).long()
            predicted_lengths = predicted_lengths.clamp(args.min_length, min(args.max_length, config.max_len))

            scaled_logits = logits / max(args.temperature, 1e-6)
            probs = torch.softmax(scaled_logits, dim=-1)
            probs[:, :, :5] = 0.0
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            sampled = torch.multinomial(
                probs.view(-1, probs.shape[-1]), num_samples=1
            ).view(probs.shape[0], probs.shape[1])

            sampled_np = sampled.cpu().numpy()
            lengths_np = predicted_lengths.cpu().numpy()
            for local, (ids, length) in enumerate(zip(sampled_np, lengths_np)):
                global_idx = start + local
                sequence = module.decode_ids(ids[: int(length)])
                sequence = module.clean_sequence(sequence)
                if not module.is_valid_sequence(sequence, min_len=args.min_length, max_len=args.max_length):
                    continue
                if sequence in existing_sequences or sequence in accepted_sequences:
                    continue

                features = module.candidate_score(sequence, max_train_identity=0.0)
                charge = int(features["net_charge_KR_minus_DE"])
                hydro = float(features["hydrophobic_fraction"])
                if not (args.min_charge <= charge <= args.max_charge):
                    continue
                if not (args.min_hydrophobic <= hydro <= args.max_hydrophobic):
                    continue

                pidx = int(parent_indices[global_idx])
                qidx = int(partner_indices[global_idx])
                is_cross = bool(crossover_mask[global_idx])
                parent = parents.iloc[pidx]
                partner = parents.iloc[qidx]
                accepted_sequences.add(sequence)
                accepted_z.append(proposal_z[global_idx].copy())
                accepted.append({
                    "candidate_id": stable_id(sequence),
                    "generation": 1,
                    "parent_candidate_id": str(parent["candidate_id"]),
                    "second_parent_candidate_id": str(partner["candidate_id"]) if is_cross else "",
                    "lineage_depth": int(parent.get("lineage_depth", 0)) + 1,
                    "generation_operator": "latent_crossover_mutation" if is_cross else "latent_mutation",
                    "parent_selection_stratum": str(parent["parent_selection_stratum"]),
                    "sequence": sequence,
                    "length": int(len(sequence)),
                    "net_charge_KR_minus_DE": charge,
                    "hydrophobic_fraction": hydro,
                    "v3_rank_score_pre_apex": float(features["v3_rank_score"]),
                    "latent_sigma": float(args.latent_sigma),
                    "decode_temperature": float(args.temperature),
                    "proposal_index": int(global_idx),
                })
                if len(accepted) >= args.n_offspring:
                    break
            if len(accepted) >= args.n_offspring:
                break

    if not accepted:
        raise RuntimeError("No valid Generation 1 offspring were produced.")

    offspring = pd.DataFrame(accepted)
    z_output = np.stack(accepted_z).astype(np.float32, copy=False)
    outdir.mkdir(parents=True, exist_ok=True)
    offspring.to_csv(candidates_path, index=False)
    np.savez_compressed(
        proposals_path,
        candidate_id=offspring["candidate_id"].astype(str).to_numpy(),
        z=z_output,
        generation=np.ones(len(offspring), dtype=np.int16),
    )

    summary = {
        "schema_version": "1.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "generation": 1,
        "parent_count": int(parent_count),
        "requested_offspring": int(args.n_offspring),
        "proposals_attempted": int(target_proposals),
        "accepted_offspring": int(len(offspring)),
        "acceptance_rate": float(len(offspring) / target_proposals),
        "latent_dimension": int(z_output.shape[1]),
        "seed": int(args.seed),
        "parameters": {
            "temperature": float(args.temperature),
            "latent_sigma": float(args.latent_sigma),
            "crossover_rate": float(args.crossover_rate),
            "min_length": int(args.min_length),
            "max_length": int(args.max_length),
            "min_charge": int(args.min_charge),
            "max_charge": int(args.max_charge),
            "min_hydrophobic": float(args.min_hydrophobic),
            "max_hydrophobic": float(args.max_hydrophobic),
        },
        "operator_counts": offspring["generation_operator"].value_counts().to_dict(),
        "parent_stratum_counts": offspring["parent_selection_stratum"].value_counts().to_dict(),
        "outputs": {
            "candidates_pre_apex": str(candidates_path),
            "latent_proposals": str(proposals_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))
    if len(offspring) < args.n_offspring:
        print(
            f"[V4B] Warning: accepted {len(offspring):,} of {args.n_offspring:,} requested offspring. "
            "Increase --proposal-multiplier or relax filters if needed."
        )


if __name__ == "__main__":
    main()
