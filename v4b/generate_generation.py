#!/usr/bin/env python3
"""Generate V4B offspring for any generation from selected latent-space parents."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from evolution_core import align_latents, basic_features, clean_sequence, is_valid_sequence, load_latents, utc_now, write_json


def load_v3_module(repo_root: Path):
    module_path = repo_root / "v3" / "ampjepa_hybrid_v3.py"
    if not module_path.exists():
        raise FileNotFoundError(f"V3 module not found: {module_path}")
    module_name = "ampjepa_hybrid_v3"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module specification: {module_path}")
    module = importlib.util.module_from_spec(spec)
    # Required for dataclasses: @dataclass resolves cls.__module__ through sys.modules.
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def resolve_device(requested: str) -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
    return torch.device(requested)


def stable_id(sequence: str, generation: int) -> str:
    return f"V4B_G{generation:02d}_" + hashlib.sha256(sequence.encode("utf-8")).hexdigest()[:16]


def load_exclusion_sequences(paths: list[str], parent_sequences: pd.Series) -> set[str]:
    excluded = {clean_sequence(seq) for seq in parent_sequences.astype(str)}
    for item in paths:
        path = Path(item)
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path, usecols=["sequence"], low_memory=False)
        except Exception:
            df = pd.read_csv(path, low_memory=False)
            if "sequence" not in df.columns:
                continue
        excluded.update(clean_sequence(seq) for seq in df["sequence"].astype(str))
    excluded.discard("")
    return excluded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parents", required=True)
    parser.add_argument("--parent-latents", required=True)
    parser.add_argument("--generation", type=int, required=True)
    parser.add_argument("--checkpoint", default="v3/checkpoints/amp_jepa_hybrid_v3_qc_core.pt")
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--exclude-csv", action="append", default=[])
    parser.add_argument("--latent-key", default="auto")
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

    g = int(args.generation)
    gtag = f"generation_{g:02d}"
    outdir = Path(args.outdir or f"v4b/results/{gtag}")
    candidates_path = outdir / f"{gtag}_candidates_pre_apex.csv"
    proposals_path = outdir / f"{gtag}_latent_proposals.npz"
    summary_path = outdir / f"{gtag}_generation_summary.json"

    if any(path.exists() for path in (candidates_path, proposals_path, summary_path)) and not args.overwrite:
        raise FileExistsError(f"Generation outputs already exist for {gtag}. Use --overwrite deliberately.")

    parents = pd.read_csv(args.parents, low_memory=False)
    required = {"candidate_id", "sequence"}
    missing = sorted(required - set(parents.columns))
    if missing:
        raise ValueError(f"Parent table is missing required columns: {missing}")
    parents["candidate_id"] = parents["candidate_id"].astype(str)

    latent_ids, latent_matrix, latent_key = load_latents(args.parent_latents, args.latent_key)
    parent_z = align_latents(parents, latent_ids, latent_matrix)

    checkpoint_path = Path(args.checkpoint)
    model, config, _ = module.load_v3_checkpoint(str(checkpoint_path), map_location=device)
    model.to(device)
    model.eval()

    rng = np.random.default_rng(args.seed + g)
    torch.manual_seed(args.seed + g)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed + g)

    target_proposals = max(args.n_offspring, int(np.ceil(args.n_offspring * args.proposal_multiplier)))
    parent_count = len(parents)
    parent_indices = rng.integers(0, parent_count, size=target_proposals)
    partner_indices = rng.integers(0, parent_count, size=target_proposals)
    crossover_mask = rng.random(target_proposals) < args.crossover_rate
    alpha = rng.beta(2.0, 2.0, size=target_proposals).astype(np.float32)

    base = parent_z[parent_indices].copy()
    if crossover_mask.any():
        mix = alpha[crossover_mask, None]
        base[crossover_mask] = (
            mix * parent_z[parent_indices[crossover_mask]]
            + (1.0 - mix) * parent_z[partner_indices[crossover_mask]]
        )

    latent_scale = parent_z.std(axis=0, keepdims=True).astype(np.float32)
    latent_scale[latent_scale < 1e-6] = 1.0
    noise = rng.normal(size=base.shape).astype(np.float32)
    proposal_z = base + args.latent_sigma * latent_scale * noise

    excluded_sequences = load_exclusion_sequences(args.exclude_csv, parents["sequence"])
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
            sampled = torch.multinomial(probs.view(-1, probs.shape[-1]), num_samples=1).view(probs.shape[0], probs.shape[1])

            sampled_np = sampled.cpu().numpy()
            lengths_np = predicted_lengths.cpu().numpy()
            for local, (ids, length) in enumerate(zip(sampled_np, lengths_np)):
                global_idx = start + local
                sequence = module.decode_ids(ids[: int(length)])
                sequence = clean_sequence(sequence)
                if not is_valid_sequence(sequence, min_len=args.min_length, max_len=args.max_length):
                    continue
                if sequence in excluded_sequences or sequence in accepted_sequences:
                    continue
                feats = basic_features(sequence)
                charge = int(feats["net_charge_KR_minus_DE"])
                hydro = float(feats["hydrophobic_fraction"])
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
                    "candidate_id": stable_id(sequence, g),
                    "generation": g,
                    "parent_candidate_id": str(parent["candidate_id"]),
                    "second_parent_candidate_id": str(partner["candidate_id"]) if is_cross else "",
                    "lineage_depth": int(parent.get("lineage_depth", 0)) + 1,
                    "generation_operator": "latent_crossover_mutation" if is_cross else "latent_mutation",
                    "parent_selection_stratum": str(parent.get("parent_selection_stratum", "unknown")),
                    "sequence": sequence,
                    "length": int(len(sequence)),
                    "net_charge_KR_minus_DE": charge,
                    "hydrophobic_fraction": hydro,
                    "latent_sigma": float(args.latent_sigma),
                    "decode_temperature": float(args.temperature),
                    "proposal_index": int(global_idx),
                })
                if len(accepted) >= args.n_offspring:
                    break
            if len(accepted) >= args.n_offspring:
                break

    if not accepted:
        raise RuntimeError(f"No valid {gtag} offspring were produced.")

    offspring = pd.DataFrame(accepted)
    z_output = np.stack(accepted_z).astype(np.float32, copy=False)
    outdir.mkdir(parents=True, exist_ok=True)
    offspring.to_csv(candidates_path, index=False)
    np.savez_compressed(
        proposals_path,
        candidate_id=offspring["candidate_id"].astype(str).to_numpy(),
        z=z_output,
        generation=np.full(len(offspring), g, dtype=np.int16),
    )

    summary = {
        "schema_version": "1.0",
        "created_utc": utc_now(),
        "generation": g,
        "parent_count": int(parent_count),
        "requested_offspring": int(args.n_offspring),
        "proposals_attempted": int(target_proposals),
        "accepted_offspring": int(len(offspring)),
        "acceptance_rate": float(len(offspring) / target_proposals),
        "latent_dimension": int(z_output.shape[1]),
        "parent_latent_key": latent_key,
        "seed": int(args.seed + g),
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
        "outputs": {"candidates_pre_apex": str(candidates_path), "latent_proposals": str(proposals_path)},
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, default=str))
    if len(offspring) < args.n_offspring:
        print(f"[V4B] Warning: accepted {len(offspring):,} of {args.n_offspring:,} requested offspring.")


if __name__ == "__main__":
    main()
