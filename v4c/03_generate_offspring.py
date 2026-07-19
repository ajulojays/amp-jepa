#!/usr/bin/env python3
"""Run the frozen V4B offspring generator while assigning V4C candidate IDs."""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np


def load_module_from_path(module_name: str, module_path: Path):
    """Load a module safely, including support for dataclass type resolution."""
    if not module_path.exists():
        raise FileNotFoundError(f"Module not found: {module_path}")

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module: {module_path}")

    module = importlib.util.module_from_spec(spec)

    # dataclasses and postponed annotations resolve the defining module through
    # sys.modules while the class body is being executed. Register first.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(spec.name, None)
        raise

    return module


def load_v4b_generator():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "v4b" / "03_generate_offspring.py"
    return load_module_from_path("v4b_generate_offspring", module_path)


def load_v3_module_safe(repo_root: Path):
    """Replacement for the older V4B loader that omitted sys.modules registration."""
    module_path = repo_root / "v3" / "ampjepa_hybrid_v3.py"
    return load_module_from_path("ampjepa_hybrid_v3", module_path)


def v4c_stable_id(sequence: str, generation: int) -> str:
    digest = hashlib.sha256(sequence.encode("utf-8")).hexdigest()[:16]
    return f"V4C_G{generation:02d}_{digest}"


def cli_value(flag: str) -> str | None:
    """Return the value following one command-line flag without consuming argv."""
    try:
        index = sys.argv.index(flag)
    except ValueError:
        return None
    if index + 1 >= len(sys.argv):
        raise ValueError(f"Missing value after {flag}")
    return sys.argv[index + 1]


def normalize_candidate_id_archive(path: Path) -> None:
    """Rewrite local proposal IDs as fixed-width Unicode, never object dtype."""
    if not path.exists():
        raise FileNotFoundError(f"Proposal archive not found after generation: {path}")

    # This archive is produced locally by the trusted V4C generator. allow_pickle
    # is used only to read legacy object-dtype IDs so they can be normalized once.
    with np.load(path, allow_pickle=True) as archive:
        payload = {name: archive[name] for name in archive.files}

    if "candidate_id" not in payload:
        raise ValueError(f"Proposal archive lacks candidate_id: {path}")

    original_dtype = payload["candidate_id"].dtype
    payload["candidate_id"] = np.asarray(payload["candidate_id"], dtype=str)

    if original_dtype.kind != "O":
        return

    temporary = path.with_name(path.stem + ".normalized.tmp.npz")
    np.savez_compressed(temporary, **payload)
    os.replace(temporary, path)

    # Verify that future readers can use NumPy's safe default.
    with np.load(path, allow_pickle=False) as archive:
        archive["candidate_id"]

    print(f"[V4C] Normalized proposal candidate_id dtype: {original_dtype} -> {payload['candidate_id'].dtype}")


def main() -> None:
    generation_text = cli_value("--generation")
    outdir_text = cli_value("--outdir")

    module = load_v4b_generator()
    module.stable_id = v4c_stable_id
    module.load_v3_module = load_v3_module_safe
    module.main()

    if generation_text is not None and outdir_text is not None:
        generation = int(generation_text)
        prefix = f"generation_{generation:02d}"
        proposal_path = Path(outdir_text) / f"{prefix}_latent_proposals.npz"
        normalize_candidate_id_archive(proposal_path)


if __name__ == "__main__":
    main()
