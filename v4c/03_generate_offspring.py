#!/usr/bin/env python3
"""Run the frozen V4B offspring generator while assigning V4C candidate IDs."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path


def load_v4b_generator():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "v4b" / "03_generate_offspring.py"
    if not module_path.exists():
        raise FileNotFoundError(f"V4B offspring generator not found: {module_path}")

    spec = importlib.util.spec_from_file_location("v4b_generate_offspring", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load V4B offspring generator: {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def v4c_stable_id(sequence: str, generation: int) -> str:
    digest = hashlib.sha256(sequence.encode("utf-8")).hexdigest()[:16]
    return f"V4C_G{generation:02d}_{digest}"


def main() -> None:
    module = load_v4b_generator()
    module.stable_id = v4c_stable_id
    module.main()


if __name__ == "__main__":
    main()
