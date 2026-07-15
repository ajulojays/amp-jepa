#!/usr/bin/env python3
"""Stage 1D: export AMP-JEPA sequence embeddings from a trained checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from ampjepa.stage1_jepa import load_encoder_from_checkpoint, mean_pool, tokenize_sequence  # noqa: E402


class CorpusDataset(Dataset):
    def __init__(self, corpus: pd.DataFrame, max_len: int):
        self.corpus = corpus.reset_index(drop=True)
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.corpus)

    def __getitem__(self, idx: int):
        row = self.corpus.iloc[idx]
        tokens, _ = tokenize_sequence(str(row["sequence"]), self.max_len)
        return {
            "tokens": tokens,
            "peptide_id": str(row["peptide_id"]),
            "sequence": str(row["sequence"]),
        }


def collate(batch):
    return {
        "tokens": torch.stack([x["tokens"] for x in batch]),
        "peptide_id": [x["peptide_id"] for x in batch],
        "sequence": [x["sequence"] for x in batch],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="data/processed/stage1/peptide_corpus.csv")
    parser.add_argument("--checkpoint", default="checkpoints/stage1/amp_jepa_stage1.pt")
    parser.add_argument("--out-npz", default="results/stage1/amp_jepa_embeddings.npz")
    parser.add_argument("--out-meta", default="results/stage1/amp_jepa_embedding_metadata.csv")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    corpus_path = Path(args.corpus)
    checkpoint_path = Path(args.checkpoint)
    if not corpus_path.exists():
        raise SystemExit(f"[ERROR] Missing corpus: {corpus_path}")
    if not checkpoint_path.exists():
        raise SystemExit(f"[ERROR] Missing checkpoint: {checkpoint_path}")

    corpus = pd.read_csv(corpus_path)
    required = {"peptide_id", "sequence"}
    missing = required - set(corpus.columns)
    if missing:
        raise SystemExit(f"[ERROR] Corpus missing required columns: {sorted(missing)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config, encoder, _ = load_encoder_from_checkpoint(str(checkpoint_path), map_location=device)
    encoder.to(device)
    encoder.eval()

    loader = DataLoader(CorpusDataset(corpus, config.max_len), batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    embeddings: List[np.ndarray] = []
    ids: List[str] = []
    sequences: List[str] = []
    with torch.no_grad():
        for batch in loader:
            tokens = batch["tokens"].to(device)
            hidden = encoder(tokens)
            pooled = mean_pool(hidden, tokens)
            embeddings.append(pooled.detach().cpu().numpy())
            ids.extend(batch["peptide_id"])
            sequences.extend(batch["sequence"])

    matrix = np.concatenate(embeddings, axis=0) if embeddings else np.zeros((0, config.d_model), dtype=np.float32)

    out_npz = Path(args.out_npz)
    out_meta = Path(args.out_meta)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    out_meta.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(out_npz, embeddings=matrix, peptide_id=np.array(ids), sequence=np.array(sequences))

    meta = corpus.copy()
    meta = meta.set_index("peptide_id").loc[ids].reset_index()
    meta["embedding_index"] = range(len(meta))
    meta.to_csv(out_meta, index=False)

    print(f"[INFO] Exported embedding matrix: {matrix.shape}")
    print(f"[DONE] Wrote {out_npz}")
    print(f"[DONE] Wrote {out_meta}")


if __name__ == "__main__":
    main()
