#!/usr/bin/env bash
# ==============================================================================
# STAGE 1: AMP-JEPA Embedding Foundation
# ==============================================================================
set -euo pipefail

# Edit this to point at your current peptide source files.
# You can pass multiple FASTA/CSV/TSV files after --inputs.
INPUTS=(
  "data/raw/apd/naturalAMPs_APD2024a.fasta"
)

CORPUS="data/processed/stage1/peptide_corpus.csv"
PAIRS="data/processed/stage1/jepa_pairs.csv"
CHECKPOINT="checkpoints/stage1/amp_jepa_stage1.pt"
EMBED_NPZ="results/stage1/amp_jepa_embeddings.npz"
EMBED_META="results/stage1/amp_jepa_embedding_metadata.csv"
BENCHMARK="results/stage1/embedding_benchmark_summary.csv"
APEX_OUT="results/stage1/apex_oracle_external_comparator.csv"

mkdir -p data/processed/stage1 results/stage1 checkpoints/stage1

echo "=== Stage 1A: Curate peptide corpus ==="
python scripts/stage1/01_curate_peptide_corpus.py \
  --inputs "${INPUTS[@]}" \
  --output "$CORPUS"

echo "=== Stage 1B: Build JEPA context/target pairs ==="
python scripts/stage1/02_build_jepa_pairs.py \
  --corpus "$CORPUS" \
  --output "$PAIRS" \
  --pairs-per-sequence 6

echo "=== Stage 1C: Train AMP-JEPA encoder ==="
python scripts/stage1/03_train_amp_jepa_encoder.py \
  --pairs "$PAIRS" \
  --checkpoint "$CHECKPOINT" \
  --epochs 20 \
  --batch-size 128

echo "=== Stage 1D: Export embeddings ==="
python scripts/stage1/04_export_embeddings.py \
  --corpus "$CORPUS" \
  --checkpoint "$CHECKPOINT" \
  --out-npz "$EMBED_NPZ" \
  --out-meta "$EMBED_META"

echo "=== Stage 1E: Benchmark embeddings ==="
python scripts/stage1/05_benchmark_embeddings_vs_esm2.py \
  --embedding-npz "$EMBED_NPZ" \
  --metadata "$EMBED_META" \
  --output "$BENCHMARK"

echo "=== Stage 1F: APEX external comparator ==="
python scripts/stage1/06_compare_apex_oracle.py \
  --apex data/external/apex_oracle_ranked_summary.csv \
  --output "$APEX_OUT"

echo "[DONE] Stage 1 complete. Outputs are in results/stage1/"
