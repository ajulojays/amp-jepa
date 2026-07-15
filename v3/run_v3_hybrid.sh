#!/usr/bin/env bash
# ==============================================================================
# AMP-JEPA-Hybrid v3: improved candidate-generation track
# ==============================================================================
set -euo pipefail

# Replace this with your merged APD/dbAMP/DRAMP/CAMPR file or FASTA.
INPUTS=(
  "v3/data/raw/peptides.fasta"
)

CORPUS="v3/data/processed/peptide_corpus_v3.csv"
CHECKPOINT="v3/checkpoints/amp_jepa_hybrid_v3.pt"
RAW_CANDIDATES="v3/results/raw_candidates_v3.csv"
RANKED_CANDIDATES="v3/results/ranked_candidates_v3.csv"
TOP_PANEL="v3/results/top_panel_v3.csv"
APEX_COMPARATOR="v3/results/apex_comparator_v3.csv"

mkdir -p v3/data/raw v3/data/processed v3/checkpoints v3/results

echo "=== v3-00: Prepare curated peptide corpus ==="
python v3/00_prepare_corpus.py \
  --inputs "${INPUTS[@]}" \
  --output "$CORPUS" \
  --min-len 8 \
  --max-len 64

echo "=== v3-01: Train AMP-JEPA-Hybrid v3 ==="
python v3/01_train_v3_hybrid.py \
  --corpus "$CORPUS" \
  --checkpoint "$CHECKPOINT" \
  --epochs 30 \
  --batch-size 128 \
  --max-len 64

echo "=== v3-02: Generate candidates ==="
python v3/02_generate_candidates.py \
  --checkpoint "$CHECKPOINT" \
  --output "$RAW_CANDIDATES" \
  --n 5000 \
  --temperature 0.9

echo "=== v3-03: Rank candidates ==="
python v3/03_rank_candidates.py \
  --candidates "$RAW_CANDIDATES" \
  --corpus "$CORPUS" \
  --output "$RANKED_CANDIDATES"

echo "=== v3-04: Compare with APEX oracle table ==="
python v3/04_compare_apex_oracle.py \
  --ranked "$RANKED_CANDIDATES" \
  --apex v3/data/external/apex_oracle_ranked_summary.csv \
  --output "$APEX_COMPARATOR"

echo "=== v3-06: Export top candidate panel ==="
python v3/06_make_top_panel.py \
  --ranked "$RANKED_CANDIDATES" \
  --output "$TOP_PANEL" \
  --top 50

echo "[DONE] v3 pipeline complete. Review:"
echo "  $RANKED_CANDIDATES"
echo "  $TOP_PANEL"
echo "  $APEX_COMPARATOR"
