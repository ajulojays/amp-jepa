#!/usr/bin/env bash
# ==============================================================================
# AMP-JEPA-Hybrid v3: improved candidate-generation track
# ==============================================================================
set -euo pipefail

# Default input is the APD/raw peptide FASTA. To train on an upscaled corpus, run:
#   V3_INPUTS="v3/data/processed/upscaled_peptide_corpus_v3.fasta" bash v3/run_v3_hybrid.sh
# Multiple files may be passed as a space-separated string in V3_INPUTS.
if [[ -n "${V3_INPUTS:-}" ]]; then
  # shellcheck disable=SC2206
  INPUTS=(${V3_INPUTS})
else
  INPUTS=(
    "v3/data/raw/peptides.fasta"
  )
fi

CORPUS="${V3_CORPUS:-v3/data/processed/peptide_corpus_v3.csv}"
CHECKPOINT="${V3_CHECKPOINT:-v3/checkpoints/amp_jepa_hybrid_v3.pt}"
RAW_CANDIDATES="${V3_RAW_CANDIDATES:-v3/results/raw_candidates_v3.csv}"
RANKED_CANDIDATES="${V3_RANKED_CANDIDATES:-v3/results/ranked_candidates_v3.csv}"
TOP_PANEL="${V3_TOP_PANEL:-v3/results/top_panel_v3.csv}"
APEX_COMPARATOR="${V3_APEX_COMPARATOR:-v3/results/apex_comparator_v3.csv}"
APEX_SCORED_DIR="${V3_APEX_SCORED_DIR:-v3/results/apex_scored_v3}"
APEX_ROOT_DEFAULT="${APEX_ROOT:-/home/julojays/apex}"
TOP_PANEL_N="${V3_TOP_PANEL_N:-50}"
GENERATE_N="${V3_GENERATE_N:-5000}"
TRAIN_EPOCHS="${V3_TRAIN_EPOCHS:-30}"
BATCH_SIZE="${V3_BATCH_SIZE:-128}"

mkdir -p v3/data/raw v3/data/processed v3/checkpoints v3/results

printf '=== v3 input corpus files ===\n'
printf '  %s\n' "${INPUTS[@]}"

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
  --epochs "$TRAIN_EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --max-len 64

echo "=== v3-02: Generate candidates ==="
python v3/02_generate_candidates.py \
  --checkpoint "$CHECKPOINT" \
  --output "$RAW_CANDIDATES" \
  --n "$GENERATE_N" \
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
  --top "$TOP_PANEL_N"

if [[ "${RUN_APEX:-auto}" == "0" || "${RUN_APEX:-auto}" == "false" ]]; then
  echo "=== v3-25/v3-26: APEX MIC scoring skipped because RUN_APEX=${RUN_APEX} ==="
elif [[ -d "${APEX_ROOT_DEFAULT}/trained_models" ]]; then
  echo "=== v3-25: Score top v3 candidates with APEX MIC ensemble ==="
  python v3/25_score_v3_candidates_with_apex.py \
    --candidates "$TOP_PANEL" \
    --output-dir "$APEX_SCORED_DIR" \
    --oracle v3/data/external/apex_oracle_ranked_summary.csv \
    --apex-root "$APEX_ROOT_DEFAULT"

  echo "=== v3-26: Select APEX-aware final candidate panel ==="
  python v3/26_select_apex_aware_panel.py \
    --scored-candidates "$APEX_SCORED_DIR/apex_scored_v3_candidates.csv" \
    --output-dir "$APEX_SCORED_DIR" \
    --top-n 20
else
  echo "=== v3-25/v3-26: APEX MIC scoring skipped ==="
  echo "APEX model directory not found: ${APEX_ROOT_DEFAULT}/trained_models"
  echo "Set APEX_ROOT=/path/to/apex or run: bash v3/run_score_v3_apex.sh after APEX is available."
fi

echo "[DONE] v3 pipeline complete. Review:"
echo "  $CORPUS"
echo "  $RANKED_CANDIDATES"
echo "  $TOP_PANEL"
echo "  $APEX_COMPARATOR"
echo "  $APEX_SCORED_DIR/apex_scored_v3_candidates.csv     # if APEX was available"
echo "  $APEX_SCORED_DIR/apex_scored_v3_vs_oracle.csv     # if APEX was available"
echo "  $APEX_SCORED_DIR/apex_aware_top_panel_v3.csv      # if APEX was available"
echo "  $APEX_SCORED_DIR/apex_aware_top_panel_v3.fasta    # if APEX was available"
