#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CANDIDATES="${CANDIDATES:-v4c/results/final_funnel/00_merged_qc/v4c_million_scored_qc_pass.csv}"
BROAD_REFERENCE="${BROAD_REFERENCE:-v3/data/processed/upscaled_peptide_corpus_v3.fasta}"
QC_CORE_REFERENCE="${QC_CORE_REFERENCE:-v3/data/processed/peptide_corpus_v3_qc_core.csv}"
BROAD_OUTDIR="${BROAD_OUTDIR:-v4c/results/final_funnel/01_novelty_broad_curated}"
QC_CORE_OUTDIR="${QC_CORE_OUTDIR:-v4c/results/final_funnel/02_novelty_qc_core}"
IDENTITY="${IDENTITY:-0.75}"
THREADS="${THREADS:-16}"
MEMORY_MB="${MEMORY_MB:-0}"
REST_SECONDS="${REST_SECONDS:-60}"

for required in "$CANDIDATES" "$BROAD_REFERENCE" "$QC_CORE_REFERENCE"; do
  if [[ ! -s "$required" ]]; then
    echo "[V4C-NOVELTY] Missing or empty required file: $required" >&2
    exit 2
  fi
done

mkdir -p "$BROAD_OUTDIR" "$QC_CORE_OUTDIR"

BROAD_SUMMARY="$BROAD_OUTDIR/v4c_manifest75_filter_summary.json"
QC_SUMMARY="$QC_CORE_OUTDIR/v4c_manifest75_filter_summary.json"

cat <<EOF
============================================================
V4C million-peptide dual-reference novelty screen
Candidates:       $CANDIDATES
Broad reference:  $BROAD_REFERENCE
QC-core reference:$QC_CORE_REFERENCE
Identity:         $IDENTITY
Threads:          $THREADS
Rest:             ${REST_SECONDS}s
============================================================
EOF

if [[ -s "$BROAD_SUMMARY" ]]; then
  echo "[V4C-NOVELTY] Broad-curated summary exists; skipping completed screen."
else
  echo "[V4C-NOVELTY] Starting broad-curated 75% identity screen"
  python v4c/22_filter_million_against_manifest_cdhit75.py \
    --candidates "$CANDIDATES" \
    --reference "$BROAD_REFERENCE" \
    --output-dir "$BROAD_OUTDIR" \
    --identity "$IDENTITY" \
    --threads "$THREADS" \
    --memory-mb "$MEMORY_MB" \
    --expected-candidates 1000000
fi

if [[ -s "$QC_SUMMARY" ]]; then
  echo "[V4C-NOVELTY] QC-core summary exists; skipping completed screen."
else
  if (( REST_SECONDS > 0 )); then
    echo "[V4C-NOVELTY] Resting ${REST_SECONDS}s before QC-core screen"
    sleep "$REST_SECONDS"
  fi

  echo "[V4C-NOVELTY] Starting QC-core 75% identity screen"
  python v4c/22_filter_million_against_manifest_cdhit75.py \
    --candidates "$CANDIDATES" \
    --reference "$QC_CORE_REFERENCE" \
    --output-dir "$QC_CORE_OUTDIR" \
    --identity "$IDENTITY" \
    --threads "$THREADS" \
    --memory-mb "$MEMORY_MB" \
    --expected-candidates 1000000
fi

echo "============================================================"
echo "[V4C-NOVELTY] Both reference screens completed"
echo "[V4C-NOVELTY] Broad:  $BROAD_SUMMARY"
echo "[V4C-NOVELTY] QC-core:$QC_SUMMARY"
echo "============================================================"
