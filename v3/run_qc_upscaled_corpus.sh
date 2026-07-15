#!/usr/bin/env bash
set -euo pipefail

# QC the merged upscaled AMP corpus before retraining.
#
# Defaults are conservative enough to remove obvious archive/noise artifacts while
# preserving most AMP-like sequences. Override knobs when needed:
#   QC_MIN_TRAIN_LENGTH=10
#   QC_MAX_TRAIN_LENGTH=50
#   QC_MIN_ENTROPY=1.5
#   QC_MAX_HYDRO=0.70

python v3/40_qc_upscaled_corpus.py \
  --corpus "${QC_CORPUS:-v3/data/processed/upscaled_peptide_corpus_v3.csv}" \
  --output-dir "${QC_OUTPUT_DIR:-v3/results/upscaled_corpus_qc}" \
  --min-train-length "${QC_MIN_TRAIN_LENGTH:-10}" \
  --max-train-length "${QC_MAX_TRAIN_LENGTH:-50}" \
  --min-entropy "${QC_MIN_ENTROPY:-1.5}" \
  --max-homopolymer "${QC_MAX_HOMOPOLYMER:-6}" \
  --max-hydrophobic-fraction "${QC_MAX_HYDRO:-0.70}" \
  --max-abs-charge "${QC_MAX_ABS_CHARGE:-15}" \
  --max-cysteines "${QC_MAX_CYSTEINES:-8}" \
  --max-tryptophans "${QC_MAX_TRYPTOPHANS:-6}"
