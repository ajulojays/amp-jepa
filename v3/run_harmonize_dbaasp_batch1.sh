#!/usr/bin/env bash
set -euo pipefail

# Harmonize the first manually downloaded DBAASP batch.
# Expected input files:
#   peptides.csv
#   activity-against-target-species.csv
#   hemolytic-and-cytotoxic-activities.csv
#   peptides-antibiofilm-activities.csv
#   peptides-fasta.txt

INPUT_DIR="${DBAASP_BATCH1_INPUT_DIR:-v3/data/raw/corpus_sources/dbaasp1_batch1/raw}"
OUTPUT_DIR="${DBAASP_BATCH1_OUTPUT_DIR:-v3/data/raw/corpus_sources/dbaasp1_batch1/harmonized}"
BATCH_LABEL="${DBAASP_BATCH_LABEL:-dbaasp1_batch1}"

python v3/42_harmonize_dbaasp_manual_batch.py \
  --input-dir "$INPUT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --batch-label "$BATCH_LABEL"
