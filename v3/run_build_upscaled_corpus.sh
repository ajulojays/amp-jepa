#!/usr/bin/env bash
set -euo pipefail

# Place source FASTA/CSV/TSV files in this folder, for example:
#   v3/data/raw/corpus_sources/apd.fasta
#   v3/data/raw/corpus_sources/dbamp.csv
#   v3/data/raw/corpus_sources/dramp.fasta
#   v3/data/raw/corpus_sources/campr.tsv
#   v3/data/raw/corpus_sources/dbaasp.csv
SOURCE_DIR="${SOURCE_DIR:-v3/data/raw/corpus_sources}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-v3/data/processed/upscaled_peptide_corpus_v3}"

mkdir -p "$SOURCE_DIR" v3/data/processed

python v3/37_build_upscaled_corpus.py \
  --inputs "$SOURCE_DIR" \
  --output-prefix "$OUTPUT_PREFIX" \
  --min-len 8 \
  --max-len 64

printf '\nNext full v3 run with the upscaled corpus:\n'
printf '  V3_INPUTS="%s.fasta" bash v3/run_v3_hybrid.sh\n' "$OUTPUT_PREFIX"
