#!/usr/bin/env bash
set -euo pipefail

# Default mode downloads stable direct public sources:
#   - APD2024a natural AMPs
#   - UniProt reviewed short antimicrobial entries
#
# Aggressive mode adds broader/noisier sources:
#   INCLUDE_UNIPROT_EXPANDED=1     broader UniProt AMP-family queries
#   INCLUDE_PUBLIC_ML_REPOS=1      selected public AMP benchmark GitHub repos
#   PERMISSIVE_ARCHIVE_EXTRACTION=1 extract more CSV/TSV/TXT/FASTA files from repo zips
#
# Optional length knobs for the built corpus:
#   UPSCALED_MIN_LEN=8
#   UPSCALED_MAX_LEN=64   # keep 64 for current v3 model; use 100 only for exploratory corpus audit

ARGS=(
  --output-dir v3/data/raw/corpus_sources
  --build-corpus
  --output-prefix v3/data/processed/upscaled_peptide_corpus_v3
  --min-len "${UPSCALED_MIN_LEN:-8}"
  --max-len "${UPSCALED_MAX_LEN:-64}"
)

if [[ "${INCLUDE_UNIPROT_EXPANDED:-0}" == "1" || "${INCLUDE_UNIPROT_EXPANDED:-0}" == "true" ]]; then
  ARGS+=(--include-uniprot-expanded)
fi

if [[ "${INCLUDE_PUBLIC_ML_REPOS:-0}" == "1" || "${INCLUDE_PUBLIC_ML_REPOS:-0}" == "true" ]]; then
  ARGS+=(--include-public-ml-repos)
fi

if [[ "${PERMISSIVE_ARCHIVE_EXTRACTION:-0}" == "1" || "${PERMISSIVE_ARCHIVE_EXTRACTION:-0}" == "true" ]]; then
  ARGS+=(--permissive-archive-extraction)
fi

if [[ "${OVERWRITE_CORPUS_DOWNLOADS:-0}" == "1" || "${OVERWRITE_CORPUS_DOWNLOADS:-0}" == "true" ]]; then
  ARGS+=(--overwrite)
fi

python v3/38_download_corpus_sources.py "${ARGS[@]}"
