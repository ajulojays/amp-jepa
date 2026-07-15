#!/usr/bin/env bash
set -euo pipefail

# Default mode downloads stable direct public sources:
#   - APD2024a natural AMPs
#   - UniProt reviewed short antimicrobial entries
#
# Set INCLUDE_PUBLIC_ML_REPOS=1 to additionally download selected public
# GitHub AMP benchmark repositories and extract likely positive corpus files.

ARGS=(
  --output-dir v3/data/raw/corpus_sources
  --build-corpus
  --output-prefix v3/data/processed/upscaled_peptide_corpus_v3
)

if [[ "${INCLUDE_PUBLIC_ML_REPOS:-0}" == "1" || "${INCLUDE_PUBLIC_ML_REPOS:-0}" == "true" ]]; then
  ARGS+=(--include-public-ml-repos)
fi

python v3/38_download_corpus_sources.py "${ARGS[@]}"
