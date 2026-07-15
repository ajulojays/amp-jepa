#!/usr/bin/env bash
set -euo pipefail

python v3/38_download_corpus_sources.py \
  --output-dir v3/data/raw/corpus_sources \
  --build-corpus \
  --output-prefix v3/data/processed/upscaled_peptide_corpus_v3
