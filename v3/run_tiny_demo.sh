#!/usr/bin/env bash
set -euo pipefail
python v3/08_make_demo_fasta.py
python v3/00_prepare_corpus.py --inputs v3/data/raw/peptides.fasta --output v3/data/processed/peptide_corpus_v3.csv
python v3/05_smoke_test_apex_only.py
