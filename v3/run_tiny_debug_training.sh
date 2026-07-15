#!/usr/bin/env bash
set -euo pipefail
python v3/08_make_demo_fasta.py
python v3/00_prepare_corpus.py --inputs v3/data/raw/peptides.fasta --output v3/data/processed/peptide_corpus_v3.csv
python v3/09_train_tiny_debug.py
python v3/02_generate_candidates.py --checkpoint v3/checkpoints/amp_jepa_hybrid_v3_tiny_debug.pt --output v3/results/raw_candidates_v3_tiny_debug.csv --n 20
python v3/03_rank_candidates.py --candidates v3/results/raw_candidates_v3_tiny_debug.csv --corpus v3/data/processed/peptide_corpus_v3.csv --output v3/results/ranked_candidates_v3_tiny_debug.csv
