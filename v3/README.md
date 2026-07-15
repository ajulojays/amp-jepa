# AMP-JEPA-Hybrid v3

This folder preserves and improves the original AMP-JEPA design direction instead of replacing it with a pure JEPA foundation model.

## What v3 is

**AMP-JEPA-Hybrid v3** is a lightweight predictive-generative architecture for antimicrobial peptide design:

```text
curated AMP sequences
        ↓
sequence encoder
        ↓
VAE latent space
        ↓
JEPA-inspired masked-view latent prediction
        ↓
controlled candidate generation
        ↓
novelty / developability / diversity filters
        ↓
APEX external-comparator ranking
```

This is intentionally separate from the long-term teacher/student Stage 1 foundation route. v3 is the fast candidate-generation track.

## Why this exists

The earlier AMP-JEPA pilot was already producing plausible AMP candidates from around 20k sequences. That means the architecture had a usable signal. v3 hardens that architecture by adding:

- stronger corpus curation and duplicate removal;
- constrained VAE sequence generation;
- JEPA-inspired masked-view latent consistency;
- physicochemical feature awareness;
- novelty scoring against the training corpus;
- candidate filtering by length, charge, hydrophobic fraction, and redundancy;
- APEX/ApexOracle table comparison without treating APEX as ground truth.

## Files

```text
v3/
├── README.md
├── requirements-v3.txt
├── run_v3_hybrid.sh
├── ampjepa_hybrid_v3.py
├── 00_prepare_v3_corpus.py
├── 01_train_v3_hybrid.py
├── 02_generate_v3_candidates.py
├── 03_rank_v3_candidates.py
└── data/
    └── apex_oracle_ranked_summary.csv
```

## Quick start

Edit `INPUTS` in `run_v3_hybrid.sh` so it points to your 20k AMP file or FASTA files.

```bash
cd amp-jepa
pip install -r v3/requirements-v3.txt
bash v3/run_v3_hybrid.sh
```

## Expected outputs

```text
v3/outputs/peptide_corpus_v3.csv
v3/outputs/amp_jepa_hybrid_v3.pt
v3/outputs/generated_candidates_v3.csv
v3/outputs/ranked_candidates_v3.csv
v3/outputs/v3_summary.txt
```

## Honest interpretation

v3 can support claims like:

> AMP-JEPA-Hybrid v3 generates AMP-like, novel, diverse candidates that are computationally plausible and can be externally compared with APEX/ApexOracle predictions.

Do **not** claim it beats APEX unless the same peptides are evaluated under the same benchmark or validated experimentally.
