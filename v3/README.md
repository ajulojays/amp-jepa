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
├── 00_prepare_corpus.py
├── 01_train_v3_hybrid.py
├── 02_generate_candidates.py
├── 03_rank_candidates.py
└── data/
    └── external/
        └── apex_oracle_ranked_summary.csv
```

## Quick run

Edit the input file path in `run_v3_hybrid.sh`, then run:

```bash
cd /path/to/amp-jepa
pip install -r v3/requirements-v3.txt
bash v3/run_v3_hybrid.sh
```

Default expected input:

```text
v3/data/raw/peptides.fasta
```

You can replace this with APD, dbAMP, DRAMP, CAMPR, or a merged curated file.

## Outputs

```text
v3/data/processed/peptide_corpus_v3.csv
v3/checkpoints/amp_jepa_hybrid_v3.pt
v3/results/raw_candidates_v3.csv
v3/results/ranked_candidates_v3.csv
v3/results/apex_comparator_v3.csv
```

## Interpretation

v3 can honestly support this type of claim:

> AMP-JEPA-Hybrid v3 generates candidate AMPs that satisfy AMP-like design constraints and can be prioritized by novelty, diversity, developability, and external APEX-style oracle support.

It should not yet claim:

> AMP-JEPA-Hybrid v3 beats APEX experimentally.

That requires matching wet-lab MIC/hemolysis/cytotoxicity data or a shared benchmark with the same target labels.
