# AMP-JEPA-Hybrid V4C: million-peptide scale-up

V4C is the controlled scale-up of V4B. It keeps the V4B model, evolutionary schedule, APEX organism panel, physicochemical guardrails, parent bottleneck, survivor bottleneck, ranking logic and downstream novelty filters fixed while increasing the generated population from 10,000 to 100,000 peptides per generation.

## Primary experiment

```text
V4B: 10 generations × 10,000 offspring  = 100,000 generated peptides
V4C: 10 generations × 100,000 offspring = up to 1,000,000 generated peptides
```

The principal experimental variable is search scale.

## Frozen V4B settings

V4C deliberately retains:

- the same V3 checkpoint and latent decoder;
- 512 selected parents per generation;
- 2,500 survivors per generation;
- the same latent-sigma schedule, decoding-temperature schedule and crossover rate;
- the same length, charge and hydrophobicity generation guardrails;
- the same 34 APEX pathogen/strain models;
- the same final median predicted MIC cutoff of 32 µM;
- the same broad-curated and QC-corpus novelty threshold of less than 75% identity;
- the same final self-nonredundancy threshold of less than 75% identity.

Keeping the parent and survivor bottlenecks fixed makes V4C a stricter search-scale experiment: ten times more candidates compete for the same evolutionary bottleneck.

## Directory structure

```text
v4c/
├── README.md
├── configs/v4c_scale.yaml
├── v4c_config.env.example
├── 00_initialize_from_v4b.py
├── 03_generate_offspring.py
└── run_v4c_generations.sh

v4c/results/
├── generation_00/
├── generation_01/
├── ...
└── generation_10/
```

V4C outputs are isolated under `v4c/results` and never overwrite V4B.

## 1. Pull the workflow

```bash
cd ~/amp-jepa
git pull origin v3-hybrid-improved
conda activate ampjepa
```

## 2. Freeze and link the V4B Generation 0 baseline

```bash
python v4c/00_initialize_from_v4b.py \
  --source-root v4b/results/generation_00 \
  --target-root v4c/results/generation_00 \
  --mode symlink
```

The initializer verifies candidate IDs, row counts and latent alignment and writes a SHA-256 baseline manifest.

## 3. Configure APEX once

```bash
cp v4c/v4c_config.env.example v4c/v4c_config.env
nano v4c/v4c_config.env
```

Set `APEX_SCORER_CMD` to the exact scorer command already used successfully for V4B. The command must preserve `candidate_id` or `sequence` and return one row for every input peptide.

## 4. Optional isolated smoke test

This does not touch the full V4C result directory.

```bash
set -a
source v4c/v4c_config.env
set +a

START_GENERATION=1 \
END_GENERATION=1 \
N_OFFSPRING=1000 \
RESULTS_ROOT=v4c/smoke_results \
bash v4c/run_v4c_generations.sh
```

Initialize `v4c/smoke_results/generation_00` separately before this smoke test, or skip directly to the full run after validating the configuration.

## 5. Run the full V4C search

```bash
set -a
source v4c/v4c_config.env
set +a

mkdir -p v4c/results/logs
nohup bash v4c/run_v4c_generations.sh \
  > v4c/results/logs/v4c_generations_01_to_10.log 2>&1 &

tail -f v4c/results/logs/v4c_generations_01_to_10.log
```

The runner is resumable. A completed generation is skipped only when its completion marker, survivor table and latent archive are all present.

## Expected scale

At the target maximum:

- generated peptides: up to 1,000,000;
- pathogen-specific APEX predictions: up to 34,000,000;
- archived generation candidate tables: 10;
- final downstream filtering: applied only after Generation 10 using the frozen V4B criteria.

## Final V4C funnel

```text
Up to 1,000,000 generated peptides
        ↓
Canonical and physicochemical QC
        ↓
Elite/Pareto portfolio selection
        ↓
Broad-curated novelty <75% identity
        ↓
QC-corpus novelty <75% identity
        ↓
Median predicted MIC ≤32 µM
        ↓
Self-nonredundancy <75% identity
        ↓
Final V4C portfolio
        ↓
Global and species-specific MIC plots
```

All MIC values remain computational predictions until experimentally measured.