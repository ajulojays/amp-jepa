# Stage 1: AMP-JEPA Embedding Foundation

Stage 1 turns AMP-JEPA from a promising pilot into a reusable peptide embedding foundation workflow.

The goal is not yet to claim that AMP-JEPA beats APEX/ApexGO. The goal is to train, export, and benchmark an AMP-specific JEPA-derived embedding space against ordinary ESM2-style embeddings and external oracle predictions.

## Stage layout

```text
Stage 1A — Curate peptide corpus
Stage 1B — Build biologically masked JEPA pairs
Stage 1C — Train AMP-JEPA encoder
Stage 1D — Export embeddings
Stage 1E — Benchmark embeddings vs ESM2/baselines
Stage 1F — Use APEX table as external comparator
```

## Scientific question

Does JEPA-style masked latent prediction learn antimicrobial-peptide representations that are more useful than generic protein embeddings for AMP discovery, novelty, and parent-to-variant improvement?

## Key principle

APEX/ApexGO predictions are treated as an external computational comparator, not experimental ground truth. AMP-JEPA should first show that its embeddings organize AMP sequence space meaningfully, then later show whether its candidates outperform APEX-selected candidates experimentally.

## Recommended input folders

```text
data/raw/
  apd/
  dbamp/
  dramp/
  campr/
  ampsphere/

data/external/
  apex_oracle_ranked_summary.csv

data/processed/stage1/
results/stage1/
checkpoints/stage1/
```

## Minimal run order

From the repository root:

```bash
# 1A: curate a canonical corpus from one or more FASTA/CSV/TSV files
python scripts/stage1/01_curate_peptide_corpus.py \
  --inputs data/raw/apd/naturalAMPs_APD2024a.fasta \
  --output data/processed/stage1/peptide_corpus.csv

# 1B: build context/target masking pairs
python scripts/stage1/02_build_jepa_pairs.py \
  --corpus data/processed/stage1/peptide_corpus.csv \
  --output data/processed/stage1/jepa_pairs.csv \
  --pairs-per-sequence 6

# 1C: train a true teacher/student latent-prediction encoder
python scripts/stage1/03_train_amp_jepa_encoder.py \
  --pairs data/processed/stage1/jepa_pairs.csv \
  --checkpoint checkpoints/stage1/amp_jepa_stage1.pt \
  --epochs 20

# 1D: export sequence-level embeddings
python scripts/stage1/04_export_embeddings.py \
  --corpus data/processed/stage1/peptide_corpus.csv \
  --checkpoint checkpoints/stage1/amp_jepa_stage1.pt \
  --out-npz results/stage1/amp_jepa_embeddings.npz \
  --out-meta results/stage1/amp_jepa_embedding_metadata.csv

# 1E: run lightweight embedding diagnostics / supervised benchmarks when labels exist
python scripts/stage1/05_benchmark_embeddings_vs_esm2.py \
  --embedding-npz results/stage1/amp_jepa_embeddings.npz \
  --metadata results/stage1/amp_jepa_embedding_metadata.csv \
  --output results/stage1/embedding_benchmark_summary.csv

# 1F: rank the uploaded APEX oracle table as an external comparator
python scripts/stage1/06_compare_apex_oracle.py \
  --apex data/external/apex_oracle_ranked_summary.csv \
  --output results/stage1/apex_oracle_external_comparator.csv
```

## Interpretation rules

Use these rules to avoid overclaiming:

1. If AMP-JEPA candidates are also scored highly by APEX, say **APEX supports the computational plausibility** of the candidate.
2. Do not say AMP-JEPA beats APEX unless AMP-JEPA is tested on the same benchmark with a stronger metric or validated experimentally.
3. For Stage 1, the strongest claim is representation quality: family-held-out generalization, retrieval quality, and useful latent geometry.
4. Use AMPSphere and predicted catalogs for pretraining/mining, not as unquestioned experimental positives.
5. Keep external oracle files separate from training labels unless explicitly running weak-supervision experiments.

## Expected Stage 1 output

The stage is complete when the repository contains:

- a deduplicated canonical peptide corpus;
- JEPA context/target training pairs;
- a trained teacher/student AMP-JEPA encoder checkpoint;
- exported embeddings;
- embedding diagnostics and baseline comparisons;
- an APEX external-comparator table for the selected candidates.
