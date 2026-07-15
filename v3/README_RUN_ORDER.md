# v3 run order

For a quick wiring check:

```bash
pip install -r v3/requirements-v3.txt
bash v3/run_tiny_demo.sh
```

For a tiny debug train using the bundled APEX sequences only:

```bash
bash v3/run_tiny_debug_training.sh
```

## Download public corpus sources

The repo includes a downloader for public/direct corpus URLs. By default it downloads APD2024a natural AMPs into `v3/data/raw/corpus_sources/` and then builds the upscaled corpus.

```bash
bash v3/run_download_corpus_sources.sh
```

For a broader but noisier public-corpus build:

```bash
INCLUDE_UNIPROT_EXPANDED=1 \
INCLUDE_PUBLIC_ML_REPOS=1 \
PERMISSIVE_ARCHIVE_EXTRACTION=1 \
bash v3/run_download_corpus_sources.sh
```

Main download/build outputs:

```text
v3/data/raw/corpus_sources/apd2024a_natural_amps.fasta
v3/data/raw/corpus_sources/corpus_download_report.json
v3/data/processed/upscaled_peptide_corpus_v3.csv
v3/data/processed/upscaled_peptide_corpus_v3.fasta
v3/data/processed/upscaled_peptide_corpus_v3_source_summary.csv
v3/data/processed/upscaled_peptide_corpus_v3_report.json
```

For dbAMP, DRAMP, CAMPR/CAMP, DBAASP, StarPep, or local lab-curated exports, either place FASTA/CSV/TSV files in:

```text
v3/data/raw/corpus_sources/
```

or copy the template below, paste direct export URLs, set `enabled=true`, and rerun the downloader:

```bash
cp v3/data/raw/corpus_sources_manifest.example.tsv v3/data/raw/corpus_sources_manifest.local.tsv
nano v3/data/raw/corpus_sources_manifest.local.tsv

python v3/38_download_corpus_sources.py \
  --manifest v3/data/raw/corpus_sources_manifest.local.tsv \
  --output-dir v3/data/raw/corpus_sources \
  --build-corpus \
  --output-prefix v3/data/processed/upscaled_peptide_corpus_v3
```

Raw third-party corpus files are local working data and should generally remain uncommitted unless their license explicitly permits redistribution.

## Build an upscaled corpus from existing local files

Place local corpus source files here:

```text
v3/data/raw/corpus_sources/
```

Examples:

```text
v3/data/raw/corpus_sources/apd.fasta
v3/data/raw/corpus_sources/dbamp.csv
v3/data/raw/corpus_sources/dramp.fasta
v3/data/raw/corpus_sources/campr.tsv
v3/data/raw/corpus_sources/dbaasp.csv
```

Then run:

```bash
bash v3/run_build_upscaled_corpus.sh
```

Main upscaled-corpus outputs:

```text
v3/data/processed/upscaled_peptide_corpus_v3.csv
v3/data/processed/upscaled_peptide_corpus_v3.fasta
v3/data/processed/upscaled_peptide_corpus_v3_source_summary.csv
v3/data/processed/upscaled_peptide_corpus_v3_report.json
```

## QC the upscaled corpus before retraining

Run QC before training on broad public/benchmark corpus builds:

```bash
bash v3/run_qc_upscaled_corpus.sh
```

Main QC outputs:

```text
v3/results/upscaled_corpus_qc/upscaled_corpus_qc_table.csv
v3/results/upscaled_corpus_qc/upscaled_corpus_trainable_core.csv
v3/results/upscaled_corpus_qc/upscaled_corpus_trainable_core.fasta
v3/results/upscaled_corpus_qc/upscaled_corpus_review_or_excluded.csv
v3/results/upscaled_corpus_qc/source_qc_summary.csv
v3/results/upscaled_corpus_qc/length_bin_summary.csv
v3/results/upscaled_corpus_qc/upscaled_corpus_qc_report.json
v3/results/upscaled_corpus_qc/upscaled_corpus_qc_report.md
```

Use `upscaled_corpus_trainable_core.fasta` for the cleaner expanded-corpus v3 run. Keep the full corpus for audit and ablation.

## Run v3 on the QC-cleaned upscaled corpus

```bash
export APEX_ROOT=/home/julojays/apex
V3_INPUTS="v3/results/upscaled_corpus_qc/upscaled_corpus_trainable_core.fasta" \
V3_CORPUS="v3/data/processed/peptide_corpus_v3_upscaled_qc_core.csv" \
V3_CHECKPOINT="v3/checkpoints/amp_jepa_hybrid_v3_upscaled_qc_core.pt" \
V3_RAW_CANDIDATES="v3/results/raw_candidates_v3_upscaled_qc_core.csv" \
V3_RANKED_CANDIDATES="v3/results/ranked_candidates_v3_upscaled_qc_core.csv" \
V3_TOP_PANEL="v3/results/top_panel_v3_upscaled_qc_core_500.csv" \
V3_TOP_PANEL_N=500 \
V3_APEX_SCORED_DIR="v3/results/apex_scored_v3_upscaled_qc_core_500" \
bash v3/run_v3_hybrid.sh
```

## Run v3 on the raw upscaled corpus

Use this only for ablation against the cleaner QC core:

```bash
export APEX_ROOT=/home/julojays/apex
V3_INPUTS="v3/data/processed/upscaled_peptide_corpus_v3.fasta" \
V3_CORPUS="v3/data/processed/peptide_corpus_v3_upscaled.csv" \
V3_CHECKPOINT="v3/checkpoints/amp_jepa_hybrid_v3_upscaled.pt" \
V3_RAW_CANDIDATES="v3/results/raw_candidates_v3_upscaled.csv" \
V3_RANKED_CANDIDATES="v3/results/ranked_candidates_v3_upscaled.csv" \
V3_TOP_PANEL="v3/results/top_panel_v3_upscaled_500.csv" \
V3_TOP_PANEL_N=500 \
V3_APEX_SCORED_DIR="v3/results/apex_scored_v3_upscaled_500" \
bash v3/run_v3_hybrid.sh
```

## Standard APD/default v3 run

```bash
# first place your curated APD FASTA here or edit V3_INPUTS
# v3/data/raw/peptides.fasta
export APEX_ROOT=/home/julojays/apex
bash v3/run_v3_hybrid.sh
```

If your local APEX checkout exists at `$APEX_ROOT` or `/home/julojays/apex`, `run_v3_hybrid.sh` automatically:

1. trains v3,
2. generates candidates,
3. ranks them by v3 heuristic filters,
4. scores the top panel with the APEX MIC ensemble,
5. selects an APEX-aware final panel using MIC, breadth, worst-case MIC, novelty, and sequence developability filters.

To run only the v3 APEX/MIC scoring and final selection after candidates already exist:

```bash
export APEX_ROOT=/home/julojays/apex
bash v3/run_score_v3_apex.sh
```

To rerun only the final APEX-aware panel selector after MIC scoring already exists:

```bash
bash v3/run_select_apex_aware_panel.sh
```

Main MIC outputs:

```text
v3/results/apex_scored_v3/apex_scored_v3_candidates.csv
v3/results/apex_scored_v3/apex_scored_v3_benchmarks.csv
v3/results/apex_scored_v3/apex_scored_v3_combined.csv
v3/results/apex_scored_v3/apex_scored_v3_vs_oracle.csv
v3/results/apex_scored_v3/apex_top_v3_candidates.fasta
v3/results/apex_scored_v3/apex_scoring_summary.json
```

Main APEX-aware final-panel outputs:

```text
v3/results/apex_scored_v3/apex_aware_ranked_v3.csv
v3/results/apex_scored_v3/apex_aware_top_panel_v3.csv
v3/results/apex_scored_v3/apex_aware_top_panel_v3.fasta
v3/results/apex_scored_v3/apex_aware_selection_summary.json
```

For ApexOracle-9 parent-variant scan:

```bash
bash v3/run_parent_scan_apex9.sh
```
