# DBAASP manual download batch 1

This folder tracks the first manually downloaded DBAASP batch for AMP-JEPA.

Batch label: `dbaasp1_batch1`

## Status

The raw DBAASP web/API routes tested from `/api/v1` returned HTTP 200 with empty bodies, so batch 1 was downloaded manually and harmonized offline.

The full harmonized bundle was created in the ChatGPT workspace as:

- `dbaasp_harmonized_bundle.zip`

Because the GitHub connector used here can write UTF-8 text files but cannot directly attach local binary zip files by path, this repo folder records the batch manifest, checksums, harmonization summary, and the reusable harmonization script. The large harmonized CSV/FASTA bundle should be added from the local machine with normal `git add` once downloaded.

## Batch 1 uploaded source files

Manual DBAASP files used:

- `peptides.csv`
- `activity-against-target-species.csv`
- `hemolytic-and-cytotoxic-activities.csv`
- `peptides-antibiofilm-activities.csv`
- `peptides-fasta.txt`

## Harmonized output files generated

- `dbaasp_peptides_master_harmonized.csv`
- `dbaasp_activity_target_species_long.csv`
- `dbaasp_mic_labels_long.csv`
- `dbaasp_hemolysis_cytotoxicity_long.csv`
- `dbaasp_antibiofilm_long.csv`
- `dbaasp_trainable_single_chain_sequences.csv`
- `dbaasp_trainable_single_chain_sequences.fasta`
- `dbaasp_fasta_id_sequence_map.csv`
- `dbaasp_harmonization_summary.json`
- `README_DBAASP_HARMONIZED.md`

## Summary counts

```text
Peptide metadata rows:                 2,000
FASTA records parsed:                  1,988
Target-species activity rows:         18,304
Hemolysis/cytotoxicity rows:           1,812
Antibiofilm rows:                         80

Master harmonized peptides:            2,000
Peptides with FASTA record:            1,988
Peptides missing FASTA record:            12
Clean single-chain trainable peptides: 1,922
MIC label rows with sequence:         13,788
```

## Important harmonization choices

- MIC units are kept separate. No conversion was performed between `uM`, `ug_per_ml`, and `nmol_per_g`.
- Multimer/multi-peptide records are preserved in the peptide master table but excluded from the single-chain training FASTA/CSV.
- DBAASP identifiers are harmonized as numeric `dbaasp_id`.
- FASTA classes such as `DBAASPR` and `DBAASPS` are preserved where available.
- Long-format activity tables are intended for downstream activity-aware AMP-JEPA v4 modeling.

## Intended downstream use

For AMP-JEPA v3/v4, the most useful harmonized files are:

```text
dbaasp_trainable_single_chain_sequences.fasta
dbaasp_trainable_single_chain_sequences.csv
dbaasp_mic_labels_long.csv
dbaasp_hemolysis_cytotoxicity_long.csv
dbaasp_antibiofilm_long.csv
```

This batch is the first step toward replacing a purely post-generation APEX filter with real-label activity and toxicity heads in an activity-aware AMP-JEPA model.

## Suggested local add after downloading bundle

```bash
mkdir -p v3/data/raw/corpus_sources/dbaasp1_batch1
unzip dbaasp_harmonized_bundle.zip -d /tmp/dbaasp_batch1
cp -v /tmp/dbaasp_batch1/dbaasp_harmonized/* v3/data/raw/corpus_sources/dbaasp1_batch1/

git add v3/data/raw/corpus_sources/dbaasp1_batch1
git commit -m "Add DBAASP manual batch 1 harmonized data"
git push origin v3-hybrid-improved
```
