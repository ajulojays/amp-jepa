# AMP-JEPA v3 QC-core APEX progress

Date: 2026-07-14
Branch: `v3-hybrid-improved`

## Current milestone

AMP-JEPA-Hybrid v3 has moved from an APD-only prototype to a QC-filtered multi-source corpus workflow with APEX-aware multi-organism candidate selection.

The key result is that training on the QC-core upscaled corpus improved the APEX-aware candidate pool and produced a stronger global lead than the original APD-only run.

## Corpus status

Expanded public corpus build:

- Raw rows: 519,662
- Valid rows after filters: 385,570
- Unique sequences: 214,232
- Median length: 12 aa

QC-core corpus:

- Input unique sequences: 214,232
- Trainable core: 34,065 sequences
- Review/excluded: 180,167
- Main exclusion driver: suspicious/ambiguous source path filtering

The QC-core FASTA used for retraining:

```text
v3/results/upscaled_corpus_qc/upscaled_corpus_trainable_core.fasta
```

The prepared training corpus:

```text
v3/data/processed/peptide_corpus_v3_qc_core.csv
```

## Model run

Model checkpoint:

```text
v3/checkpoints/amp_jepa_hybrid_v3_qc_core.pt
```

The model remains AMP-JEPA-Hybrid v3:

- VAE-style latent sequence model
- Transformer sequence decoder
- JEPA-inspired latent consistency objective
- Not yet a full teacher-student JEPA foundation model
- No APEX supervision inside training yet

The current model learns AMP sequence grammar from the QC-core corpus, generates candidates, and then uses APEX as an external oracle for post-generation MIC-aware selection.

## APEX-aware screening progression

### APD-only baseline

- Strict pass candidates: 13
- Best candidate: `WLIKKVKNYVGNGKAKWCKI`
- APEX mean MIC: ~81.19
- APEX median MIC: ~42.61
- APEX worst MIC: ~573.18
- Organisms MIC <=64: 23

### QC-core top-500 screen

- Strict pass candidates: 66
- Best candidate: `KNKKGNIPRWLNKKVGLLGKNIK`
- APEX mean MIC: ~57.56
- APEX median MIC: ~35.67
- APEX worst MIC: ~302.80
- Organisms MIC <=64: 26

### QC-core top-1000 screen

- Strict pass candidates: 115
- Best candidate: `KRLKIINKKITIKDTKLRI`
- Best median MIC: ~30.75
- Best mean MIC: ~55.21
- Best worst MIC: ~222.73
- Max organisms MIC <=64: 26

### QC-core all-available screen

This run requested a larger top panel, but only 2,133 valid unique ranked candidates were available from that generation run. Therefore, it effectively scored all available candidates from that generation.

- Total scored: 2,133
- Strict pass candidates: 201
- Pass fraction: 9.42%
- Best mean MIC: 38.324
- Best median MIC: 23.726
- Best worst MIC: 203.709
- Max organisms MIC <=64: 30

Top lead:

```text
RLLSISSKLLSRL
```

Metrics:

- Length: 13 aa
- Net charge: +3
- Hydrophobic fraction: 0.462
- Cysteines: 0
- Tryptophans: 0
- Max train identity: 0.538
- Novelty score: 0.462
- APEX mean MIC: 38.324
- APEX median MIC: 23.726
- APEX worst MIC: 210.457
- Organisms MIC <=64: 30
- APEX-aware score: 0.754

### QC-core 50k stochastic generation, top-5000 screen

A larger generation was performed from the saved QC-core checkpoint, followed by ranking and APEX scoring of the top 5,000 candidates.

Best new 50k-generated lead:

```text
GWKSIKLGKKKLKATLLK
```

Metrics:

- Length: 18 aa
- Net charge: +7
- Hydrophobic fraction: 0.389
- Cysteines: 0
- Tryptophans: 1
- Max train identity: 0.444
- Novelty score: 0.556
- APEX mean MIC: 49.214
- APEX median MIC: 24.851
- APEX worst MIC: 259.151
- Organisms MIC <=64: 27
- APEX-aware score: 0.741

The 50k run added useful alternatives, including:

- `AKWLKSKKLILKKLKKA` with median MIC ~23.109
- `GLIAIKITRKLAKKIK` with worst MIC ~193.479
- `LIRKIIAGVKWPGKIGLLLAKAKK` with novelty score ~0.708

However, the 50k run did not replace `RLLSISSKLLSRL` as the best global APEX-aware lead.

## Global merged panel

Global merged ranking combined:

```text
v3/results/apex_scored_v3_qc_core_5000/apex_aware_ranked_v3.csv
v3/results/apex_scored_v3_qc_core_50k_top5000/apex_aware_ranked_v3.csv
```

Saved outputs:

```text
v3/results/global_qc_core_apex_panel/global_qc_core_apex_passed_ranked.csv
v3/results/global_qc_core_apex_panel/global_qc_core_apex_top50.csv
```

Global top 10:

| Global rank | Run | Sequence | Length | Charge | APEX mean MIC | APEX median MIC | APEX worst MIC | Organisms MIC <=64 | APEX-aware score |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | qc_core_2133_all_available | `RLLSISSKLLSRL` | 13 | +3 | 38.324 | 23.726 | 210.457 | 30 | 0.754 |
| 2 | qc_core_50k_top5000 | `GWKSIKLGKKKLKATLLK` | 18 | +7 | 49.214 | 24.851 | 259.151 | 27 | 0.741 |
| 3 | qc_core_50k_top5000 | `AKWLKSKKLILKKLKKA` | 17 | +8 | 56.867 | 23.109 | 374.107 | 27 | 0.737 |
| 4 | qc_core_50k_top5000 | `LIRKIIAGVKWPGKIGLLLAKAKK` | 24 | +7 | 65.344 | 29.757 | 315.038 | 27 | 0.729 |
| 5 | qc_core_2133_all_available | `GIKWIKKLLFEAKKL` | 15 | +4 | 50.498 | 25.850 | 255.978 | 28 | 0.729 |
| 6 | qc_core_50k_top5000 | `KWKKIKKGSILAKKKK` | 16 | +9 | 53.494 | 30.805 | 286.392 | 26 | 0.713 |
| 7 | qc_core_50k_top5000 | `KKKWWPIIGIIAAKIKPK` | 18 | +6 | 55.805 | 36.484 | 233.363 | 27 | 0.710 |
| 8 | qc_core_50k_top5000 | `KLLVKNYKFLIGKKHKLVLKV` | 21 | +7 | 62.111 | 32.711 | 349.764 | 26 | 0.705 |
| 9 | qc_core_50k_top5000 | `GLIAIKITRKLAKKIK` | 16 | +6 | 48.112 | 35.078 | 193.479 | 25 | 0.700 |
| 10 | qc_core_50k_top5000 | `HIRIGLGLVLILVGGKVVGGKIKLLK` | 26 | +5 | 81.727 | 34.365 | 609.331 | 28 | 0.700 |

## Nearest-neighbor audit

The top candidate `RLLSISSKLLSRL` is not an exact training copy.

Nearest training sequence by difflib audit:

```text
RLLSLIRKLLT
```

Similarity:

- Reported max train identity: 0.538
- difflib nearest similarity: 0.667

Interpretation:

- The lead is AMP-motif-like, not a direct memorized sequence.
- It should be described as motif-novel or sequence-divergent relative to nearest training peptide, not fully de novo.

More novelty-forward alternatives include:

- `KLIWHSIGKLLRAVGKILNNGTQ`
- `LIRKIIAGVKWPGKIGLLLAKAKK`
- `LIIKIGIVRKYKQKIGGGIKS`
- `KKNWIRVKTFSGKIISILILPILTKNPNL`

## Current architecture

Current implemented architecture:

```text
QC-core AMP corpus
      ↓
00_prepare_corpus.py
      ↓
AMP-JEPA-Hybrid v3 training
  - sequence reconstruction
  - latent regularization
  - JEPA-inspired latent consistency
      ↓
02_generate_candidates.py
      ↓
03_rank_candidates.py
  - novelty
  - charge
  - hydrophobicity
  - length
  - internal v3 score
      ↓
25_score_v3_candidates_with_apex.py
  - APEX 40-model MIC ensemble
  - organism-specific MIC predictions
      ↓
26_select_apex_aware_panel.py
  - mean MIC
  - median MIC
  - worst MIC
  - organisms MIC <=64
  - developability filters
  - novelty/internal score
      ↓
Global APEX-aware final panel
```

What the architecture is currently good at:

- Learning AMP-like sequence grammar from a larger QC-filtered corpus
- Generating novel or motif-novel peptides
- Filtering candidates by simple AMP-like properties
- Externally scoring with a multi-organism APEX MIC ensemble
- Selecting balanced potency/breadth/robustness candidates

What it does not yet do:

- It does not train directly on MIC labels
- It does not use APEX labels during generation
- It does not perform organism-conditioned generation
- It does not yet include a true teacher-student JEPA objective
- It does not yet include family-held-out evaluation
- It does not yet include experimental validation

## Recommended next architecture step

The next major architecture should be AMP-JEPA v4 / APEX-distilled activity-aware AMP-JEPA.

Proposed direction:

```text
QC-core corpus
      ↓
AMP-JEPA encoder / latent space
      ├── sequence reconstruction
      ├── JEPA latent consistency
      ├── APEX median MIC head
      ├── APEX mean MIC head
      ├── APEX worst MIC head
      ├── organisms <=64 breadth head
      ├── Gram-negative activity head
      └── Gram-positive activity head
```

Important framing:

- APEX MIC is not experimental truth.
- APEX should be treated as a teacher/oracle for weak-supervised distillation.
- The current v3 result should be preserved as the pre-distillation baseline.

## Current conclusion

AMP-JEPA-Hybrid v3 has a strong computational proof-of-progress:

> Training on a QC-filtered 34,065-sequence multi-source AMP corpus improved APEX-aware candidate discovery relative to APD-only training, producing a larger pass-candidate pool and a stronger global lead, `RLLSISSKLLSRL`, with predicted median MIC ~23.7, mean MIC ~38.3, worst MIC ~210.5, and predicted activity against 30 organisms at MIC <=64.

This is still computational and APEX-predicted. It should not be represented as experimentally validated antimicrobial activity.
