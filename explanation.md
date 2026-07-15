# AMP-JEPA-Hybrid v3: Current Architecture and Training Explanation

This document summarizes the current AMP-JEPA-Hybrid v3 architecture, training objective, corpus state, and the current best computational result.

## One-line summary

**AMP-JEPA-Hybrid v3 is a compact generative antimicrobial-peptide model trained on a QC-filtered multi-source AMP corpus, using a VAE-style latent space, a JEPA-inspired masked-view latent prediction objective, and physicochemical-property regularization. APEX is currently used after generation as an external multi-organism MIC teacher/evaluator, not yet as an in-training supervision signal.**

## Current pipeline

```text
QC-core AMP corpus
      ↓
AMP-JEPA-Hybrid v3 training
      ↓
Candidate generation from latent z
      ↓
Internal AMP-like ranking
      ↓
APEX multi-organism MIC scoring
      ↓
APEX-aware final selection
      ↓
Global computational lead panel
```

## Corpus state

The current strongest run used the QC-filtered multi-source corpus:

```text
Merged raw/upscaled corpus:        214,232 unique sequences
QC-core trainable corpus:           34,065 sequences
Original APD-only corpus:           ~2,962 sequences
```

The QC-core corpus removes or flags likely noisy archive-derived inputs such as non-AMP controls, feature matrices, requirements/docs, validation-result tables, and automatically inferred columns that may not be true peptide sequences.

The current training corpus used for the strongest v3 run was:

```text
v3/results/upscaled_corpus_qc/upscaled_corpus_trainable_core.fasta
```

and was prepared into:

```text
v3/data/processed/peptide_corpus_v3_qc_core.csv
```

## What goes into training

Each peptide contributes three training views:

```text
1. Full sequence tokens
2. Masked sequence tokens
3. Simple physicochemical features
```

The physicochemical feature vector contains:

```text
normalized length
approximate net charge
hydrophobic fraction
glycine/proline fraction
aromatic fraction
```

These features are not MIC labels. They are weak biological regularizers that encourage the latent space to encode AMP-relevant structure.

## Model components

The core model is `HybridVAEJEPA`.

It contains:

```text
token embedding
position embedding
Transformer encoder
mean pooling over non-padding residues
latent mean μ
latent log variance
sampled latent vector z
linear sequence decoder
masked-view latent predictor
physicochemical property prediction head
```

The model is intentionally compact:

```text
max_len:       64
latent_dim:    64
d_model:       192
n_layers:      4
n_heads:       6
ff_dim:        512
dropout:       0.1
```

## Why it is called “Hybrid”

AMP-JEPA-Hybrid v3 combines four ideas:

```text
VAE:
  learns a smooth, sampleable latent peptide space

Transformer encoder:
  learns peptide sequence context

JEPA-inspired latent consistency:
  predicts the full-peptide latent representation from a masked peptide view

Physicochemical property head:
  forces the latent space to encode AMP-like biological properties
```

So v3 is not a plain VAE and not a pure JEPA. It is a **VAE–JEPA-inspired hybrid generator**.

## Training objective

The total loss is:

```text
total_loss =
    reconstruction_loss
  + beta_kl * KL_loss
  + jepa_weight * masked_view_latent_prediction_loss
  + property_weight * property_prediction_loss
```

Current default weights:

```text
beta_kl:         0.02
jepa_weight:     0.25
property_weight: 0.10
```

### 1. Reconstruction loss

This teaches the model to reconstruct the peptide sequence from its latent vector.

In plain language:

```text
Can the model compress the peptide into a latent vector and decode it back into a plausible peptide?
```

This is the main generative learning signal.

### 2. KL loss

This regularizes the latent space so that it remains smooth and sampleable.

In plain language:

```text
Do not memorize every peptide as an isolated point.
Keep the latent space structured enough that random z samples can decode into plausible peptides.
```

### 3. JEPA-inspired latent consistency loss

The model receives a masked peptide view and encodes it into a masked latent representation. A predictor then tries to recover the full unmasked peptide latent representation.

```text
masked peptide
      ↓
masked latent μ
      ↓
view predictor
      ↓
predicted full latent μ
      ↓
MSE against full unmasked latent μ
```

The target full latent representation is detached from the gradient flow.

In plain language:

```text
Can the model infer the complete peptide-level representation from incomplete sequence evidence?
```

This is the JEPA-inspired part because the model predicts in representation space rather than only reconstructing raw residues.

### 4. Property prediction loss

The latent representation is also trained to predict the peptide's simple physicochemical features.

In plain language:

```text
Does the latent space know peptide length, charge, hydrophobicity, gly/pro content, and aromaticity?
```

This provides a weak biological structure signal.

## What v3 learns right now

AMP-JEPA-Hybrid v3 currently learns:

```text
AMP sequence grammar
latent peptide organization
sequence reconstruction
masked-view latent consistency
basic physicochemical structure
```

It does **not yet** learn experimental MIC values or APEX-predicted MIC values during training.

## What APEX currently does

APEX is currently external to the training loop.

Current relationship:

```text
AMP-JEPA generates candidates.
APEX scores candidates after generation.
The APEX-aware selector chooses the final panel.
```

APEX currently provides:

```text
mean predicted MIC
median predicted MIC
best predicted MIC
worst predicted MIC
organisms with predicted MIC ≤32
organisms with predicted MIC ≤64
Gram-negative summaries
Gram-positive summaries
```

Important caveat:

```text
APEX MIC values are computational predictions, not experimental ground truth.
```

## Current best computational result

The strongest current result is from the merged global QC-core APEX panel:

```text
v3/results/global_qc_core_apex_panel/global_qc_core_apex_passed_ranked.csv
v3/results/global_qc_core_apex_panel/global_qc_core_apex_top50.csv
```

The global #1 candidate is:

```text
RLLSISSKLLSRL
```

Metrics:

```text
Length:              13 aa
Charge:              +3
Hydrophobicity:      0.462
Cysteines:           0
Tryptophans:         0
Max train identity:  0.538
Novelty score:       0.462
APEX mean MIC:       38.324
APEX median MIC:     23.726
APEX worst MIC:      210.457
Organisms MIC ≤64:   30
APEX-aware score:    0.754
```

The top global candidates include:

```text
1. RLLSISSKLLSRL
2. GWKSIKLGKKKLKATLLK
3. AKWLKSKKLILKKLKKA
4. LIRKIIAGVKWPGKIGLLLAKAKK
5. GIKWIKKLLFEAKKL
6. KWKKIKKGSILAKKKK
7. KKKWWPIIGIIAAKIKPK
8. KLLVKNYKFLIGKKHKLVLKV
9. GLIAIKITRKLAKKIK
10. HIRIGLGLVLILVGGKVVGGKIKLLK
```

## Evidence that QC-core helped

The APD-only model produced fewer strict APEX-aware candidates and a weaker top lead.

Summary trend:

```text
APD-only top-500:
  strict-pass candidates: 13
  best median MIC:        ~42.61
  best worst MIC:         ~573.18
  max organisms ≤64:      23

QC-core top-500:
  strict-pass candidates: 66
  best median MIC:        ~35.67
  best worst MIC:         ~297.15
  max organisms ≤64:      26

QC-core all-available screen:
  scored candidates:      2,133
  strict-pass candidates: 201
  best median MIC:        23.73
  best worst MIC:         203.71
  max organisms ≤64:      30
```

This supports the current working conclusion:

```text
Training AMP-JEPA-Hybrid v3 on a QC-filtered multi-source AMP corpus improved candidate yield and produced stronger APEX-aware leads compared with APD-only training.
```

## Important limitation: this is not full JEPA yet

The current v3 model is **JEPA-inspired**, not a full teacher-student JEPA model.

Current v3:

```text
same encoder processes full and masked peptide views
masked-view latent predictor learns to match full-view latent μ
VAE decoder generates sequences from latent z
```

A fuller JEPA-style AMP foundation model would likely include:

```text
student encoder
teacher encoder
EMA teacher updates
context/target masking
latent target prediction without sequence reconstruction as the main objective
family-held-out representation evaluation
possibly ESM/protein-LM latent targets
```

So the honest terminology is:

```text
AMP-JEPA-Hybrid v3 uses JEPA-inspired latent consistency.
It is not yet a full JEPA foundation model.
```

## Next architecture direction

The natural next step is **APEX-distilled activity-aware AMP-JEPA**.

In that version, APEX would move from being only an external post-generation evaluator to also being an in-training pseudo-label teacher.

Future architecture:

```text
AMP-JEPA latent z
      ├── sequence reconstruction
      ├── JEPA latent consistency
      ├── physicochemical property prediction
      ├── APEX median MIC prediction
      ├── APEX mean MIC prediction
      ├── APEX worst MIC prediction
      ├── organisms MIC ≤64 prediction
      ├── Gram-negative activity prediction
      └── Gram-positive activity prediction
```

This would change the model from:

```text
v3: generate first, score later
```

to:

```text
v4: learn an activity-aware latent space, then generate
```

## Current scientific framing

Best current wording:

```text
AMP-JEPA-Hybrid v3 is a VAE–JEPA-inspired antimicrobial peptide generator trained on a QC-filtered multi-source AMP corpus. The model learns a smooth peptide latent space using sequence reconstruction, masked-view latent prediction, and physicochemical property regularization. Generated candidates are subsequently scored with an external APEX multi-organism MIC ensemble and selected using a multiobjective APEX-aware ranking function.
```

Avoid claiming:

```text
experimental MIC validation
true wet-lab activity
full JEPA foundation model
superiority over APEX/APEXGO/AMPainter
```

until those comparisons or experiments are performed.
