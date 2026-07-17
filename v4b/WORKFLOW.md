# AMP-JEPA V4B workflow

## Scope

V4B is a ten-generation closed-loop extension of V4A. It re-encodes successful descendants, updates the learned peptide manifold, generates again, and treats hemolysis and cytotoxicity as first-class optimization objectives.

V4A remains frozen as Generation 0.

```text
V4A Generation 0
      ↓
Re-encode selected candidates into latent μ
      ↓
Generate latent-neighborhood offspring
      ↓
Sequence optimization and G-Rescue
      ↓
APEX activity scoring
      ↓
Hemolysis and cytotoxicity scoring
      ↓
Elite, Pareto, spectrum and specialist classification
      ↓
Select diverse parents
      ↓
Update/retrain latent manifold
      ↓
Generation n + 1
```

## Mandatory Phase 0: safety-manifold pilot

The full ten-generation loop must not start until the frozen V3/V4A latent representation is tested for safety-related structure.

Phase 0 asks:

1. Does latent μ separate hemolytic from non-hemolytic peptides?
2. Does latent μ separate cytotoxic from non-cytotoxic peptides?
3. Does latent μ outperform or complement simple physicochemical features?
4. Are the results stable under stratified cross-validation?
5. Are there enough labeled positive and negative examples to justify safety heads?

```text
Safety-labeled reference peptides
      ↓
Frozen V3 encoder
      ↓
Latent μ vectors
      ↓
PCA visualization + logistic-regression cross-validation
      ↓
Compare latent-only, physicochemical-only and combined models
      ↓
GO / REVISE / NO-GO decision
```

Missing labels are never interpreted as safe.

## Phase 0 inputs

A CSV containing at minimum:

```text
sequence
hemolysis_label
cytotoxicity_label
```

Labels must be binary when present:

```text
0 = negative / low risk under the source definition
1 = positive / high risk under the source definition
blank = unknown and excluded from that task
```

Recommended provenance columns:

```text
source
assay_type
assay_endpoint
concentration
unit
cell_type
species
reference
```

## Phase 0 outputs

```text
v4b/results/safety_manifold_pilot/
├── safety_reference_clean.csv
├── latent_mu.npy
├── latent_logvar.npy
├── latent_metadata.csv
├── pca_coordinates.csv
├── safety_manifold_metrics.csv
├── safety_manifold_summary.json
├── hemolysis_pca.png
└── cytotoxicity_pca.png
```

## Phase 0 decision rule

The pilot does not claim a safety model is validated. It determines whether latent safety heads are technically justified.

Suggested initial GO rule for each endpoint:

```text
minimum labeled peptides: 200
minimum positives: 40
minimum negatives: 40
5-fold stratified CV AUROC ≥ 0.70 for latent or combined features
no fold with only one class
```

A result below this threshold means the endpoint needs more labels, better harmonization, a different representation, or a dedicated external safety model.

## Ten-generation protocol

After Phase 0 passes:

```text
Generation 0  = frozen V4A
Generation 1  = first safety-aware descendant population
Generation 2
Generation 3
Generation 4
Generation 5
Generation 6
Generation 7
Generation 8
Generation 9
Generation 10 = final evolved population
```

Every generation stores candidates, latent vectors, activity predictions, safety predictions, lineage, parent selection, model checkpoint and metrics. No generation is overwritten.

## Candidate labels retained in every generation

```text
is_optimization_success
is_g_rescue
is_elite
is_pareto
is_elite_pareto
is_potent_any_organism
is_narrow_spectrum_specialist
is_broad_spectrum
is_safety_qualified
is_lead
```

`is_safety_qualified` can only be true when explicit hemolysis and cytotoxicity predictions are available and pass the configured thresholds.

## Scientific boundary

All V4B safety outputs remain computational predictions until experimentally tested. Use:

```text
predicted low hemolysis risk
predicted low cytotoxicity risk
safety-qualified in silico candidate
```

Do not use:

```text
non-toxic
non-hemolytic
safe peptide
```
