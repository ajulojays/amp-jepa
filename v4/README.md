# AMP-JEPA-Hybrid V4A

**AMP-JEPA-Hybrid V4A** is the APEX-only, full-scale optimization extension of the AMP-JEPA-Hybrid v3 latent peptide generator.

V4A does not replace the v3 backbone. It turns v3 into a full design system.

```text
V3:
  generate AMP-like candidates -> APEX screen -> select top hits

V4A:
  generate candidates -> score all candidates -> map the landscape -> diagnose failures -> optimize/rescue -> rescore -> stress-test -> select Pareto lead panel
```

## Core definition

> AMP-JEPA-Hybrid V4A extends the v3 latent AMP generator into a population-wide, failure-aware, optimization-guided peptide design system using APEX-predicted multi-organism activity and multiobjective Pareto selection.

V4A is intentionally **APEX-only for now**. DBAASP is preserved as a future real-label layer, but it is not required for this version.

## Why V4A exists

V3 already showed that the QC-core corpus improved generated candidate quality. However, v3 mostly performs broad latent sampling followed by post-generation screening.

V4A asks a stronger biological question:

```text
Can the entire generated peptide landscape be understood, repaired, optimized, and stress-tested?
```

This makes V4A different from a simple top-hit selector. V4A treats failed candidates as biological information. Weak candidates may contain recoverable motifs, immature scaffolds, or correctable physicochemical imbalance.

## Design philosophy

V4A follows the principle:

```text
Out of ashes comes beauty.
```

A low-scoring candidate is not automatically discarded. It is diagnosed first.

A candidate may fail because it is:

- too hydrophobic
- too weakly cationic
- too strongly cationic
- too short or too long
- too similar to training sequences
- too narrow in predicted organism coverage
- poor only in worst-case predicted MIC
- chemically/developability imbalanced
- locally fragile but biologically promising

Only true degenerate junk is discarded immediately.

## V4A system architecture

```text
QC-core trained AMP-JEPA-Hybrid v3
          |
          v
Large seed candidate generation
          |
          v
APEX full-panel scoring
          |
          v
Whole-population landscape mapping
          |
          +----------------------+----------------------+
          |                      |                      |
          v                      v                      v
Elite refinement          Near-pass rescue          G-Rescue
          |                      |                      |
          +----------------------+----------------------+
                                 |
                                 v
                    Optimized variant pool
                                 |
                                 v
                    APEX rescoring
                                 |
                                 v
                    Local robustness testing
                                 |
                                 v
                    Pareto final lead panel
```

## Candidate classes

V4A classifies generated candidates into design roles:

| Class | Meaning | Action |
|---|---|---|
| A | Elite candidates | refine and preserve |
| B | Near-pass candidates | rescue failed criterion |
| C | High-novelty candidates | improve activity while preserving novelty |
| D | Broad-spectrum candidates | improve potency/developability |
| E | Worst-case robust candidates | improve median/mean while preserving robustness |
| F | Developable candidates | improve activity while preserving clean properties |
| G | Rescue candidates | diagnose and repair hidden biological structure |
| G5 | True junk | discard |

## Optimization objectives

V4A optimizes multiple objectives at once:

- minimize APEX-predicted median MIC
- minimize APEX-predicted mean MIC
- minimize APEX-predicted worst MIC
- maximize number of organisms with predicted MIC <= 64
- maximize novelty against the training corpus
- maximize sequence diversity
- maximize local robustness
- keep length biologically reasonable
- keep charge biologically reasonable
- keep hydrophobicity biologically reasonable
- avoid excessive cysteine burden
- avoid excessive tryptophan/aromatic burden
- avoid simple memorization of known training peptides

V4A is not a lowest-MIC-at-all-cost system. It is a robust biological lead-design system.

## Planned output directories

```text
v4/results/
├── seed_pool/
│   ├── v4a_seed_candidates.csv
│   └── v4a_seed_apex_scored.csv
├── landscape/
│   ├── candidate_landscape.csv
│   ├── candidate_clusters.csv
│   └── candidate_class_assignments.csv
├── rescue/
│   ├── g_rescue_candidates.csv
│   ├── g_rescue_variants.csv
│   └── g_rescue_successes.csv
├── optimization/
│   ├── optimized_variants.csv
│   └── optimized_variants_apex_scored.csv
├── robustness/
│   └── robustness_scores.csv
└── final_panel/
    ├── v4a_pareto_front.csv
    ├── v4a_top20_panel.csv
    ├── v4a_top50_panel.csv
    └── v4a_top50_panel.fasta
```

## Version ladder

```text
V3:
  latent AMP generator + APEX post-screening

V4A:
  V3 backbone + APEX-guided full-population optimization + G-Rescue

V5:
  future activity-aware training with DBAASP/real-label integration
```

## Current status

V4A is the next planned architecture layer. This folder documents the system design and expected implementation layout.

The immediate implementation target is:

```text
APEX-only, no DBAASP,
whole-pool optimization,
failure-aware G-Rescue,
robustness scoring,
Pareto final panel.
```
