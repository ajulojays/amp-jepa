# AMP-JEPA-Hybrid V4A Architecture Specification

## Name

**AMP-JEPA-Hybrid V4A: APEX-Guided Full-Population Optimization with Failure-Aware G-Rescue**

## One-line definition

AMP-JEPA-Hybrid V4A uses the trained V3 latent peptide generator as a backbone and adds a full-population optimization system that scores, maps, rescues, improves, and stress-tests generated AMP candidates using APEX-guided multiobjective selection.

## What changes from V3

V3 is a generative discovery engine:

```text
latent sampling -> peptide decoding -> filtering -> APEX scoring -> top panel
```

V4A is an optimization-guided design engine:

```text
candidate population -> APEX scoring -> landscape diagnosis -> candidate-class assignment -> optimization/rescue -> rescoring -> robustness testing -> Pareto panel
```

The core neural backbone can remain V3 initially. The architecture jump comes from the optimization layer around it.

## Major modules

### 1. Seed generation module

Inputs:

- QC-core trained V3 checkpoint
- V3 vocabulary/tokenizer
- generation count
- temperature settings
- candidate validity filters

Outputs:

- raw generated peptides
- valid unique canonical candidates
- initial physicochemical features

Purpose:

Generate a large, diverse starting population rather than a small top-hit pool.

### 2. APEX scoring module

Inputs:

- seed candidates
- APEX model directory

Outputs:

- APEX-predicted mean MIC
- APEX-predicted median MIC
- APEX-predicted worst MIC
- number of organisms predicted MIC <= 32/64/80/128
- Gram-negative and Gram-positive summaries if available

Purpose:

Convert the generated peptide population into a multi-organism predicted activity landscape.

### 3. Landscape mapping module

Inputs:

- APEX-scored candidate table
- physicochemical features
- novelty scores

Outputs:

- candidate landscape table
- score distributions
- class assignments
- cluster annotations

Purpose:

Understand the whole candidate population rather than only selecting the highest-ranked hits.

### 4. Candidate class assignment module

V4A assigns candidates into biological design roles:

```text
A  Elite
B  Near-pass
C  High-novelty
D  Broad-spectrum
E  Worst-case robust
F  Developable
G  Rescue candidate
G5 True junk
```

The class assignment controls how each candidate is optimized.

### 5. Optimization module

Optimization operates on both strong candidates and recoverable weak candidates.

Sequence-level operations:

- single substitution
- double substitution
- terminal trimming
- terminal extension
- K/R charge tuning
- D/E acidic reduction when appropriate
- I/L/V hydrophobic tuning
- W/F/Y aromatic burden tuning
- cysteine control
- motif-preserving mutation
- local shuffle with constraints

Latent-level operations, later stage:

- encode strong or rescue candidates
- sample around latent mu neighborhoods
- decode nearby z values
- score decoded variants

### 6. G-Rescue module

G-Rescue is the signature V4A biological feature.

It treats low-scoring candidates as possible immature biological scaffolds rather than automatic waste.

Failure modes:

- hydrophobicity failure
- charge failure
- length failure
- worst-case MIC failure
- breadth failure
- novelty failure
- W/F/Y burden failure
- cysteine burden failure
- local fragility
- low-complexity degeneration

Each failure mode maps to a rescue strategy.

### 7. Robustness module

A final lead should not only be one good exact sequence.

Robustness is evaluated by generating local variants and asking whether the design neighborhood remains favorable.

Robustness checks:

- K/R swaps
- I/L/V swaps
- one-residue substitutions
- terminal trimming
- terminal extension
- low-burden aromatic substitutions
- local physicochemical perturbations

Output:

```text
robustness_score = fraction of local variants that remain acceptable under APEX-aware filters
```

### 8. Pareto selection module

V4A should not force all candidates into one artificial scalar score.

The final panel should preserve different kinds of biological value:

- best overall
- best predicted median MIC
- best predicted worst-case MIC
- best predicted organism breadth
- best novelty
- best short/simple peptide
- best low-W candidate
- best low-C candidate
- best G-Rescue success
- best balanced developability candidate

## V4A scoring philosophy

V4A may compute a scalar score for convenience, but final selection should be Pareto-aware.

Candidate value includes:

```text
potency
breadth
worst-case robustness
novelty
developability
diversity
local robustness
rescue potential
```

## Non-goals for V4A

V4A does not claim experimental MIC validation.

V4A does not use DBAASP for training yet.

V4A does not claim that APEX predictions are true biological MIC values.

V4A is an APEX-guided computational design and optimization system.

## Scientific claim

The central scientific claim of V4A is:

> Generated peptide landscapes contain recoverable biological structure beyond the top-ranked candidates. A failure-aware optimization engine can transform weak or near-pass candidates into stronger, more robust AMP-like designs.
