# G-Rescue: Failure-Aware Peptide Rescue for AMP-JEPA-Hybrid V4A

## Core idea

G-Rescue is the part of AMP-JEPA-Hybrid V4A that works on weak or failed candidates instead of discarding them immediately.

The principle is:

```text
Out of ashes comes beauty.
```

In computational screening, low score often means discard. In biological design, low score may mean the scaffold is immature, imbalanced, incomplete, or fragile but still recoverable.

G-Rescue asks:

```text
Why did this candidate fail?
Can the failure be repaired?
Can a weak region become a stronger biological design?
```

## Why G-Rescue matters

If V4A only optimizes elite candidates, the search becomes greedy. It may collapse into one motif family and miss hidden biological regions.

G-Rescue expands the design search by treating failure as information.

A failed candidate may still contain:

- a useful amphipathic motif
- a promising cationic pattern
- a rare high-novelty scaffold
- a good local sequence family
- a near-pass profile with one correctable defect
- a weak but broad predicted organism profile
- a poor median but good worst-case stability

## G classes

### G1: Near-biological failures

Candidates that look AMP-like but score poorly.

Possible issue:

- incorrect charge/hydrophobic balance
- local motif imbalance
- one bad terminal region

Action:

- charge tuning
- hydrophobicity correction
- terminal trim/extension

### G2: Property-failed candidates

Candidates that fail mostly because their physicochemical properties are outside acceptable bounds.

Possible issue:

- too hydrophobic
- too many aromatic residues
- too many cysteines
- too low charge
- too high charge
- too long or too short

Action:

- targeted residue replacement
- terminal correction
- burden reduction

### G3: Novel-but-weak candidates

Candidates with high novelty but weak APEX score.

Action:

- preserve novelty-critical positions
- modify non-core positions
- improve charge/hydrophobicity
- rescore for activity gain

### G4: Fragile candidates

Candidates that score decently but fail under small perturbations or have poor worst-case organism profile.

Action:

- local robustness edits
- worst-case-guided mutation
- remove fragile residue burden

### G5: True junk

Candidates with little recoverable structure.

Examples:

- extreme low-complexity repeats
- impossible physicochemical profile
- degenerate short fragments with no useful signal
- extreme aromatic/cysteine burden without useful structure

Action:

- discard

## Failure-mode diagnosis

Each candidate receives one or more failure labels:

```text
fail_length_low
fail_length_high
fail_charge_low
fail_charge_high
fail_hydrophobicity_low
fail_hydrophobicity_high
fail_cysteine_burden
fail_tryptophan_burden
fail_aromatic_burden
fail_low_novelty
fail_low_breadth
fail_high_median_mic
fail_high_mean_mic
fail_high_worst_mic
fail_local_fragility
fail_low_complexity
```

## Rescue operators

### Charge rescue

Used when candidate charge is too low or cationic density is poorly placed.

Operations:

- replace selected neutral residues with K/R
- add terminal K/R extension
- preserve hydrophobic face when possible

### Hydrophobicity rescue

Used when candidate is too hydrophobic or too weakly hydrophobic.

Operations:

- replace selected L/I/V/F/W with A/S/N/Q/K
- replace selected polar residues with A/L/I/V if hydrophobicity is too low
- avoid making peptide membrane-toxic by excessive hydrophobic increase

### Aromatic burden rescue

Used when W/F/Y burden is high.

Operations:

- W to F/Y/L/A/K depending on context
- F to L/A/Y
- reduce clustered aromatics

### Cysteine rescue

Used when cysteine burden is high or disulfide-like patterns are not desired for the current design mode.

Operations:

- C to S/A depending on context
- preserve cysteine only if candidate belongs to a defensin-like family mode

### Length rescue

Used when candidate is too short or too long.

Operations:

- terminal trimming
- terminal extension with constrained cationic/amphipathic residues
- preserve central motif

### Worst-case rescue

Used when predicted median is good but one organism has poor predicted MIC.

Operations:

- generate variants and optimize toward lower APEX_worst_MIC
- preserve breadth and median MIC

### Novelty rescue

Used when candidate is too similar to training sequences.

Operations:

- mutate non-core positions
- preserve global physicochemical profile
- increase sequence distance while retaining predicted activity

## G-Rescue success criteria

A rescued candidate is successful if it improves at least one important objective without catastrophic loss elsewhere.

Examples:

- lower APEX_median_MIC with acceptable breadth
- lower APEX_worst_MIC with similar median
- improved organisms_MIC_le_64
- improved novelty with similar predicted activity
- improved developability with acceptable predicted activity
- improved local robustness

## Output files

```text
v4/results/rescue/g_rescue_candidates.csv
v4/results/rescue/g_rescue_failure_modes.csv
v4/results/rescue/g_rescue_variants.csv
v4/results/rescue/g_rescue_successes.csv
```

## Scientific framing

G-Rescue makes V4A biologically richer than a simple optimizer.

It says:

> Weak candidates are not only failures. They are hypotheses about peptide design space. Some will be junk, but some may be immature scaffolds that can be repaired into stronger AMP-like designs.
