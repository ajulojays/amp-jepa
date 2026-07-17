#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

DEVICE="${DEVICE:-auto}"
N_PARENTS="${N_PARENTS:-512}"
N_OFFSPRING="${N_OFFSPRING:-10000}"
BATCH_SIZE="${BATCH_SIZE:-512}"
PROPOSAL_MULTIPLIER="${PROPOSAL_MULTIPLIER:-4.0}"
LATENT_SIGMA="${LATENT_SIGMA:-0.35}"
TEMPERATURE="${TEMPERATURE:-0.85}"
CROSSOVER_RATE="${CROSSOVER_RATE:-0.20}"
SEED="${SEED:-20260716}"

OUTDIR="v4b/results/generation_01"
PARENTS="$OUTDIR/generation_01_parents.csv"
CANDIDATES="$OUTDIR/generation_01_candidates_pre_apex.csv"

echo "[V4B] Generation 1"
echo "[V4B] Parents:          $N_PARENTS"
echo "[V4B] Offspring target: $N_OFFSPRING"
echo "[V4B] Device:           $DEVICE"
echo "[V4B] Output:           $OUTDIR"

if [[ -f "$PARENTS" && -f "$OUTDIR/parent_selection_summary.json" ]]; then
  echo "[V4B] Parent selection already complete; resuming."
else
  echo "[V4B] Selecting fitness-diverse parents"
  python v4b/02_select_generation_01_parents.py \
    --n-parents "$N_PARENTS" \
    --seed "$SEED"
fi

if [[ -f "$CANDIDATES" && -f "$OUTDIR/generation_01_latent_proposals.npz" && -f "$OUTDIR/generation_01_generation_summary.json" ]]; then
  echo "[V4B] Generation 1 offspring already exist; nothing to do."
else
  echo "[V4B] Generating latent offspring"
  python v4b/03_generate_generation_01.py \
    --n-offspring "$N_OFFSPRING" \
    --proposal-multiplier "$PROPOSAL_MULTIPLIER" \
    --batch-size "$BATCH_SIZE" \
    --latent-sigma "$LATENT_SIGMA" \
    --temperature "$TEMPERATURE" \
    --crossover-rate "$CROSSOVER_RATE" \
    --seed "$SEED" \
    --device "$DEVICE"
fi

echo "[V4B] Generation 1 pre-APEX stage complete"
echo "[V4B] Candidate table: $CANDIDATES"
