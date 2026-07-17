#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

START_GENERATION="${START_GENERATION:-1}"
END_GENERATION="${END_GENERATION:-10}"
DEVICE="${DEVICE:-auto}"
N_PARENTS="${N_PARENTS:-512}"
N_OFFSPRING="${N_OFFSPRING:-10000}"
N_SURVIVORS="${N_SURVIVORS:-2500}"
BATCH_SIZE="${BATCH_SIZE:-512}"
ENCODE_BATCH_SIZE="${ENCODE_BATCH_SIZE:-256}"
PROPOSAL_MULTIPLIER="${PROPOSAL_MULTIPLIER:-4.0}"
CROSSOVER_RATE="${CROSSOVER_RATE:-0.20}"
LATENT_SIGMA_START="${LATENT_SIGMA_START:-0.35}"
LATENT_SIGMA_END="${LATENT_SIGMA_END:-0.15}"
TEMPERATURE_START="${TEMPERATURE_START:-0.85}"
TEMPERATURE_END="${TEMPERATURE_END:-0.65}"
BASE_SEED="${BASE_SEED:-20260716}"
CHECKPOINT="${CHECKPOINT:-v3/checkpoints/amp_jepa_hybrid_v3_qc_core.pt}"
RESULTS_ROOT="${RESULTS_ROOT:-v4b/results}"
APEX_SCORER_CMD="${APEX_SCORER_CMD:-}"

if (( START_GENERATION < 1 )); then
  echo "[V4B] START_GENERATION must be at least 1." >&2
  exit 2
fi
if (( END_GENERATION < START_GENERATION )); then
  echo "[V4B] END_GENERATION must be >= START_GENERATION." >&2
  exit 2
fi
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "[V4B] Checkpoint not found: $CHECKPOINT" >&2
  exit 2
fi

schedule_value() {
  local start_value="$1"
  local end_value="$2"
  local generation="$3"
  local final_generation="$4"
  awk -v a="$start_value" -v b="$end_value" -v g="$generation" -v n="$final_generation" \
    'BEGIN { if (n <= 1) { printf "%.8f", b } else { t=(g-1)/(n-1); printf "%.8f", a+(b-a)*t } }'
}

render_apex_command() {
  local template="$1"
  local input_path="$2"
  local output_path="$3"
  local generation="$4"
  local rendered="$template"
  rendered="${rendered//\{input\}/$input_path}"
  rendered="${rendered//\{output\}/$output_path}"
  rendered="${rendered//\{generation\}/$generation}"
  printf '%s' "$rendered"
}

mkdir -p "$RESULTS_ROOT"

echo "============================================================"
echo "AMP-JEPA V4B iterative evolution"
echo "Generations:       $START_GENERATION -> $END_GENERATION"
echo "Parents/generation: $N_PARENTS"
echo "Offspring/generation: $N_OFFSPRING"
echo "Survivors/generation: $N_SURVIVORS"
echo "Device:            $DEVICE"
echo "Checkpoint:        $CHECKPOINT"
echo "============================================================"

for (( GEN=START_GENERATION; GEN<=END_GENERATION; GEN++ )); do
  PREV=$((GEN - 1))
  GEN_PAD=$(printf "%02d" "$GEN")
  PREV_PAD=$(printf "%02d" "$PREV")
  OUTDIR="$RESULTS_ROOT/generation_${GEN_PAD}"
  PREVDIR="$RESULTS_ROOT/generation_${PREV_PAD}"
  mkdir -p "$OUTDIR"

  if (( PREV == 0 )); then
    SOURCE_POPULATION="$PREVDIR/generation_00_candidates.csv"
  else
    SOURCE_POPULATION="$PREVDIR/generation_${PREV_PAD}_survivors.csv"
  fi
  SOURCE_METADATA="$PREVDIR/latent_metadata.csv"
  SOURCE_LATENTS="$PREVDIR/latent_vectors.npz"

  PARENTS="$OUTDIR/generation_${GEN_PAD}_parents.csv"
  PARENT_SUMMARY="$OUTDIR/parent_selection_summary.json"
  CANDIDATES="$OUTDIR/generation_${GEN_PAD}_candidates_pre_apex.csv"
  PROPOSALS="$OUTDIR/generation_${GEN_PAD}_latent_proposals.npz"
  GENERATION_SUMMARY="$OUTDIR/generation_${GEN_PAD}_generation_summary.json"
  APEX_SCORED="$OUTDIR/generation_${GEN_PAD}_apex_scores.csv"
  SCORED_OFFSPRING="$OUTDIR/generation_${GEN_PAD}_scored_offspring.csv"
  SURVIVORS="$OUTDIR/generation_${GEN_PAD}_survivors.csv"
  SURVIVOR_SUMMARY="$OUTDIR/survivor_selection_summary.json"
  LATENT_METADATA="$OUTDIR/latent_metadata.csv"
  LATENT_VECTORS="$OUTDIR/latent_vectors.npz"
  LATENT_SUMMARY="$OUTDIR/latent_encoding_summary.json"
  COMPLETE_MARKER="$OUTDIR/GENERATION_COMPLETE"

  if [[ -f "$COMPLETE_MARKER" && -s "$SURVIVORS" && -s "$LATENT_VECTORS" ]]; then
    echo "[V4B:G${GEN_PAD}] Complete marker found; skipping generation."
    continue
  fi

  for required in "$SOURCE_POPULATION" "$SOURCE_METADATA" "$SOURCE_LATENTS"; do
    if [[ ! -s "$required" ]]; then
      echo "[V4B:G${GEN_PAD}] Missing required source file: $required" >&2
      exit 3
    fi
  done

  GEN_SEED=$((BASE_SEED + GEN * 1009))
  LATENT_SIGMA=$(schedule_value "$LATENT_SIGMA_START" "$LATENT_SIGMA_END" "$GEN" "$END_GENERATION")
  TEMPERATURE=$(schedule_value "$TEMPERATURE_START" "$TEMPERATURE_END" "$GEN" "$END_GENERATION")

  echo
  echo "------------------------------------------------------------"
  echo "[V4B:G${GEN_PAD}] Starting"
  echo "[V4B:G${GEN_PAD}] Source population: $SOURCE_POPULATION"
  echo "[V4B:G${GEN_PAD}] Latent sigma:     $LATENT_SIGMA"
  echo "[V4B:G${GEN_PAD}] Temperature:      $TEMPERATURE"
  echo "[V4B:G${GEN_PAD}] Seed:             $GEN_SEED"
  echo "------------------------------------------------------------"

  if [[ -s "$PARENTS" && -s "$PARENT_SUMMARY" ]]; then
    echo "[V4B:G${GEN_PAD}] Parent selection already complete; resuming."
  else
    echo "[V4B:G${GEN_PAD}] Selecting fitness-diverse parents"
    python v4b/02_select_parents.py \
      --generation "$GEN" \
      --metadata "$SOURCE_METADATA" \
      --latents "$SOURCE_LATENTS" \
      --outdir "$OUTDIR" \
      --n-parents "$N_PARENTS" \
      --seed "$GEN_SEED"
  fi

  if [[ -s "$CANDIDATES" && -s "$PROPOSALS" && -s "$GENERATION_SUMMARY" ]]; then
    echo "[V4B:G${GEN_PAD}] Offspring generation already complete; resuming."
  else
    echo "[V4B:G${GEN_PAD}] Generating latent offspring"
    python v4b/03_generate_offspring.py \
      --generation "$GEN" \
      --parents "$PARENTS" \
      --source-population "$SOURCE_POPULATION" \
      --latents "$SOURCE_LATENTS" \
      --checkpoint "$CHECKPOINT" \
      --results-root "$RESULTS_ROOT" \
      --outdir "$OUTDIR" \
      --n-offspring "$N_OFFSPRING" \
      --proposal-multiplier "$PROPOSAL_MULTIPLIER" \
      --batch-size "$BATCH_SIZE" \
      --latent-sigma "$LATENT_SIGMA" \
      --temperature "$TEMPERATURE" \
      --crossover-rate "$CROSSOVER_RATE" \
      --seed "$GEN_SEED" \
      --device "$DEVICE"
  fi

  if [[ -s "$APEX_SCORED" ]]; then
    echo "[V4B:G${GEN_PAD}] APEX scores already exist; resuming."
  else
    if [[ -z "$APEX_SCORER_CMD" ]]; then
      cat >&2 <<EOF
[V4B:G${GEN_PAD}] APEX_SCORER_CMD is required because no score file exists.
Set it once as a command template containing these placeholders:
  {input}       candidate CSV
  {output}      required APEX score CSV
  {generation}  generation number

Example shape:
  export APEX_SCORER_CMD='python /path/to/apex_scorer.py --input {input} --output {output}'

The scorer must write one row per candidate and preserve candidate_id or sequence.
EOF
      exit 4
    fi
    APEX_COMMAND=$(render_apex_command "$APEX_SCORER_CMD" "$CANDIDATES" "$APEX_SCORED" "$GEN")
    echo "[V4B:G${GEN_PAD}] Running APEX scorer"
    echo "[V4B:G${GEN_PAD}] $APEX_COMMAND"
    eval "$APEX_COMMAND"
    if [[ ! -s "$APEX_SCORED" ]]; then
      echo "[V4B:G${GEN_PAD}] APEX command finished but output is missing/empty: $APEX_SCORED" >&2
      exit 5
    fi
  fi

  if [[ -s "$SURVIVORS" && -s "$SCORED_OFFSPRING" && -s "$SURVIVOR_SUMMARY" ]]; then
    echo "[V4B:G${GEN_PAD}] Survivor selection already complete; resuming."
  else
    echo "[V4B:G${GEN_PAD}] Selecting APEX-aware fitness-diverse survivors"
    python v4b/04_select_survivors.py \
      --generation "$GEN" \
      --source-population "$SOURCE_POPULATION" \
      --source-latents "$SOURCE_LATENTS" \
      --offspring "$CANDIDATES" \
      --offspring-scored "$APEX_SCORED" \
      --offspring-latents "$PROPOSALS" \
      --outdir "$OUTDIR" \
      --n-survivors "$N_SURVIVORS" \
      --seed "$GEN_SEED"
  fi

  if [[ -s "$LATENT_VECTORS" && -s "$LATENT_METADATA" && -s "$LATENT_SUMMARY" ]]; then
    echo "[V4B:G${GEN_PAD}] Survivor encoding already complete; resuming."
  else
    echo "[V4B:G${GEN_PAD}] Re-encoding survivor population"
    python v4b/05_encode_population.py \
      --generation "$GEN" \
      --input "$SURVIVORS" \
      --checkpoint "$CHECKPOINT" \
      --outdir "$OUTDIR" \
      --batch-size "$ENCODE_BATCH_SIZE" \
      --device "$DEVICE"
  fi

  python - "$GEN" "$SURVIVORS" "$LATENT_VECTORS" "$LATENT_METADATA" <<'PY'
import sys
from pathlib import Path
import numpy as np
import pandas as pd

generation = int(sys.argv[1])
survivor_path = Path(sys.argv[2])
latent_path = Path(sys.argv[3])
metadata_path = Path(sys.argv[4])

survivors = pd.read_csv(survivor_path, low_memory=False)
metadata = pd.read_csv(metadata_path, low_memory=False)
latent = np.load(latent_path)
ids = latent["candidate_id"].astype(str)

assert len(survivors) == len(metadata) == len(ids), "Final row-count mismatch"
assert survivors["candidate_id"].astype(str).is_unique, "Duplicate survivor IDs"
assert survivors["sequence"].astype(str).is_unique, "Duplicate survivor sequences"
assert np.array_equal(metadata["candidate_id"].astype(str).to_numpy(), ids), "Latent/metadata IDs misaligned"
assert not np.isnan(latent["mu"]).any(), "NaN in latent mu"
assert not np.isinf(latent["mu"]).any(), "Inf in latent mu"
print(f"[V4B:G{generation:02d}] Integrity check passed for {len(survivors):,} survivors")
PY

  date -u +"%Y-%m-%dT%H:%M:%SZ" > "$COMPLETE_MARKER"
  echo "[V4B:G${GEN_PAD}] Complete"
done

python - "$RESULTS_ROOT" "$END_GENERATION" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

root = Path(sys.argv[1])
end_generation = int(sys.argv[2])
generations = []
for generation in range(1, end_generation + 1):
    directory = root / f"generation_{generation:02d}"
    survivor_summary = directory / "survivor_selection_summary.json"
    generation_summary = directory / f"generation_{generation:02d}_generation_summary.json"
    latent_summary = directory / "latent_encoding_summary.json"
    if not (survivor_summary.exists() and generation_summary.exists() and latent_summary.exists()):
        continue
    generations.append({
        "generation": generation,
        "generation_summary": json.loads(generation_summary.read_text()),
        "survivor_summary": json.loads(survivor_summary.read_text()),
        "latent_summary": json.loads(latent_summary.read_text()),
    })
manifest = {
    "schema_version": "1.0",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "completed_generations": [item["generation"] for item in generations],
    "generations": generations,
}
(root / "v4b_evolution_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
print(f"[V4B] Evolution manifest written for {len(generations)} completed generations")
PY

echo "============================================================"
echo "[V4B] Requested generation range complete"
echo "[V4B] Manifest: $RESULTS_ROOT/v4b_evolution_manifest.json"
echo "============================================================"
