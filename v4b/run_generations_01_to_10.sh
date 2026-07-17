#!/usr/bin/env bash
set -euo pipefail

# Closed-loop V4B evolutionary optimization:
# parent selection -> latent generation -> APEX scoring -> survivor selection -> repeat.
#
# APEX hook:
#   export APEX_SCORE_CMD='python path/to/apex_score.py --input {input} --output {output}'
#
# Safety:
#   REQUIRE_APEX=1 makes the loop fail if APEX_SCORE_CMD is absent.
#   REQUIRE_APEX=0 allows a fallback passthrough so the mechanics can be tested.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

START_GENERATION="${START_GENERATION:-1}"
END_GENERATION="${END_GENERATION:-10}"

N_PARENTS="${N_PARENTS:-512}"
N_OFFSPRING="${N_OFFSPRING:-10000}"
N_SURVIVORS="${N_SURVIVORS:-2048}"

DEVICE="${DEVICE:-auto}"
BATCH_SIZE="${BATCH_SIZE:-512}"
TEMPERATURE="${TEMPERATURE:-0.85}"
LATENT_SIGMA="${LATENT_SIGMA:-0.35}"
CROSSOVER_RATE="${CROSSOVER_RATE:-0.20}"
PROPOSAL_MULTIPLIER="${PROPOSAL_MULTIPLIER:-4.0}"

CHECKPOINT="${CHECKPOINT:-v3/checkpoints/amp_jepa_hybrid_v3_qc_core.pt}"
FITNESS_COLUMN="${FITNESS_COLUMN:-}"
SURVIVOR_FITNESS_COLUMN="${SURVIVOR_FITNESS_COLUMN:-${FITNESS_COLUMN}}"
APEX_SCORE_CMD="${APEX_SCORE_CMD:-}"
REQUIRE_APEX="${REQUIRE_APEX:-0}"
OVERWRITE="${OVERWRITE:-0}"
SEED="${SEED:-20260716}"

if [[ ! -f "v4b/results/generation_00/generation_00_candidates.csv" ]]; then
  echo "[V4B] Missing Generation 0 candidates. Run v4b/run_generation_00.sh first." >&2
  exit 1
fi

if [[ ! -f "v4b/results/generation_00/latent_vectors.npz" ]]; then
  echo "[V4B] Missing Generation 0 latent archive. Run v4b/run_generation_00.sh first." >&2
  exit 1
fi

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "[V4B] Missing checkpoint: ${CHECKPOINT}" >&2
  exit 1
fi

if [[ "${REQUIRE_APEX}" == "1" && -z "${APEX_SCORE_CMD}" ]]; then
  echo "[V4B] REQUIRE_APEX=1 but APEX_SCORE_CMD is empty." >&2
  echo "[V4B] Example: export APEX_SCORE_CMD='python scripts/apex_score.py --input {input} --output {output}'" >&2
  exit 1
fi

common_overwrite=()
if [[ "${OVERWRITE}" == "1" ]]; then
  common_overwrite=(--overwrite)
fi

fitness_args=()
if [[ -n "${FITNESS_COLUMN}" ]]; then
  fitness_args=(--fitness-column "${FITNESS_COLUMN}")
fi

survivor_fitness_args=()
if [[ -n "${SURVIVOR_FITNESS_COLUMN}" ]]; then
  survivor_fitness_args=(--fitness-column "${SURVIVOR_FITNESS_COLUMN}")
fi

score_args=()
if [[ -n "${APEX_SCORE_CMD}" ]]; then
  score_args=(--apex-command "${APEX_SCORE_CMD}")
fi
if [[ "${REQUIRE_APEX}" == "1" ]]; then
  score_args+=(--require-apex)
fi

printf '[V4B] Closed-loop run: generations %s to %s\n' "${START_GENERATION}" "${END_GENERATION}"
printf '[V4B] N_PARENTS=%s N_OFFSPRING=%s N_SURVIVORS=%s DEVICE=%s\n' "${N_PARENTS}" "${N_OFFSPRING}" "${N_SURVIVORS}" "${DEVICE}"

for (( GEN=START_GENERATION; GEN<=END_GENERATION; GEN++ )); do
  GPAD=$(printf "%02d" "${GEN}")
  GTAG="generation_${GPAD}"
  OUTDIR="v4b/results/${GTAG}"
  mkdir -p "${OUTDIR}"

  echo
  echo "============================================================"
  echo "[V4B] ${GTAG}"
  echo "============================================================"

  if [[ "${GEN}" -eq 1 ]]; then
    POPULATION="v4b/results/generation_00/generation_00_candidates.csv"
    POP_LATENTS="v4b/results/generation_00/latent_vectors.npz"
  else
    PREV=$((GEN - 1))
    PREV_PAD=$(printf "%02d" "${PREV}")
    POPULATION="v4b/results/generation_${PREV_PAD}/generation_${PREV_PAD}_survivors.csv"
    POP_LATENTS="v4b/results/generation_${PREV_PAD}/generation_${PREV_PAD}_survivor_latents.npz"
  fi

  if [[ ! -f "${POPULATION}" ]]; then
    echo "[V4B] Missing parent population for ${GTAG}: ${POPULATION}" >&2
    exit 1
  fi
  if [[ ! -f "${POP_LATENTS}" ]]; then
    echo "[V4B] Missing parent latents for ${GTAG}: ${POP_LATENTS}" >&2
    exit 1
  fi

  PARENTS="${OUTDIR}/${GTAG}_parents.csv"
  PRE_APEX="${OUTDIR}/${GTAG}_candidates_pre_apex.csv"
  PROPOSALS="${OUTDIR}/${GTAG}_latent_proposals.npz"
  SCORED="${OUTDIR}/${GTAG}_candidates_scored.csv"
  SURVIVORS="${OUTDIR}/${GTAG}_survivors.csv"
  SURVIVOR_LATENTS="${OUTDIR}/${GTAG}_survivor_latents.npz"

  if [[ ! -f "${PARENTS}" || "${OVERWRITE}" == "1" ]]; then
    echo "[V4B] Selecting parents for ${GTAG}"
    python v4b/select_generation_parents.py \
      --population "${POPULATION}" \
      --latents "${POP_LATENTS}" \
      --generation "${GEN}" \
      --outdir "${OUTDIR}" \
      --n-parents "${N_PARENTS}" \
      --seed "${SEED}" \
      "${fitness_args[@]}" \
      "${common_overwrite[@]}"
  else
    echo "[V4B] Parents already exist; skipping parent selection: ${PARENTS}"
  fi

  exclude_args=(--exclude-csv "v4b/results/generation_00/generation_00_candidates.csv")
  for (( OLD=1; OLD<GEN; OLD++ )); do
    OLD_PAD=$(printf "%02d" "${OLD}")
    OLD_SCORED="v4b/results/generation_${OLD_PAD}/generation_${OLD_PAD}_candidates_scored.csv"
    OLD_PRE="v4b/results/generation_${OLD_PAD}/generation_${OLD_PAD}_candidates_pre_apex.csv"
    if [[ -f "${OLD_SCORED}" ]]; then
      exclude_args+=(--exclude-csv "${OLD_SCORED}")
    elif [[ -f "${OLD_PRE}" ]]; then
      exclude_args+=(--exclude-csv "${OLD_PRE}")
    fi
  done

  if [[ ! -f "${PRE_APEX}" || ! -f "${PROPOSALS}" || "${OVERWRITE}" == "1" ]]; then
    echo "[V4B] Generating offspring for ${GTAG}"
    python v4b/generate_generation.py \
      --parents "${PARENTS}" \
      --parent-latents "${POP_LATENTS}" \
      --generation "${GEN}" \
      --checkpoint "${CHECKPOINT}" \
      --outdir "${OUTDIR}" \
      --n-offspring "${N_OFFSPRING}" \
      --proposal-multiplier "${PROPOSAL_MULTIPLIER}" \
      --batch-size "${BATCH_SIZE}" \
      --temperature "${TEMPERATURE}" \
      --latent-sigma "${LATENT_SIGMA}" \
      --crossover-rate "${CROSSOVER_RATE}" \
      --device "${DEVICE}" \
      --seed "${SEED}" \
      "${exclude_args[@]}" \
      "${common_overwrite[@]}"
  else
    echo "[V4B] Offspring already exist; skipping generation: ${PRE_APEX}"
  fi

  if [[ ! -f "${SCORED}" || "${OVERWRITE}" == "1" ]]; then
    echo "[V4B] Scoring ${GTAG} candidates"
    python v4b/score_generation.py \
      --input "${PRE_APEX}" \
      --output "${SCORED}" \
      --generation "${GEN}" \
      "${score_args[@]}" \
      "${common_overwrite[@]}"
  else
    echo "[V4B] Scored candidates already exist; skipping scoring: ${SCORED}"
  fi

  if [[ ! -f "${SURVIVORS}" || ! -f "${SURVIVOR_LATENTS}" || "${OVERWRITE}" == "1" ]]; then
    echo "[V4B] Selecting survivors for ${GTAG}"
    python v4b/select_generation_survivors.py \
      --scored-candidates "${SCORED}" \
      --candidate-latents "${PROPOSALS}" \
      --generation "${GEN}" \
      --outdir "${OUTDIR}" \
      --n-survivors "${N_SURVIVORS}" \
      --seed "${SEED}" \
      "${survivor_fitness_args[@]}" \
      "${common_overwrite[@]}"
  else
    echo "[V4B] Survivors already exist; skipping survivor selection: ${SURVIVORS}"
  fi

done

echo
echo "[V4B] Closed-loop V4B evolutionary run complete."
echo "[V4B] Final generation directory: v4b/results/generation_$(printf "%02d" "${END_GENERATION}")"
