#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG_FILE="${V4B_CONFIG:-v4b/v4b_config.env}"
if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "[V4B] Configuration file not found: $CONFIG_FILE" >&2
  echo "[V4B] Create it once with:" >&2
  echo "       cp v4b/v4b_config.env.example v4b/v4b_config.env" >&2
  exit 2
fi

# shellcheck disable=SC1090
set -a
source "$CONFIG_FILE"
set +a

exec bash v4b/run_v4b_generations.sh
