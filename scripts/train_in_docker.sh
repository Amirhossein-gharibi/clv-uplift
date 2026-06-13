#!/usr/bin/env bash
# scripts/train_in_docker.sh
# Train inside the SAME image that serves (ABI match: identical numpy/econml/sklearn).
# Data mounts in read-only; the trained artifact and figures mount out writable; the
# training script mounts in read-only (it is not baked into the serving image).
#
# Prerequisite: `docker compose build api` so clv-uplift-api:latest exists,
# and data/raw/online_retail_II.xlsx present on the host.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_FILE="$ROOT/data/raw/online_retail_II.xlsx"

if [[ ! -f "$DATA_FILE" ]]; then
  echo "ERROR: $DATA_FILE not found." >&2
  echo "Download the UCI Online Retail II dataset to data/raw/ before training." >&2
  exit 1
fi

if ! docker image inspect clv-uplift-api:latest >/dev/null 2>&1; then
  echo "ERROR: image clv-uplift-api:latest not found. Run 'docker compose build api' first." >&2
  exit 1
fi

mkdir -p "$ROOT/artifacts/figures"

echo "Training inside clv-uplift-api:latest (same image that serves) ..."
docker run --rm \
  --volume "$ROOT/data/raw:/app/data/raw:ro" \
  --volume "$ROOT/artifacts:/app/artifacts:rw" \
  --volume "$ROOT/notebooks:/app/notebooks:ro" \
  --workdir /app \
  clv-uplift-api:latest \
  python /app/notebooks/01_uplift_training.py

echo "Done. Artifact written to $ROOT/artifacts/uplift_model.pkl"