#!/usr/bin/env bash
# Build the image, then train on ./data and draw outputs/results.png (same as `docker compose up`).
set -euo pipefail
cd "$(dirname "$0")/.."
docker compose build
docker compose up
