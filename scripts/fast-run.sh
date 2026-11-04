#!/usr/bin/env bash
# Run every step from scratch, all in Docker:
#   new blurred images -> a freshly trained model -> the comparison figure -> the web demo's model.
set -euo pipefail
cd "$(dirname "$0")/.."
rm -f model/model.h5
docker compose build
docker compose run --rm detector generate
docker compose run --rm detector run
docker compose run --rm detector export
echo "Done: model/model.h5, outputs/results.png and src/web/public/model/"
