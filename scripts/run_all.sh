#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-mock}"
CONFIG="configs/${MODE}.yaml"
OUT="runs/${MODE}"

python scripts/run_matrix.py --config "${CONFIG}"
python scripts/summarize_runs.py --matrix_out "${OUT}"

echo "done: ${OUT}/results.csv ${OUT}/summary.json"