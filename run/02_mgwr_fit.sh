#!/bin/bash
# Fit at the bandwidths in src/config.py. No search is performed, so this is
# far cheaper than step 01, but it still holds the full parameter surface and
# the backfitting working arrays in memory.

set -euo pipefail

INPUT="${INPUT:-Final_submitted_data.csv}"
NPROC="${NPROC:-220}"
LOG="mgwr_fit_$(date +%Y%m%d_%H%M%S).log"

echo "Writing to $LOG"

nohup python3 ../src/02_mgwr_fit.py \
    --input "$INPUT" \
    --out country_ind_params.csv \
    --nproc "$NPROC" \
    > "$LOG" 2>&1 &

echo "Started as PID $!"
