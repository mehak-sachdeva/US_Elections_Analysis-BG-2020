#!/bin/bash
# Bandwidth search. The longest step in the pipeline by a wide margin.
#
# Not fire-and-forget: tail the log and watch the bandwidth vector printed at
# the end of each backfitting iteration. Paste the converged values from
# bandwidths.txt into src/config.py before running step 02.

set -euo pipefail

INPUT="${INPUT:-Final_submitted_data.csv}"
NPROC="${NPROC:-220}"
LOG="bandwidth_search_$(date +%Y%m%d_%H%M%S).log"

echo "Writing to $LOG"
echo "Watch with:  tail -f $LOG | grep -i bandwidth"

nohup python3 ../src/01_bandwidth_search.py \
    --input "$INPUT" \
    --out bandwidths.txt \
    --nproc "$NPROC" \
    > "$LOG" 2>&1 &

echo "Started as PID $!"
