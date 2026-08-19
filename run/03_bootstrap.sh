#!/bin/bash
# Residual bootstrap. Ten batches of ten replicates gives the 100 used for
# inference. Batching means a crash costs one batch rather than the whole run.
#
# Runs the batches in sequence. To spread them across machines, call
# src/03_bootstrap.py directly with a different --batch on each.

set -euo pipefail

INPUT="${INPUT:-Final_submitted_data.csv}"

# Written by step 02. Not distributed: generate it by running
# run/02_mgwr_fit.sh first.
PARAMS="${PARAMS:-country_ind_params.csv}"
N_BATCHES="${N_BATCHES:-10}"
N_BOOTSTRAP="${N_BOOTSTRAP:-10}"
NPROC="${NPROC:-220}"

echo "$N_BATCHES batches of $N_BOOTSTRAP replicates"

for i in $(seq -w 1 "$N_BATCHES"); do
    LOG="bootstrap_${i}_$(date +%Y%m%d_%H%M%S).log"
    echo "batch $i -> $LOG"
    python3 ../src/03_bootstrap.py \
        --input "$INPUT" \
        --params "$PARAMS" \
        --n-bootstrap "$N_BOOTSTRAP" \
        --batch "$i" \
        --nproc "$NPROC" \
        > "$LOG" 2>&1
done

echo "Done. Next: src/04_compile_bootstrap.py and src/05_composition_ci.py"
