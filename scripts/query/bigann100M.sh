#!/bin/bash

INDEX="/tmp/bigann100M-index" 
QUERIES="/projects/SaltSystemsLab/ann_data/bigann/bigann10kquery"
DATATYPE="uint8"
DIM="128"

GROUNDTRUTH="/projects/SaltSystemsLab/ann_gt/GT_100M/bigann-100M"
DISTANCE="l2"
N_NEIGHBORS=64
BEAM_LIMITS="1:128 2:128 4:128 8:128 16:128 32:128 64:128 128:256 256:512 512:1024 1024:2048 1536:2048 1982:2048"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd ../.. && pwd)"
BINARY="$SCRIPT_DIR/build/bin/run_query"
LOG_BASE="$SCRIPT_DIR/results/query_bigann100M"

mkdir -p "$(dirname "$LOG_BASE")"

if [[ ! -x "$BINARY" ]]; then
    echo "ERROR: Binary not found or not executable: $BINARY"
    exit 1
fi

optional_args=()

[[ -n "$GROUNDTRUTH" ]] && optional_args+=(--groundtruth "$GROUNDTRUTH")

for bl in $BEAM_LIMITS; do
    optional_args+=(--beam_limits "$bl")
done

for K in 1 10 50 100; do
    LOG_FILE="${LOG_BASE}_k${K}.log"

    echo "========================================" | tee "$LOG_FILE"
    echo "Running with K=$K" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"

    "$BINARY" \
        --index        "$INDEX" \
        --queries      "$QUERIES" \
        --datatype     "$DATATYPE" \
        --dim          "$DIM" \
        --distance     "$DISTANCE" \
        --n_neighbors  "$N_NEIGHBORS" \
        --k            "$K" \
        "${optional_args[@]}" \
        2>&1 | tee -a "$LOG_FILE"
done