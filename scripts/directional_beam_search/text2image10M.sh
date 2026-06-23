#!/bin/bash

FILENAME="/projects/SaltSystemsLab/ann_data/text2image/text2image10M"
QUERIES="/projects/SaltSystemsLab/ann_data/text2image/text2image100kquery"
GROUNDTRUTH="/projects/SaltSystemsLab/ann_gt/GT_10M/text2image-10M"
DATATYPE="float"
DIM="200"

DISTANCE="ip"
N_NEIGHBORS=64
ALPHA=1.0
WORKSPACE_BUDGET="10GB"
K_RANKS=16
BEAM_LIMITS="1:128 2:128 4:128 8:128 16:128 32:128 64:128 128:256 256:512 512:1024 1024:2048 1536:2048 1982:2048"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd ../.. && pwd)"
BINARY="$SCRIPT_DIR/build/bin/create_lsh_index"
LOG_BASE="$SCRIPT_DIR/results/dbs_text2image10M"

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
        --filename         "$FILENAME" \
        --datatype         "$DATATYPE" \
        --dim              "$DIM" \
        --distance         "$DISTANCE" \
        --n_neighbors      "$N_NEIGHBORS" \
        --alpha            "$ALPHA" \
        --workspace_budget "$WORKSPACE_BUDGET" \
        --k_ranks          "$K_RANKS" \
        --queries          "$QUERIES" \
        --k                "$K" \
        "${optional_args[@]}" \
        2>&1 | tee -a "$LOG_FILE"
done
