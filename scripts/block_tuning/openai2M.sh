#!/bin/bash

FILENAME="/projects/SaltSystemsLab/ann_data/openai/openai_base.bin"
QUERIES="/projects/SaltSystemsLab/ann_data/openai/openai_query.bin"
GROUNDTRUTH="/projects/SaltSystemsLab/ann_gt/openai_gt"
DATATYPE="float"
DIM="1536"

DISTANCE="l2"
N_NEIGHBORS=64
ALPHA=1.2
WORKSPACE_BUDGET="10GB"
K=10
BEAM_LIMITS="1:128 2:128 4:128 8:128 16:128 32:128 64:128 128:256 256:512 512:1024 1024:2048 1536:2048 1982:2048"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd ../.. && pwd)"
BINARY="$SCRIPT_DIR/build/bin/create_lsh_index_block_tuning"
LOG_FILE="$SCRIPT_DIR/results/block_tuning_openai2M_k${K}.log"

mkdir -p "$(dirname "$LOG_FILE")"

if [[ ! -x "$BINARY" ]]; then
    echo "ERROR: Binary not found or not executable: $BINARY"
    exit 1
fi

optional_args=()

[[ -n "$GROUNDTRUTH" ]] && optional_args+=(--groundtruth "$GROUNDTRUTH")

optional_args+=(--beam_limits $BEAM_LIMITS)

"$BINARY" \
    --filename         "$FILENAME" \
    --queries          "$QUERIES" \
    --datatype         "$DATATYPE" \
    --dim              "$DIM" \
    --distance         "$DISTANCE" \
    --n_neighbors      "$N_NEIGHBORS" \
    --alpha            "$ALPHA" \
    --workspace_budget "$WORKSPACE_BUDGET" \
    --k                "$K" \
    "${optional_args[@]}" \
    2>&1 | tee "$LOG_FILE"
