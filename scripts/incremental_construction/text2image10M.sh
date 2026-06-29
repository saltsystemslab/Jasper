#!/bin/bash

FILENAME="/projects/SaltSystemsLab/ann_data/text2image/text2image10M"
DATATYPE="float"
DIM="200"
INDEX_OUT="/tmp/text2image10M-incremental-index"

DISTANCE="ip"
N_NEIGHBORS=64
ALPHA=1.0
PIECES=10
WORKSPACE_BUDGET="10GB"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd ../.. && pwd)"
BINARY="$SCRIPT_DIR/build/bin/incremental_construction"
LOG_FILE="$SCRIPT_DIR/results/incremental_construct_text2image10M.log"

if [[ ! -x "$BINARY" ]]; then
    echo "ERROR: Binary not found or not executable: $BINARY"
    exit 1
fi

"$BINARY" \
    --filename        "$FILENAME" \
    --datatype        "$DATATYPE" \
    --dim             "$DIM" \
    --index_out       "$INDEX_OUT" \
    --distance        "$DISTANCE" \
    --n_neighbors     "$N_NEIGHBORS" \
    --alpha           "$ALPHA" \
    --pieces          "$PIECES" \
    --workspace_budget "$WORKSPACE_BUDGET" \
    2>&1 | tee "$LOG_FILE"
