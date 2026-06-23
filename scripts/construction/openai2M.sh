#!/bin/bash

FILENAME="/projects/SaltSystemsLab/ann_data/openai/openai_base.bin"    
DATATYPE="float"
DIM="1536"         
INDEX_OUT="/tmp/openai2M-index"   

DISTANCE="l2"
N_NEIGHBORS=64
ALPHA=1.2      
WORKSPACE_BUDGET="10GB"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd ../.. && pwd)"
BINARY="$SCRIPT_DIR/build/bin/create_index"
LOG_FILE="$SCRIPT_DIR/results/construct_openai2M.log"

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
    --workspace_budget "$WORKSPACE_BUDGET" \
    2>&1 | tee "$LOG_FILE"