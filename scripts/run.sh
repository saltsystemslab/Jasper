#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd .. && pwd)"
chmod +x $SCRIPT_DIR/scripts/**/*.sh
mkdir -p $SCRIPT_DIR/results

# Construction
# $SCRIPT_DIR/scripts/construction/bigann100M.sh
# $SCRIPT_DIR/scripts/construction/deep100M.sh
# $SCRIPT_DIR/scripts/construction/gist1M.sh
# $SCRIPT_DIR/scripts/construction/openai2M.sh
# $SCRIPT_DIR/scripts/construction/text2image10M.sh

# # Query
# $SCRIPT_DIR/scripts/query/bigann100M.sh
# $SCRIPT_DIR/scripts/query/deep100M.sh
# $SCRIPT_DIR/scripts/query/gist1M.sh
# $SCRIPT_DIR/scripts/query/openai2M.sh
# $SCRIPT_DIR/scripts/query/text2image10M.sh

# # Block tuning
# $SCRIPT_DIR/scripts/block_tuning/bigann100M.sh
# $SCRIPT_DIR/scripts/block_tuning/openai2M.sh

# Incremental construction
$SCRIPT_DIR/scripts/incremental_construction/bigann100M.sh
$SCRIPT_DIR/scripts/incremental_construction/deep100M.sh
$SCRIPT_DIR/scripts/incremental_construction/gist1M.sh
$SCRIPT_DIR/scripts/incremental_construction/openai2M.sh
$SCRIPT_DIR/scripts/incremental_construction/text2image10M.sh