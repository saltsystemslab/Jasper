#!/bin/bash
#SBATCH --partition=gpu-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:h200
#SBATCH --job-name=fused_bench
#SBATCH --output=fused_bench_%j.log
#SBATCH --error=fused_bench_%j.err
#SBATCH --time=01:30:00

set -euo pipefail

module load cuda
module load cmake

cd /projects/SaltSystemsLab/hunter/jasperpy

module unload cuda || true
module unload cmake || true
set +u
source ~/.bashrc || true   # ~/.bashrc line 29 (~/.local/bin/env missing) returns non-zero under set -e
conda activate /projects/SaltSystemsLab/zikun/env
set -u
THRUST_DIR=/projects/SaltSystemsLab/zikun/env/lib/rapids/cmake/thrust
DATA=/projects/SaltSystemsLab/ann_data
GT=/projects/SaltSystemsLab/ann_gt

echo "=== branch ===" && git rev-parse --abbrev-ref HEAD && git log --oneline -1

set -e
cmake -B build -DJASPER_BUILD_CMD=ON -DJASPER_BUILD_TESTS=OFF -DJASPER_BUILD_FFI=OFF \
  -DThrust_DIR="$THRUST_DIR" -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" >/dev/null
# --clean-first: NFS clock skew has produced stale/incomplete objects; force a
# fresh compile of the target so we never run a corrupt binary.
cmake --build build --target fused_query --clean-first -j8

BASE="$DATA/bigann/bigann10M"
Q="$DATA/bigann/bigann10kquery"
GTF="$GT/GT_10M/bigann-10M"

echo ""
echo "############### bigann10M — Trick 3: block-size + early-termination sweeps (fused+cache) ###############"
./build/bin/fused_query \
  --base "$BASE" --queries "$Q" --groundtruth "$GTF" \
  -t uint8 -d l2 -n 64 --dim 128 -r 8 -k 10 \
  --seeds 32 --coarse 64:128 \
  --subsample 0.1 --coarse_R 32 --sampler random \
  --pq_train 40000 \
  --beam_limits 16:128 32:128 64:128 128:256 256:512 \
  2>&1 | grep -iE "config|subsample|sampler|seeds|coarse|full graph|block-size|early-term|baseline|fused|cache|recall|QPS|visited|mem|Done|Error"

echo "=== done ==="
