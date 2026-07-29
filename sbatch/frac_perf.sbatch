#!/bin/bash
#SBATCH --partition=gpu-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:h200
#SBATCH --job-name=frac_perf
#SBATCH --output=frac_perf_%j.log
#SBATCH --error=frac_perf_%j.err
#SBATCH --time=01:30:00

set -euo pipefail
module load cuda
module load cmake
cd /projects/SaltSystemsLab/hunter/jasperpy
module unload cuda || true
module unload cmake || true
set +u
source ~/.bashrc || true
conda activate /projects/SaltSystemsLab/zikun/env
set -u
THRUST_DIR=/projects/SaltSystemsLab/zikun/env/lib/rapids/cmake/thrust
DATA=/projects/SaltSystemsLab/ann_data
GT=/projects/SaltSystemsLab/ann_gt
BASE="$DATA/bigann/bigann10M"; Q="$DATA/bigann/bigann10kquery"; GTF="$GT/GT_10M/bigann-10M"

echo "=== branch ===" && git rev-parse --abbrev-ref HEAD && git log --oneline -1
set -e
cmake -B build -DJASPER_BUILD_CMD=ON -DJASPER_BUILD_TESTS=OFF -DJASPER_BUILD_FFI=OFF \
  -DThrust_DIR="$THRUST_DIR" -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" >/dev/null
cmake --build build --target fused_query --clean-first -j8

echo ""
echo "############### bigann10M — new-mode PERFORMANCE vs device-Jasper size ###############"
for F in 0.005 0.01 0.02 0.05 0.1; do
  echo ""
  echo ">>>>>>>>>>>>>>> subsample=$F <<<<<<<<<<<<<<<"
  ./build/bin/fused_query \
    --base "$BASE" --queries "$Q" --groundtruth "$GTF" \
    -t uint8 -d l2 -n 64 --dim 128 -r 8 -k 10 \
    --seeds 32 --coarse 64:128 --subsample "$F" --coarse_R 32 --sampler random \
    --pq_train 40000 \
    --beam_limits 64:128 128:256 \
    2>&1 | grep -iE "subsample:|baseline|fused|cache b256|c256 s1.10|recall|QPS"
done
echo "=== done ==="
