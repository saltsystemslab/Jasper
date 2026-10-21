#!/bin/bash
#SBATCH --partition=gpu-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:h200
#SBATCH --job-name=test_rabitq_gist
#SBATCH --output=test_rabitq_gist_%j.log
#SBATCH --error=test_rabitq_gist_%j.err
#SBATCH --time=02:00:00

# RaBitQ unit tests, then exact vs RaBitQ beam search on GIST-1M (R=64, k=10).
# Submit from the repo root:  sbatch sbatch/test_rabitq_gist.sbatch
# The GIST index is built once into perf_tmp/ and reused by later runs.

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

set +u
source ~/.bashrc
conda activate /projects/SaltSystemsLab/zikun/env
set -u
THRUST_DIR=/projects/SaltSystemsLab/zikun/env/lib/rapids/cmake/thrust
DATA=/projects/SaltSystemsLab/ann_data/gist
GT=/projects/SaltSystemsLab/ann_gt/gist-gt
IDX=perf_tmp/gist_r64.idx
BEAMS="10:128 16:128 32:128 64:128 128:256 256:512"
mkdir -p perf_tmp

echo "=== build ($(git rev-parse --short HEAD), $(git status --short | wc -l) dirty files) ==="
cmake -B build -DJASPER_BUILD_CMD=ON -DJASPER_BUILD_TESTS=ON -DJASPER_BUILD_FFI=OFF \
  -DThrust_DIR="$THRUST_DIR" -DCMAKE_PREFIX_PATH="$CONDA_PREFIX"
cmake --build build --target jasper_tests create_index run_query -j16

echo ""; echo "=== unit tests (RaBitQ + rotation) ==="
./build/tests/jasper_tests \
  --gtest_filter='EstimatorTest.*:QuantizeTest.*:DataStoreTest.*:BitExtractionTest.*:RotationTest.*'

if [ ! -f "$IDX" ]; then
  echo ""; echo "=== build GIST index -> $IDX ==="
  ./build/bin/create_index -f "$DATA/gist_base.bin" -t float -d l2 -n 64 --dim 960 \
    -i "$IDX" -w 10GB
fi

echo ""; echo "=== exact beam search ==="
# shellcheck disable=SC2086
./build/bin/run_query -i "$IDX" -q "$DATA/gist_query.bin" -g "$GT" -t float -d l2 \
  -n 64 --dim 960 -k 10 --beam_limits $BEAMS

for BITS in 1 4 8; do
  echo ""; echo "=== RaBitQ beam search, ${BITS} bits/dim ==="
  # shellcheck disable=SC2086
  ./build/bin/run_query -i "$IDX" -q "$DATA/gist_query.bin" -g "$GT" -t float -d l2 \
    -n 64 --dim 960 -k 10 --beam_limits $BEAMS --rabitq_bits "$BITS"
done

echo ""; echo "=== done ==="
