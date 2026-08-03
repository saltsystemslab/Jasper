#!/bin/bash
#SBATCH --partition=SaltSystemsLab
#SBATCH --nodelist=d4067
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --job-name=verify_stable
#SBATCH --output=verify_stable_%j.log
#SBATCH --error=verify_stable_%j.err
#SBATCH --time=01:00:00

set -euo pipefail

module load cuda
module load cmake

cd /projects/SaltSystemsLab/hunter/jasperpy

module unload cuda || true
module unload cmake || true
set +u
source ~/.bashrc
conda activate /projects/SaltSystemsLab/zikun/env
set -u
ulimit -c 0
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
THRUST_DIR=/projects/SaltSystemsLab/zikun/env/lib/rapids/cmake/thrust
DATA=/projects/SaltSystemsLab/ann_data
GT=/projects/SaltSystemsLab/ann_gt
echo "### GPU ###"; nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader || true

set -e
# Default flags: JASPER_STABLE_IDS=OFF, JASPER_ENABLE_COMPACT=OFF (no WarpSpeed).
cmake -B build -DJASPER_BUILD_CMD=ON -DJASPER_BUILD_TESTS=OFF -DJASPER_BUILD_FFI=OFF \
  -DThrust_DIR="$THRUST_DIR" -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" 2>&1 | grep -iE "warpSpeed|STABLE|COMPACT|Configuring done" || true
cmake --build build --target create_index run_query bench_all test_deletion test_append -j8 >/dev/null
echo "=== cmd BUILD OK (labels off, compact off) ==="

echo "### test_deletion 10M: mark + consolidate, --compact should be skipped ###"
./build/bin/test_deletion -B "$DATA/bigann/bigann10M" -q "$DATA/bigann/bigann10kquery" \
  -g "$GT/GT_10M/bigann-10M" -t uint8 -d l2 -n 64 --dim 128 -k 10 -b 64 \
  --delete_fraction 0.05 --rounds 4 --compact 2>&1 | grep -iE "round|compact|summary|PASS|FAIL"

echo "### bench_all 10M (compact line should say disabled) ###"
./build/bin/bench_all -b "$DATA/bigann/bigann10M" -q "$DATA/bigann/bigann10kquery" \
  -g "$GT/GT_10M/bigann-10M" -t uint8 --dim 128 -k 10 --beam_width 64 -f 0.05 -a 100000 \
  2>&1 | grep -iE "INSERT|QUERY|DELETE|APPEND"

echo "### FFI build check (labels off) ###"
TVM_CMAKE=$(python -m tvm_ffi.config --cmakedir 2>/dev/null)
cmake -B build_ffi -DJASPER_BUILD_CMD=OFF -DJASPER_BUILD_TESTS=OFF -DJASPER_BUILD_FFI=ON \
  -DThrust_DIR="$THRUST_DIR" -DCMAKE_PREFIX_PATH="$CONDA_PREFIX;$TVM_CMAKE" >/dev/null
cmake --build build_ffi --target jasper_ffi -j8 >/dev/null && echo "=== FFI BUILD OK (labels off) ==="
echo "=== done ==="
