#!/bin/bash
#SBATCH --partition=gpu-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:h200
#SBATCH --job-name=jasper_feat
#SBATCH --output=jasper_feat_%j.log
#SBATCH --error=jasper_feat_%j.err
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
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

THRUST_DIR=/projects/SaltSystemsLab/zikun/env/lib/rapids/cmake/thrust
DATA=/projects/SaltSystemsLab/ann_data
GT=/projects/SaltSystemsLab/ann_gt

set -e
# (1) cmd test_append: confirm resize now triggers at 90% load (expect the
#     table_cap to grow around round 3: n=1.4M > 0.9*1.501M).
cmake -B build -DJASPER_BUILD_CMD=ON -DJASPER_BUILD_TESTS=OFF -DJASPER_BUILD_FFI=OFF \
  -DThrust_DIR="$THRUST_DIR" -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" >/dev/null
cmake --build build --target test_append -j8 >/dev/null
echo "=== test_append (90% resize check) ==="
./build/bin/test_append -b "$DATA/bigann/bigann10M" -t uint8 --dim 128 \
  -n 1000000 -a 100000 --rounds 8 -k 10 --beam_width 64 2>&1 | grep -iE "append r|self-recall|PASS|FAIL"

# (2) Build + install the FFI, then run the Python feature test.
TVM_CMAKE=$(python -m tvm_ffi.config --cmakedir)
cmake -B build_ffi -DJASPER_BUILD_CMD=OFF -DJASPER_BUILD_TESTS=OFF -DJASPER_BUILD_FFI=ON \
  -DThrust_DIR="$THRUST_DIR" -DCMAKE_PREFIX_PATH="$CONDA_PREFIX;$TVM_CMAKE" >/dev/null
cmake --build build_ffi --target jasper_ffi -j8 >/dev/null
cmake --install build_ffi >/dev/null
echo "=== jasper_ffi installed to python/jasper/lib ==="
ls -la python/jasper/lib/

echo "=== python feature test on bigann10M ==="
PYTHONPATH=python python example/features_bigann10m.py \
  --base "$DATA/bigann/bigann10M" \
  --queries "$DATA/bigann/bigann10kquery" \
  --gt "$GT/GT_10M/bigann-10M" \
  --k 10 --beam_width 64 --n_delete 500000 --n_append 50000
