#!/bin/bash
#SBATCH --partition=gpu-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:h200
#SBATCH --job-name=jasper_buildchk
#SBATCH --output=jasper_buildchk_%j.log
#SBATCH --error=jasper_buildchk_%j.err
#SBATCH --time=00:40:00

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

THRUST_DIR=/projects/SaltSystemsLab/zikun/env/lib/rapids/cmake/thrust

# (1) Header compile check via test_deletion (FFI off) — must pass.
set -e
cmake -B build -DJASPER_BUILD_CMD=ON -DJASPER_BUILD_TESTS=OFF -DJASPER_BUILD_FFI=OFF \
  -DThrust_DIR="$THRUST_DIR" -DCMAKE_PREFIX_PATH="$CONDA_PREFIX"
cmake --build build --target test_deletion -j8
echo "=== HEADERS OK (test_deletion built) ==="

# (2) FFI build attempt with conda libstdc++ on the loader path (best effort).
set +e
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
python -c "import tvm_ffi; print('tvm_ffi import OK')" || echo "tvm_ffi import still failing"
TVM_CMAKE=$(python -m tvm_ffi.config --cmakedir 2>/dev/null)
echo "tvm_ffi cmakedir=$TVM_CMAKE"
cmake -B build_ffi -DJASPER_BUILD_CMD=OFF -DJASPER_BUILD_TESTS=OFF -DJASPER_BUILD_FFI=ON \
  -DThrust_DIR="$THRUST_DIR" -DCMAKE_PREFIX_PATH="$CONDA_PREFIX;$TVM_CMAKE"
cmake --build build_ffi --target jasper_ffi -j8 && echo "=== jasper_ffi OK ===" || echo "=== jasper_ffi build FAILED ==="
