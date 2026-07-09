#!/bin/bash
#SBATCH --partition=gpu-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:h200
#SBATCH --job-name=jasper_idmap
#SBATCH --output=jasper_idmap_%j.log
#SBATCH --error=jasper_idmap_%j.err
#SBATCH --time=00:30:00

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
set -e
cmake -B build -DJASPER_BUILD_CMD=ON -DJASPER_BUILD_TESTS=OFF -DJASPER_BUILD_FFI=OFF \
  -DThrust_DIR="$THRUST_DIR" -DCMAKE_PREFIX_PATH="$CONDA_PREFIX"
cmake --build build --target test_idmap -j8
echo "=== BUILD OK ==="
./build/bin/test_idmap
