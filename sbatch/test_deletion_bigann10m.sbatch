#!/bin/bash
#SBATCH --partition=gpu-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:h200
#SBATCH --job-name=jasper_del_10m
#SBATCH --output=jasper_del_10m_%j.log
#SBATCH --error=jasper_del_10m_%j.err
#SBATCH --time=00:50:00

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
  -DThrust_DIR="$THRUST_DIR" -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" >/dev/null
cmake --build build --target test_deletion -j8
echo "=== BUILD OK ==="

BASE=/projects/SaltSystemsLab/ann_data/bigann/bigann10M
Q=/projects/SaltSystemsLab/ann_data/bigann/bigann10kquery
GT=/projects/SaltSystemsLab/ann_gt/GT_10M/bigann-10M

./build/bin/test_deletion -B "$BASE" -q "$Q" -g "$GT" \
  -t uint8 -d l2 -n 64 --dim 128 -k 10 -b 64 \
  --delete_fraction 0.05 --rounds 4 --compact
