#!/bin/bash
#SBATCH --partition=gpu-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:h200
#SBATCH --job-name=bench_100m_cur
#SBATCH --output=bench_100m_cur_%j.log
#SBATCH --error=bench_100m_cur_%j.err
#SBATCH --time=02:00:00

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
DATA=/projects/SaltSystemsLab/ann_data
GT=/projects/SaltSystemsLab/ann_gt

set -e
cmake -B build -DJASPER_BUILD_CMD=ON -DJASPER_BUILD_TESTS=OFF -DJASPER_BUILD_FFI=OFF \
  -DThrust_DIR="$THRUST_DIR" -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" >/dev/null
cmake --build build --target bench_all -j8 >/dev/null

echo "############### bigann100M — CURRENT (labeled) ###############"
./build/bin/bench_all -b "$DATA/bigann/bigann100M" -q "$DATA/bigann/bigann10kquery" \
  -g "$GT/GT_100M/bigann-100M" -t uint8 --dim 128 -k 10 --beam_width 64 -f 0.05 -a 100000 \
  2>&1 | grep -iE "INSERT|QUERY|DELETE|APPEND|construct.*total"
echo "=== done ==="
