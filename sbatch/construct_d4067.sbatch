#!/bin/bash
#SBATCH --partition=SaltSystemsLab
#SBATCH --nodelist=d4067
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:rtx6000:1
#SBATCH --job-name=construct_d4067
#SBATCH --output=construct_d4067_%j.log
#SBATCH --error=construct_d4067_%j.err
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
ulimit -c 0   # never dump multi-GB cores
THRUST_DIR=/projects/SaltSystemsLab/zikun/env/lib/rapids/cmake/thrust
DATA=/projects/SaltSystemsLab/ann_data
GT=/projects/SaltSystemsLab/ann_gt

echo "### GPU on this node ###"
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader || true

set -e
cmake -B build -DJASPER_BUILD_CMD=ON -DJASPER_BUILD_TESTS=OFF -DJASPER_BUILD_FFI=OFF \
  -DThrust_DIR="$THRUST_DIR" -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" >/dev/null
cmake --build build --target bench_all -j8

echo "### bigann10M construction (dedicated node) ###"
./build/bin/bench_all -b "$DATA/bigann/bigann10M" -q "$DATA/bigann/bigann10kquery" \
  -g "$GT/GT_10M/bigann-10M" -t uint8 --dim 128 -k 10 --beam_width 64 -f 0.05 -a 100000 \
  2>&1 | grep -iE "INSERT|QUERY|DELETE|APPEND|construct.*total"
echo "=== done ==="
