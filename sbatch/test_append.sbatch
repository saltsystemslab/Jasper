#!/bin/bash
#SBATCH --partition=gpu-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:h200
#SBATCH --job-name=jasper_append
#SBATCH --output=jasper_append_%j.log
#SBATCH --error=jasper_append_%j.err
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
  -DThrust_DIR="$THRUST_DIR" -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" >/dev/null
cmake --build build --target test_append -j8
echo "=== BUILD OK ==="

# N=1M construct, then 8 rounds x 100k appends = 800k (crosses initial 1.5M
# table capacity around round 6 -> exercises incremental upsert AND resize).
./build/bin/test_append -b /projects/SaltSystemsLab/ann_data/bigann/bigann10M \
  -t uint8 --dim 128 -n 1000000 -a 100000 --rounds 8 -k 10 --beam_width 64
