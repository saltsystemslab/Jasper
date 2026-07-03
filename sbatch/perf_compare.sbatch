#!/bin/bash
#SBATCH --partition=gpu-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:h200
#SBATCH --job-name=jasper_perf
#SBATCH --output=jasper_perf_%j.log
#SBATCH --error=jasper_perf_%j.err
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

THRUST_DIR=/projects/SaltSystemsLab/zikun/env/lib/rapids/cmake/thrust
BASE=/projects/SaltSystemsLab/ann_data/bigann/bigann10M
Q=/projects/SaltSystemsLab/ann_data/bigann/bigann10kquery
GT=/projects/SaltSystemsLab/ann_gt/GT_10M/bigann-10M
mkdir -p perf_tmp
nvidia-smi --query-gpu=name --format=csv,noheader || true

run_bench () {  # $1=label  $2=builddir
  cmake -B "$2" -DJASPER_BUILD_CMD=ON -DJASPER_BUILD_TESTS=OFF -DJASPER_BUILD_FFI=OFF \
    -DThrust_DIR="$THRUST_DIR" -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" >/dev/null 2>&1
  cmake --build "$2" --target create_index run_query -j8 >/dev/null 2>&1 || { echo "[$1] BUILD FAILED"; return 1; }
  echo "===================== [$1] CONSTRUCTION ====================="
  t0=$(date +%s.%N)
  "$2"/bin/create_index -f "$BASE" -t uint8 -d l2 -n 64 --dim 128 \
    -i "perf_tmp/idx_$1" -w 10GB 2>&1 | grep -iE "construct|ms total| total |saved|done" || true
  t1=$(date +%s.%N)
  echo "[$1] create_index WALL = $(echo "$t1 - $t0" | bc) s"
  echo "===================== [$1] QUERY ====================="
  "$2"/bin/run_query -i "perf_tmp/idx_$1" -q "$Q" -g "$GT" -t uint8 -d l2 \
    -n 64 --dim 128 -k 10 --beam_limits 32:64 64:128 128:256 2>&1 \
    | grep -iE "beam_width|recall|throughput|QPS" || true
}

set +e
git rev-parse HEAD >/dev/null 2>&1 && IS_GIT=1 || IS_GIT=0
echo "IS_GIT=$IS_GIT"

run_bench current build

if [ "$IS_GIT" = "1" ]; then
  # tracked-only stash: reverts modified sources to original; leaves untracked
  # new headers (inert, not referenced by original sources) and the job logs.
  git stash push -m perfbench >/dev/null 2>&1 && STASHED=1 || STASHED=0
  echo "STASHED=$STASHED"
  if [ "$STASHED" = "1" ]; then
    run_bench original build_orig
    git stash pop >/dev/null 2>&1 && echo "stash restored" || echo "STASH POP FAILED"
  fi
else
  echo "remote is not a git checkout; original not benchmarked here"
fi
echo "=== perf compare done ==="
