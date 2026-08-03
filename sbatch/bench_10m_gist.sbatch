#!/bin/bash
#SBATCH --partition=gpu-interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:h200
#SBATCH --job-name=bench_10m_gist
#SBATCH --output=bench_10m_gist_%j.log
#SBATCH --error=bench_10m_gist_%j.err
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
mkdir -p perf_tmp

bench_current () {  # label base q gt dtype dim
  echo ""; echo "############### $1 — CURRENT (labeled) ###############"
  ./build/bin/bench_all -b "$2" -q "$3" -g "$4" -t "$5" --dim "$6" \
    -k 10 --beam_width 64 -f 0.05 -a 100000 2>&1 \
    | grep -iE "INSERT|QUERY|DELETE|APPEND|construct.*total"
}
bench_original () {  # label base q gt dtype dim idx
  echo ""; echo "############### $1 — ORIGINAL (non-labeled) ###############"
  ./build_orig/bin/create_index -f "$2" -t "$5" -d l2 -n 64 --dim "$6" -i "$7" -w 10GB 2>&1 \
    | grep -iE "construct.*total"
  ./build_orig/bin/run_query -i "$7" -q "$3" -g "$4" -t "$5" -d l2 -n 64 --dim "$6" \
    -k 10 --beam_limits 64:128 2>&1 | grep -iE "beam_width|recall|throughput"
}

set -e
cmake -B build -DJASPER_BUILD_CMD=ON -DJASPER_BUILD_TESTS=OFF -DJASPER_BUILD_FFI=OFF \
  -DThrust_DIR="$THRUST_DIR" -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" >/dev/null
cmake --build build --target bench_all create_index run_query -j8 >/dev/null

bench_current bigann10M "$DATA/bigann/bigann10M" "$DATA/bigann/bigann10kquery" "$GT/GT_10M/bigann-10M" uint8 128
bench_current GIST      "$DATA/gist/gist_base.bin" "$DATA/gist/gist_query.bin" "$GT/gist-gt" float 960

set +e
git stash push -m bench >/dev/null 2>&1 && ST=1 || ST=0
echo ""; echo "STASHED=$ST"
if [ "$ST" = "1" ]; then
  set -e
  cmake -B build_orig -DJASPER_BUILD_CMD=ON -DJASPER_BUILD_TESTS=OFF -DJASPER_BUILD_FFI=OFF \
    -DThrust_DIR="$THRUST_DIR" -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" >/dev/null
  cmake --build build_orig --target create_index run_query -j8 >/dev/null
  bench_original bigann10M "$DATA/bigann/bigann10M" "$DATA/bigann/bigann10kquery" "$GT/GT_10M/bigann-10M" uint8 128 perf_tmp/orig10
  bench_original GIST      "$DATA/gist/gist_base.bin" "$DATA/gist/gist_query.bin" "$GT/gist-gt" float 960 perf_tmp/origgist
  git stash pop >/dev/null 2>&1 && echo "stash restored" || echo "STASH POP FAILED"
fi
echo "=== done ==="
