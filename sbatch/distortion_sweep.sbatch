#!/bin/bash
#SBATCH --partition=SaltSystemsLab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --nodelist=d4067
#SBATCH --mem=987842M
#SBATCH --gres=gpu:1
#SBATCH --job-name=distortion
#SBATCH --output=distortion_%j.log
#SBATCH --error=distortion_%j.err
#SBATCH --time=24:00:00
#
# Estimator accuracy data for every quantizer and k_ranks, via distortion_test.
#
#   estimators: lsh, pq, rabitq        k_ranks: 4, 8, 16
#
# One distortion_test run per (dataset, method, k) writes, under $OUT:
#
#   pops/<dataset>_<method>_k<k>.csv        recall@K vs nodes popped per query,
#                                           one row per BEAM_LIMITS pair
#   trajectory/<dataset>_<method>_k<k>.csv  edge scores for (query, popped node)
#                                           pairs sampled from real searches
#   random/<dataset>_<method>_k<k>.csv      edge scores for random node x random
#                                           query (the original Fig. 8 sampling)
#
# Edge-score CSVs: method,k_ranks,distance,sample,u,query,step,n_steps,v,exact,est
# with squared-L2 distances.
#
# After the sweep, the job draws every figure into $OUT/figures/ (PLOT=0 skips
# this; PLOT_ONLY=1 redraws from existing CSVs without touching the GPU):
#
#   recall_vs_pops.pdf                    plot_pops.py           <- pops/
#   random/distortion_<dataset>.pdf       plot_distortion.py     <- random/
#   trajectory/distortion_<dataset>.pdf   plot_distortion.py     <- trajectory/
#   random/neighborhood_ranking.pdf       neighborhood_ranking.py, all pops
#   trajectory/neighborhood_ranking.pdf   neighborhood_ranking.py, all pops
#   trajectory_late/neighborhood_ranking.pdf   ... only pops with
#                                              step >= LATE_FRAC * n_steps
#
# Every cell of a dataset runs on the SAME plain index (codes are built at
# load). The random samples are identical across cells; trajectory samples come
# from each method's own searches. Indexes are kept in $WORK, so a rerun skips
# the build. distortion_test compiles L2 configs only.
#
#   DRY_RUN=1 ./sbatch/distortion_sweep.sbatch
#   sbatch sbatch/distortion_sweep.sbatch
#   DATASETS="bigann10M openai" METHODS="rabitq" sbatch sbatch/distortion_sweep.sbatch
#   PLOT_ONLY=1 ./sbatch/distortion_sweep.sbatch       # redraw figures, no GPU
#   INDEX_openai=/path/to/plain.bin sbatch sbatch/distortion_sweep.sbatch

set -euo pipefail

# ── Tunables ───────────────────────────────────────────────────────────────
DATASETS="${DATASETS:-bigann10M deep10M gist openai sra50}"
METHODS="${METHODS:-lsh pq rabitq}"
K_LIST="${K_LIST:-4 8 16}"
SAMPLES="${SAMPLES:-10000}"
K="${K:-10}"
BEAM_LIMITS="${BEAM_LIMITS:-8:64 16:128 32:128 64:256 128:256 256:512 512:1024}"
TRAJ_BEAM="${TRAJ_BEAM:-64:256}"
MODES="${MODES:-pops trajectory random}"   # any subset
SEED="${SEED:-7}"
R_INT="${R_INT:-64}"
R_FIN="${R_FIN:-64}"           # dim > 128 configs are compiled only at R=64
N_PARTS="${N_PARTS:-1}"        # every dataset here fits on the card in one part
N_DUP="${N_DUP:-2}"
ALPHA="${ALPHA:-1.2}"
WORKSPACE_BUDGET="${WORKSPACE_BUDGET:-10GB}"
DRY_RUN="${DRY_RUN:-0}"
PLOT="${PLOT:-1}"
PLOT_ONLY="${PLOT_ONLY:-0}"
LATE_FRAC="${LATE_FRAC:-0.5}"       # "late" = second half of each search
FIG_FORMAT="${FIG_FORMAT:-pdf}"

ANN_DATA="${ANN_DATA:-/projects/SaltSystemsLab/ann_data}"
ANN_GT="${ANN_GT:-/projects/SaltSystemsLab/ann_gt}"
LOCALE="${LOCALE:-/tmp2/ann_data/locale}"
REPO="${REPO:-$HOME/jasper}"
WORK="${WORK:-/projects/SaltSystemsLab/${USER:-$(id -un)}/distortion}"
OUT="${OUT:-$REPO/distortion_data}"
LOGDIR="${LOGDIR:-$OUT/logs}"

# name -> base|queries|gt|dtype|dim   (same files as scripts/ci/bench.py)
dataset_spec() {
  case "$1" in
    bigann10M) echo "$ANN_DATA/bigann/bigann10M|$ANN_DATA/bigann/bigann10kquery|$ANN_GT/GT_10M/bigann-10M|uint8|128";;
    deep10M)   echo "$ANN_DATA/deep/deep10M|$ANN_DATA/deep/deep10kquery|$ANN_GT/GT_10M/deep-10M|float|96";;
    gist)      echo "$ANN_DATA/gist/gist_base.bin|$ANN_DATA/gist/gist_query.bin|$ANN_GT/gist-gt|float|960";;
    openai)    echo "$ANN_DATA/openai/openai_base.bin|$ANN_DATA/openai/openai_query.bin|$ANN_GT/openai-gt|float|1536";;
    sra50)     echo "$LOCALE/sra50/base.fbin|$LOCALE/sra50/query_mut0.00.fbin|$LOCALE/sra50/gt_mut0.00.ibin|float|768";;
    *)         return 1;;
  esac
}

valid_method() { case "$1" in lsh|pq|rabitq) return 0;; *) return 1;; esac; }

stamp() { date '+%Y-%m-%d %H:%M:%S'; }

# Figures. A failed plot is reported but never fails the job: the CSVs are the
# expensive part and are already on disk.
make_figures() {
  local fig="$OUT/figures" py="python3 $REPO/scripts"
  echo "=== figures -> $fig ==="
  mkdir -p "$fig"
  has_csv() { compgen -G "$1/*.csv" >/dev/null; }
  plot() { echo "  $*"; "$@" >>"$LOGDIR/plots.log" 2>&1 || echo "    PLOT FAILED — see $LOGDIR/plots.log"; }

  has_csv "$OUT/pops" && \
    plot $py/plot_pops.py "$OUT/pops" --out "$fig" --format "$FIG_FORMAT"
  for src in random trajectory; do
    has_csv "$OUT/$src" || continue
    plot $py/plot_distortion.py "$OUT/$src" --out "$fig/$src" --format "$FIG_FORMAT"
    plot $py/neighborhood_ranking.py "$OUT/$src" --out "$fig/$src" --format "$FIG_FORMAT"
  done
  has_csv "$OUT/trajectory" && \
    plot $py/neighborhood_ranking.py "$OUT/trajectory" --min-step-frac "$LATE_FRAC" \
         --out "$fig/trajectory_late" --format "$FIG_FORMAT"
  return 0
}

# ── Plan ───────────────────────────────────────────────────────────────────
echo "=== estimator distortion sweep ==="
echo "  datasets: $DATASETS"
echo "  methods:  $METHODS    k_ranks: $K_LIST"
echo "  modes:    $MODES"
echo "  samples:  $SAMPLES neighborhoods/cell, seed $SEED, trajectories at $TRAJ_BEAM"
echo "  pops:     k=$K at $BEAM_LIMITS"
echo "  graph:    R_int=$R_INT R_fin=$R_FIN dist=l2"
echo "  out:      $OUT"
echo
for ds in $DATASETS; do
  spec=$(dataset_spec "$ds") || { echo "unknown dataset: $ds" >&2; exit 2; }
  IFS='|' read -r base q gt dtype dim <<<"$spec"
  padded=$(( (dim + 7) / 8 * 8 ))
  ivar="INDEX_$ds"; idx="${!ivar:-$WORK/${ds}_r${R_INT}_${R_FIN}_plain.bin}"
  printf "  %-10s dim=%-5s %-6s index=%s%s\n" "$ds" "$dim" "$dtype" "$idx" \
    "$([ -s "$idx" ] && echo ' (exists)' || echo ' (will build)')"
  [ -r "$base" ] || echo "    WARNING: base missing: $base"
  [ -r "$q" ]    || echo "    WARNING: queries missing: $q"
  case " $MODES " in *" pops "*) [ -r "$gt" ] || echo "    WARNING: no ground truth at $gt (pops will fail)";; esac
  for m in $METHODS; do
    valid_method "$m" || { echo "unknown method: $m" >&2; exit 2; }
    for k in $K_LIST; do
      if [ "$m" = "pq" ] && [ $(( padded % k )) -ne 0 ]; then
        echo "    $m k$k: SKIP (padded_dim=$padded not divisible by $k)"
      elif [ "$m" = "rabitq" ] && [ $(( 8 * k )) -gt "$padded" ]; then
        echo "    $m k$k: note — rq_dim=$(( 8 * k )) > dim, codes all $padded dims (no projection)"
      fi
    done
  done
  if [ "$dim" -gt 128 ] && [ "$R_FIN" != "64" ]; then
    echo "    ERROR: dim > 128 needs R_FIN=64 (uint16-packed configs)" >&2; exit 2
  fi
done
echo

if [ "$DRY_RUN" != "0" ]; then echo "DRY_RUN set — nothing run."; exit 0; fi

# ── Environment + build ────────────────────────────────────────────────────
module load cuda || true
module load cmake || true
cd "$REPO"
module unload cuda || true
module unload cmake || true
set +u
source ~/.bashrc
conda activate /projects/SaltSystemsLab/zikun/env
set -u
mkdir -p "$LOGDIR"

if [ "$PLOT_ONLY" != "0" ]; then
  make_figures
  exit 0
fi
THRUST_DIR="${THRUST_DIR:-/projects/SaltSystemsLab/zikun/env/lib/rapids/cmake/thrust}"

echo "=== building binaries ==="
cmake -B build -DJASPER_BUILD_CMD=ON -DJASPER_BUILD_TESTS=OFF -DJASPER_BUILD_FFI=OFF \
  -DThrust_DIR="$THRUST_DIR" -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" >/dev/null
cmake --build build --target create_diskann_index distortion_test -j8 >/dev/null
echo "  ok"
echo

mkdir -p "$WORK" "$LOGDIR" "$OUT/pops" "$OUT/trajectory" "$OUT/random"

# ── Sweep ──────────────────────────────────────────────────────────────────
for ds in $DATASETS; do
  IFS='|' read -r base q gt dtype dim <<<"$(dataset_spec "$ds")"
  padded=$(( (dim + 7) / 8 * 8 ))
  ivar="INDEX_$ds"; idx="${!ivar:-$WORK/${ds}_r${R_INT}_${R_FIN}_plain.bin}"

  echo "######## $ds ########"
  if [ -s "$idx" ]; then
    echo "[$(stamp)] reusing index $idx"
  else
    echo "[$(stamp)] building plain index -> $idx"
    ./build/bin/create_diskann_index \
        -f "$base" -t "$dtype" -d l2 --dim "$dim" \
        --r_intermediate "$R_INT" --r_final "$R_FIN" \
        --n_parts "$N_PARTS" --n_dup "$N_DUP" -a "$ALPHA" \
        -w "$WORKSPACE_BUDGET" -i "$idx" >| "$LOGDIR/${ds}_build.log" 2>&1 \
      || { echo "  BUILD FAILED (see $LOGDIR/${ds}_build.log) — skipping $ds"; continue; }
  fi

  for m in $METHODS; do
    for k in $K_LIST; do
      [ "$m" = "pq" ] && [ $(( padded % k )) -ne 0 ] && continue
      cell="${ds}_${m}_k${k}"
      log="$LOGDIR/${cell}.log"
      mode_args=()
      for mode in $MODES; do
        case "$mode" in
          pops)       mode_args+=(--pops "$OUT/pops/${cell}.csv" -g "$gt" -b $BEAM_LIMITS);;
          trajectory) mode_args+=(--trajectory "$OUT/trajectory/${cell}.csv" --traj_beam "$TRAJ_BEAM");;
          random)     mode_args+=(--random "$OUT/random/${cell}.csv");;
          *) echo "unknown mode: $mode" >&2; exit 2;;
        esac
      done
      echo "[$(stamp)] $ds $m k$k"
      ./build/bin/distortion_test -i "$idx" --dim "$dim" -t "$dtype" -n "$R_FIN" \
          -m "$m" -r "$k" -q "$q" -k "$K" --samples "$SAMPLES" --seed "$SEED" \
          "${mode_args[@]}" >| "$log" 2>&1 \
        || { echo "  CELL FAILED ($ds $m k$k) — see $log"; continue; }
      grep -E "recall@|wrote" "$log" | sed 's/^/  /' || true
    done
  done
  echo
done

[ "$PLOT" != "0" ] && make_figures

echo "=== done: $(stamp) ==="
echo "CSVs in $OUT/{pops,trajectory,random}/<dataset>_<method>_k<k>.csv"
[ "$PLOT" != "0" ] && echo "figures in $OUT/figures/"
exit 0
