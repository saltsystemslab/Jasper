#!/bin/bash
#SBATCH --partition=SaltSystemsLab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --nodelist=d4067
#SBATCH --mem=987842M
#SBATCH --gres=gpu:1
#SBATCH --job-name=profile_search
#SBATCH --output=profile_search_%j.log
#SBATCH --error=profile_search_%j.err
#SBATCH --time=08:00:00
#
# Where a search's time goes, and how its pops are distributed over the graph:
# the two measurements that pick the next optimization.
#
# ── phases ─────────────────────────────────────────────────────────────────
# run_query built with -DJASPER_PROFILE_CLOCKS=ON (and the 4096 list width):
# thread 0 of every 64th query's block adds clock64() cycles per kernel phase,
# so every timed search prints cycles per pop for
#
#   pop        pick the next candidate (+ adaptive test, visited mark)
#   exact      directional: read u's edges, codes and vector, exact d(q, u)
#   frontier   insert u into the top-k
#   expand     append u's neighbors to the list
#   estimate   neighbor distances (estimates; exact reads for conventional)
#   unique     directional step 6b: drop copies of listed nodes
#   sort       merge (directional) / full sort (conventional) of the list
#   dedup      dedup_results
#   clip       clip_k
#
# at each PHASE_WIDTHS beam width (no re-rank, beam rule), for each method
# and placement. Cycles include stalls: on host the PCIe round trip lands in
# "exact" for directional search and in "expand"/"estimate" for conventional.
# Each point runs PHASE_PASSES times and the summary keeps the last, so the
# first pass's cold start is not in it. A profile build adds a barrier per
# phase, so its QPS is not the benchmark's; the shares are what to read.
#
# ── visits ─────────────────────────────────────────────────────────────────
# distortion_test --pops --visit_stats on the same index: per VISIT_WIDTHS
# point, the share of pops that a GPU-resident cache of the 0.1 ... 20% most
# popped nodes would serve (ranked on even queries, scored on odd ones), and
# the pops spent before a query first expands one of its true neighbors.
# Visits do not depend on placement, so this runs on device only.
#
# Outputs, under $OUT:
#   raw/<ds>_<method>_top<k>_<placement>.phase.log   run_query output
#   raw/<ds>_<method>_top<k>.visit.csv               --visit_stats
#   raw/<ds>_<method>_top<k>.pops.csv / .pops.log    the --pops run behind it
#   phases.csv, phases.txt     every profiled point, and the per-phase table
#   visits.txt                 the visit tables
#   (scripts/profile_summary.py)
#
# Indexes: the L192 two-pass plain indexes bench_adaptive_termination uses
# ($INDEX_DIR/<ds>_r64_64_L192_2pass_plain.bin); INDEX_<ds> overrides.
#
#   DRY_RUN=1 ./sbatch/profile_search.sbatch
#   sbatch sbatch/profile_search.sbatch
#   STEPS=visits METHODS=lsh sbatch sbatch/profile_search.sbatch
#   SUMMARY_ONLY=1 ./sbatch/profile_search.sbatch

set -euo pipefail

# ── Tunables ───────────────────────────────────────────────────────────────
DATASETS="${DATASETS:-deep10M sra50}"
METHODS="${METHODS:-conventional lsh rabitq}"
K_RANKS="${K_RANKS:-16}"
K="${K:-10 100}"
PLACEMENTS="${PLACEMENTS:-device host}"
STEPS="${STEPS:-phases visits}"
PHASE_WIDTHS="${PHASE_WIDTHS:-63 191 447 959 1983 4031}"  # tier tops; 4031 lsh/rabitq only
VISIT_WIDTHS="${VISIT_WIDTHS:-63 191 447 959 1983 4031}"
PHASE_PASSES="${PHASE_PASSES:-2}"
LIMIT="${LIMIT:-8192}"
QUERY_CHUNK_host_conventional="${QUERY_CHUNK_host_conventional:-1000}"  # its best, see bench_adaptive_termination
R_INT="${R_INT:-64}"
R_FIN="${R_FIN:-64}"
BUILD_TAG="${BUILD_TAG:-L192_2pass}"
BUILD_DIR="${BUILD_DIR:-build_prof}"
DRY_RUN="${DRY_RUN:-0}"
SUMMARY_ONLY="${SUMMARY_ONLY:-0}"

ANN_DATA="${ANN_DATA:-/projects/SaltSystemsLab/ann_data}"
ANN_GT="${ANN_GT:-/projects/SaltSystemsLab/ann_gt}"
LOCALE="${LOCALE:-/tmp2/ann_data/locale}"
REPO="${REPO:-$HOME/jasper}"
INDEX_DIR="${INDEX_DIR:-/scratch/${USER:-$(id -un)}}"
OUT="${OUT:-$REPO/profile_search}"
RAW="$OUT/raw"

# name -> base|queries|gt|dtype|dim   (same table as bench_adaptive_termination)
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
stamp() { date '+%Y-%m-%d %H:%M:%S'; }
index_of() { local ivar="INDEX_$1"; echo "${!ivar:-$INDEX_DIR/${1}_r${R_INT}_${R_FIN}_${BUILD_TAG}_plain.bin}"; }

summarize() {
  python3 -I "$REPO/scripts/profile_summary.py" "$RAW" --out "$OUT" \
    || echo "  SUMMARY FAILED"
}
if [ "$SUMMARY_ONLY" != "0" ]; then summarize; exit 0; fi

# Widths a method can run: list + 64 < 4096 for lsh / rabitq (the wide build),
# < 2048 for conventional; and at least k.
widths_for() {                         # $1 = method, $2 = k, $3 = width list
  local max w out=""
  max=$([ "$1" = conventional ] && echo 2048 || echo 4096)
  for w in $3; do
    if [ "$w" -ge "$2" ] && [ $(( w + 64 )) -lt "$max" ]; then out="$out $w"; fi
  done
  echo "${out# }"
}

# ── Plan ───────────────────────────────────────────────────────────────────
echo "=== search profile: phases + visit skew ==="
echo "  host:       $(hostname)"
echo "  datasets:   $DATASETS"
echo "  methods:    $METHODS    k_ranks: $K_RANKS    k: $K"
echo "  steps:      $STEPS"
echo "  phases:     widths $PHASE_WIDTHS x $PHASE_PASSES passes, placements $PLACEMENTS"
echo "  visits:     widths $VISIT_WIDTHS (device)"
echo "  binaries:   $BUILD_DIR (JASPER_PROFILE_CLOCKS, JASPER_WIDE_TIER)"
echo "  out:        $OUT"
for ds in $DATASETS; do
  spec=$(dataset_spec "$ds") || { echo "unknown dataset: $ds" >&2; exit 2; }
  IFS='|' read -r base q gt dtype dim <<<"$spec"
  idx=$(index_of "$ds")
  printf "  %-10s dim=%-5s index=%s%s\n" "$ds" "$dim" "$idx" "$([ -s "$idx" ] && echo '' || echo ' (MISSING)')"
done
for m in $METHODS; do
  case "$m" in conventional|lsh|rabitq) ;; *) echo "unknown method: $m (conventional | lsh | rabitq)" >&2; exit 2;; esac
done
if [ "$DRY_RUN" != "0" ]; then echo; echo "DRY_RUN set — nothing run."; exit 0; fi

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
THRUST_DIR="${THRUST_DIR:-/projects/SaltSystemsLab/zikun/env/lib/rapids/cmake/thrust}"

echo
echo "=== building binaries ==="
cmake -B "$BUILD_DIR" -DJASPER_BUILD_CMD=ON -DJASPER_BUILD_TESTS=OFF -DJASPER_BUILD_FFI=OFF \
  -DJASPER_WIDE_TIER=ON -DJASPER_PROFILE_CLOCKS=ON \
  -DThrust_DIR="$THRUST_DIR" -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" >/dev/null
cmake --build "$BUILD_DIR" --target run_query distortion_test -j8 >/dev/null
echo "  ok"
mkdir -p "$RAW"

# ── Runs ───────────────────────────────────────────────────────────────────
for ds in $DATASETS; do
  IFS='|' read -r base q gt dtype dim <<<"$(dataset_spec "$ds")"
  idx=$(index_of "$ds")
  [ -s "$idx" ] || { echo "[$(stamp)] $ds: no index at $idx — skipping"; continue; }
  for m in $METHODS; do
    if [ "$m" = conventional ]; then qps_mode=(); pops_k=4
    else qps_mode=("--$m" --k_ranks "$K_RANKS"); pops_k=$K_RANKS; fi
    for topk in $K; do
      case " $STEPS " in *" phases "*)
        ws=$(widths_for "$m" "$topk" "$PHASE_WIDTHS")
        pts=""; for w in $ws; do pts="$pts $w:$LIMIT"; done
        all=""; for _ in $(seq "$PHASE_PASSES"); do all="$all $pts"; done
        for place in $PLACEMENTS; do
          host_arg=(); [ "$place" = host ] && host_arg=(--on_host)
          cvar="QUERY_CHUNK_${place}_${m}"; chunk="${!cvar:-0}"
          name="${ds}_${m}_top${topk}_${place}"
          echo "[$(stamp)] phases $ds $m ${topk}@${topk} on $place (widths $ws)"
          "$BUILD_DIR"/bin/run_query -i "$idx" --dim "$dim" -d l2 -t "$dtype" -n "$R_FIN" -k "$topk" \
              ${qps_mode[@]+"${qps_mode[@]}"} ${host_arg[@]+"${host_arg[@]}"} \
              --query_chunk "$chunk" -q "$q" -g "$gt" --beam_limits $all \
              >| "$RAW/$name.phase.log" 2>&1 \
            || echo "  FAILED ($name) — see $RAW/$name.phase.log"
          echo "  $(grep -c '^phase_profile:' "$RAW/$name.phase.log" || true) profiled searches"
        done;;
      esac
      case " $STEPS " in *" visits "*)
        ws=$(widths_for "$m" "$topk" "$VISIT_WIDTHS")
        pts=""; for w in $ws; do pts="$pts $w:$LIMIT"; done
        name="${ds}_${m}_top${topk}"
        echo "[$(stamp)] visits $ds $m ${topk}@${topk} (widths $ws)"
        "$BUILD_DIR"/bin/distortion_test -i "$idx" --dim "$dim" -t "$dtype" -n "$R_FIN" -k "$topk" \
            -m "$m" -r "$pops_k" -q "$q" -g "$gt" \
            --pops "$RAW/$name.pops.csv" --visit_stats "$RAW/$name.visit.csv" -b $pts \
            >| "$RAW/$name.pops.log" 2>&1 \
          || echo "  FAILED ($name) — see $RAW/$name.pops.log"
        ;;
      esac
    done
  done
done

summarize
echo
echo "=== done: $(stamp) ==="
