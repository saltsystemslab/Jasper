#!/bin/bash
#SBATCH --partition=SaltSystemsLab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --nodelist=d4067
#SBATCH --mem=987842M
#SBATCH --gres=gpu:1
#SBATCH --job-name=list_opts
#SBATCH --output=list_opts_%j.log
#SBATCH --error=list_opts_%j.err
#SBATCH --time=08:00:00
#
# The candidate-list maintenance variants of the directional and conventional
# kernels (list_ops.cuh), one list_opt bit each, against the reference (0):
#
#   1  LIST_PREFILTER    drop new entries ranked past a full list's last one;
#                        a pop that keeps none skips the rest of the list work
#   2  LIST_RANK_SORT    sort only the surviving new entries, by rank, not a
#                        CUB block sort over all 64 slots
#   4  LIST_FUSED_MERGE  merge + dedup + clip in one pass over the changed part
#
# MASKS lists the combinations to run (default every single bit, and 7 = all),
# each against 0. Three steps:
#
#   check    distortion_test --pops --check_list_opts <mask>: every point runs
#            with 0 and with the mask, and each query must pop the same nodes
#            in the same order, with the same exact distances and re-rank count,
#            and return the same top-k. Beam points at every compiled width's
#            top, plus adaptive (γ) and re-rank points, since the variants run
#            under every stop rule. Placement does not change a search, so
#            device only.
#   qps      run_query --list_opts 0,<masks>: every variant at each point, in
#            turn, in one process (one load, drift spread over all), PASSES
#            passes; device and host.
#   phases   the same in a profile build (JASPER_PROFILE_CLOCKS): cycles per
#            pop by phase per variant, plus how many new entries the prefilter
#            keeps per pop and how many pops it skips. Its QPS is not the
#            benchmark's.
#
# Outputs, under $OUT:
#   raw/<ds>_<method>_top<k>_m<mask>.check.log      list_check: lines
#   raw/<ds>_<method>_top<k>_<placement>.qps.log    every variant, every pass
#   raw/<ds>_<method>_top<k>_<placement>.phase.log  profile build
#   check.txt, qps.txt, phases.txt   (scripts/list_opts_summary.py)
#
#   DRY_RUN=1 ./sbatch/bench_list_opts.sbatch
#   sbatch sbatch/bench_list_opts.sbatch
#   STEPS=check MASKS="1 4" sbatch sbatch/bench_list_opts.sbatch
#   SUMMARY_ONLY=1 ./sbatch/bench_list_opts.sbatch

set -euo pipefail

# ── Tunables ───────────────────────────────────────────────────────────────
DATASETS="${DATASETS:-deep10M sra50}"
METHODS="${METHODS:-conventional lsh rabitq}"
K_RANKS="${K_RANKS:-16}"
K="${K:-10 100}"
MASKS="${MASKS:-1 2 4 7}"
STEPS="${STEPS:-check qps phases}"
PLACEMENTS="${PLACEMENTS:-device host}"
PHASE_PLACEMENTS="${PHASE_PLACEMENTS:-device host}"
WIDTHS="${WIDTHS:-63 191 447 959 1983 4031}"   # each compiled width's top
GAMMAS="${GAMMAS:-0.1}"                        # adaptive points (check only), at 447 and 1983
RERANK_MULT="${RERANK_MULT:-4}"                # re-rank points (check only): M = mult x k at 447 and 1983
PASSES="${PASSES:-3}"
QUERY_CHUNK_host_conventional="${QUERY_CHUNK_host_conventional:-1000}"  # its best on host (bench_adaptive_termination)
LIMIT="${LIMIT:-8192}"
R_INT="${R_INT:-64}"
R_FIN="${R_FIN:-64}"
BUILD_TAG="${BUILD_TAG:-L192_2pass}"
DRY_RUN="${DRY_RUN:-0}"
SUMMARY_ONLY="${SUMMARY_ONLY:-0}"

ANN_DATA="${ANN_DATA:-/projects/SaltSystemsLab/ann_data}"
ANN_GT="${ANN_GT:-/projects/SaltSystemsLab/ann_gt}"
LOCALE="${LOCALE:-/tmp2/ann_data/locale}"
REPO="${REPO:-$HOME/jasper}"
INDEX_DIR="${INDEX_DIR:-/scratch/${USER:-$(id -un)}}"
OUT="${OUT:-$REPO/list_opts}"
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
  python3 -I "$REPO/scripts/list_opts_summary.py" "$RAW" --out "$OUT" || echo "  SUMMARY FAILED"
}
if [ "$SUMMARY_ONLY" != "0" ]; then summarize; exit 0; fi

# Points for search k and method: beam at every width that fits (list + 64 <
# 4096 for lsh / rabitq in the wide build, < 2048 for conventional; width >=
# k). With extras (check step, directional only): γ points and re-rank points
# at 447 and 1983 too.
points_for() {                         # $1 = k, $2 = with_extras (0/1), $3 = method
  local k=$1 w g rr out="" max=4096
  [ "$3" = conventional ] && max=2048
  for w in $WIDTHS; do
    if [ "$w" -ge "$k" ] && [ $(( w + 64 )) -lt "$max" ]; then out="$out $w:$LIMIT"; fi
  done
  if [ "$2" = 1 ] && [ "$3" != conventional ]; then
    rr=$(( RERANK_MULT * k ))
    for w in 447 1983; do
      [ "$w" -ge "$k" ] || continue
      for g in $GAMMAS; do out="$out $w:$LIMIT:$g"; done
      if [ $(( w + rr + 64 )) -lt 4096 ]; then out="$out $w:$LIMIT:-:$rr"; fi
    done
  fi
  echo "${out# }"
}
ALL_OPTS="0$(for m in $MASKS; do printf ',%s' "$m"; done)"

# ── Plan ───────────────────────────────────────────────────────────────────
echo "=== candidate-list variants vs reference ==="
echo "  host:       $(hostname)"
echo "  datasets:   $DATASETS"
echo "  methods:    $METHODS    k_ranks: $K_RANKS    k: $K"
echo "  masks:      $MASKS  (vs 0; 1 prefilter, 2 rank sort, 4 fused merge)"
echo "  steps:      $STEPS"
for k in $K; do
  echo "  ${k}@${k} points:  qps/phases: $(points_for "$k" 0 lsh)  (conventional: up to 1983)"
  echo "               check: $(points_for "$k" 1 lsh)"
done
echo "  qps:        $PASSES passes, placements $PLACEMENTS    phases: $PHASE_PLACEMENTS"
echo "  out:        $OUT"
for ds in $DATASETS; do
  spec=$(dataset_spec "$ds") || { echo "unknown dataset: $ds" >&2; exit 2; }
  IFS='|' read -r base q gt dtype dim <<<"$spec"
  idx=$(index_of "$ds")
  printf "  %-10s dim=%-5s index=%s%s\n" "$ds" "$dim" "$idx" "$([ -s "$idx" ] && echo '' || echo ' (MISSING)')"
done
for m in $METHODS; do
  case "$m" in conventional|lsh|rabitq|pq) ;; *) echo "unknown method: $m" >&2; exit 2;; esac
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

build() {                              # $1 = dir, $2 = profile ON/OFF, targets...
  local dir=$1 prof=$2; shift 2
  cmake -B "$dir" -DJASPER_BUILD_CMD=ON -DJASPER_BUILD_TESTS=OFF -DJASPER_BUILD_FFI=OFF \
    -DJASPER_WIDE_TIER=ON -DJASPER_PROFILE_CLOCKS="$prof" \
    -DThrust_DIR="$THRUST_DIR" -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" >/dev/null
  cmake --build "$dir" --target "$@" -j8 >/dev/null
}
echo
echo "=== building binaries ==="
case " $STEPS " in *" check "*|*" qps "*) build build_wide OFF run_query distortion_test;; esac
case " $STEPS " in *" phases "*) build build_prof ON run_query;; esac
echo "  ok"
mkdir -p "$RAW"

# ── Runs ───────────────────────────────────────────────────────────────────
for ds in $DATASETS; do
  IFS='|' read -r base q gt dtype dim <<<"$(dataset_spec "$ds")"
  idx=$(index_of "$ds")
  [ -s "$idx" ] || { echo "[$(stamp)] $ds: no index at $idx — skipping"; continue; }
  for m in $METHODS; do
    # run_query picks the search by flag (conventional is the default path);
    # distortion_test needs a k_ranks to pick a config, which conventional ignores.
    if [ "$m" = conventional ]; then qps_mode=(); pops_k=4
    else qps_mode=("--$m" --k_ranks "$K_RANKS"); pops_k=$K_RANKS; fi
    for k in $K; do
      case " $STEPS " in *" check "*)
        pts=$(points_for "$k" 1 "$m")
        for mask in $MASKS; do
          name="${ds}_${m}_top${k}_m${mask}"
          echo "[$(stamp)] check $ds $m ${k}@${k} mask $mask"
          build_wide/bin/distortion_test -i "$idx" --dim "$dim" -t "$dtype" -n "$R_FIN" -k "$k" \
              -m "$m" -r "$pops_k" -q "$q" -g "$gt" \
              --pops "$RAW/$name.pops.csv" --check_list_opts "$mask" -b $pts \
              >| "$RAW/$name.check.log" 2>&1 \
            || echo "  FAILED ($name) — see $RAW/$name.check.log"
          echo "  $(grep -c 'result=identical' "$RAW/$name.check.log" || true) identical," \
               "$(grep -c 'result=DIFFERENT' "$RAW/$name.check.log" || true) different"
        done;;
      esac
      pts=$(points_for "$k" 0 "$m")
      all=""; for _ in $(seq "$PASSES"); do all="$all $pts"; done
      case " $STEPS " in *" qps "*)
        for place in $PLACEMENTS; do
          host_arg=(); [ "$place" = host ] && host_arg=(--on_host)
          cvar="QUERY_CHUNK_${place}_${m}"; chunk="${!cvar:-0}"
          name="${ds}_${m}_top${k}_${place}"
          echo "[$(stamp)] qps $ds $m ${k}@${k} on $place"
          build_wide/bin/run_query -i "$idx" --dim "$dim" -d l2 -t "$dtype" -n "$R_FIN" -k "$k" \
              ${qps_mode[@]+"${qps_mode[@]}"} ${host_arg[@]+"${host_arg[@]}"} \
              --query_chunk "$chunk" --list_opts "$ALL_OPTS" -q "$q" -g "$gt" --beam_limits $all \
              >| "$RAW/$name.qps.log" 2>&1 \
            || echo "  FAILED ($name) — see $RAW/$name.qps.log"
        done;;
      esac
      case " $STEPS " in *" phases "*)
        for place in $PHASE_PLACEMENTS; do
          host_arg=(); [ "$place" = host ] && host_arg=(--on_host)
          cvar="QUERY_CHUNK_${place}_${m}"; chunk="${!cvar:-0}"
          name="${ds}_${m}_top${k}_${place}"
          echo "[$(stamp)] phases $ds $m ${k}@${k} on $place"
          build_prof/bin/run_query -i "$idx" --dim "$dim" -d l2 -t "$dtype" -n "$R_FIN" -k "$k" \
              ${qps_mode[@]+"${qps_mode[@]}"} ${host_arg[@]+"${host_arg[@]}"} \
              --query_chunk "$chunk" --list_opts "$ALL_OPTS" -q "$q" -g "$gt" --beam_limits $pts $pts \
              >| "$RAW/$name.phase.log" 2>&1 \
            || echo "  FAILED ($name) — see $RAW/$name.phase.log"
        done;;
      esac
    done
  done
done

summarize
echo
echo "=== done: $(stamp) ==="
# Fail the job if any variant differed from the reference or a check did not
# run, so a dependent job (sbatch --dependency=afterok:<id>) does not start.
case " $STEPS " in *" check "*)
  bad=$(cat /dev/null "$RAW"/*.check.log 2>/dev/null | grep -c 'result=DIFFERENT' || true)
  failed=$( { grep -L 'list_check:' "$RAW"/*.check.log 2>/dev/null || true; } | wc -l | tr -d ' ')
  if [ "${bad:-0}" != 0 ] || [ "${failed:-0}" != 0 ]; then
    echo "CHECK FAILED: $bad point(s) differ, $failed check log(s) without results"
    exit 1
  fi
  echo "check: every variant identical to the reference";;
esac
