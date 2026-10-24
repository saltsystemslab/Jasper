#!/bin/bash
#SBATCH --partition=SaltSystemsLab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --nodelist=d4067
#SBATCH --mem=400GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=graph_build
#SBATCH --output=graph_build_%j.log
#SBATCH --error=graph_build_%j.err
#SBATCH --time=12:00:00
#
# Does a better graph build help? Builds each dataset's plain index under each
# VARIANTS entry and measures recall against nodes expanded per query
# (distortion_test --pops, device only) for conventional search and for
# directional search, at recall 10@10 and 100@100 over one width grid.
# Recall per expansion is the graph-quality measure: it does not depend on
# placement or on kernel speed, so no QPS runs are needed to decide.
#
# A variant is L<n>[_2pass]:
#   L<n>     beam width (candidate list) of the build's search; indexes built
#            before L192 became the default used L64 at R = 64. A build-time
#            constant: each distinct L gets its own build directory
#            (build_L<n>, cmake -DJASPER_BUILD_L=<n>), so the config tables stay
#            as they are. n <= 192.
#   _2pass   Vamana's two-pass build (create_diskann_index's default): a round
#            at alpha 1.0, then the round at ALPHA over the same graph. Without
#            it the variant is built with --one_pass.
#
# L64 reuses $WORK/<ds>_r64_64_plain.bin, the index every earlier sweep
# searched, so it is the baseline as measured; other variants are built to
# $INDEX_DIR/<ds>_r64_64_<variant>_plain.bin (default /scratch/$USER, as $WORK
# runs out of quota: an sra50 index is ~19 GB) and reused when present.
#
# ── Outputs, under $OUT ────────────────────────────────────────────────────
#   raw/<ds>_<variant>_<method>_top<k>.pops.csv   distortion_test --pops
#   raw/<ds>_<variant>_<method>_top<k>.pops.log   its console output
#   raw/<ds>_<variant>_build.log                  create_diskann_index output
#   compare_<ds>.txt   per method and k: each variant's recall ceiling and the
#                      nodes expanded to reach each target recall, and build time
#                      (scripts/compare_builds.py)
#
#   DRY_RUN=1 ./sbatch/bench_graph_build.sbatch
#   sbatch sbatch/bench_graph_build.sbatch
#   VARIANTS="L64 L128 L192 L192_2pass" DATASETS=sra50 sbatch sbatch/bench_graph_build.sbatch
#   SUMMARY_ONLY=1 ./sbatch/bench_graph_build.sbatch

set -euo pipefail

# ── Tunables ───────────────────────────────────────────────────────────────
DATASETS="${DATASETS:-deep10M sra50}"
VARIANTS="${VARIANTS:-L64 L192 L192_2pass}"
METHODS="${METHODS:-conventional lsh}"
K_RANKS="${K_RANKS:-16}"               # for the directional method(s)
K="${K:-10 100}"
WIDTHS="${WIDTHS:-16 32 63 128 191 256 447 959 1983}"
LIMIT="${LIMIT:-8192}"
TARGETS="${TARGETS:-0.8 0.9 0.95 0.98 0.99}"
R="${R:-64}"                           # dim > 128 configs are compiled only at R=64
ALPHA="${ALPHA:-1.2}"
N_PARTS="${N_PARTS:-1}"
N_DUP="${N_DUP:-2}"
WORKSPACE_BUDGET="${WORKSPACE_BUDGET:-10GB}"
DRY_RUN="${DRY_RUN:-0}"
SUMMARY_ONLY="${SUMMARY_ONLY:-0}"

ANN_DATA="${ANN_DATA:-/projects/SaltSystemsLab/ann_data}"
ANN_GT="${ANN_GT:-/projects/SaltSystemsLab/ann_gt}"
LOCALE="${LOCALE:-/tmp2/ann_data/locale}"
REPO="${REPO:-$HOME/jasper}"
WORK="${WORK:-/projects/SaltSystemsLab/${USER:-$(id -un)}/distortion}"
INDEX_DIR="${INDEX_DIR:-/scratch/${USER:-$(id -un)}}"   # where new variants are built
OUT="${OUT:-$REPO/graph_build}"
RAW="$OUT/raw"

# name -> base|queries|gt|dtype|dim   (same table as bench_adaptive_termination.sbatch)
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
gt_width() { od -An -tu4 -j4 -N4 "$1" 2>/dev/null | tr -d ' '; }

# Variant -> L and two-pass flag.
variant_L()     { local v=${1#L}; echo "${v%%_*}"; }
variant_2pass() { case "$1" in *_2pass) echo 1;; *) echo 0;; esac; }
variant_index() {                      # $1 dataset, $2 variant
  if [ "$2" = "L64" ]; then echo "$WORK/${1}_r${R}_${R}_plain.bin"
  else echo "$INDEX_DIR/${1}_r${R}_${R}_${2}_plain.bin"; fi
}
build_dir() { echo "build_L$1"; }

summarize() {
  for ds in $DATASETS; do
    python3 -I "$REPO/scripts/compare_builds.py" "$RAW" --dataset "$ds" \
        --variants $VARIANTS --targets $TARGETS \
      | tee "$OUT/compare_$ds.txt" || echo "  SUMMARY FAILED for $ds"
  done
}
if [ "$SUMMARY_ONLY" != "0" ]; then summarize; exit 0; fi

# ── Validate + plan ────────────────────────────────────────────────────────
LS=""
for v in $VARIANTS; do
  case "$v" in L[0-9]*) ;; *) echo "bad variant: $v (L<n> or L<n>_2pass)" >&2; exit 2;; esac
  l=$(variant_L "$v")
  case "$l" in ''|*[!0-9]*) echo "bad variant: $v" >&2; exit 2;; esac
  [ "$l" -ge 1 ] && [ $(( l + 64 )) -le 256 ] || { echo "variant $v: L must be 1..192" >&2; exit 2; }
  case " $LS " in *" $l "*) ;; *) LS="$LS $l";; esac
done
POINTS_FOR() {                         # $1 = search k
  local w p=""
  for w in $WIDTHS; do
    [ "$w" -ge "$1" ] && [ $(( w + 64 )) -lt 2048 ] && p="$p $w:$LIMIT"
  done
  echo "${p# }"
}

echo "=== graph build comparison ==="
echo "  host:       $(hostname)"
echo "  datasets:   $DATASETS"
echo "  variants:   $VARIANTS   (build dirs:$(for l in $LS; do printf ' %s' "$(build_dir "$l")"; done))"
echo "  methods:    $METHODS   (directional k_ranks $K_RANKS)"
echo "  k:          $K"
echo "  widths:     $WIDTHS"
echo "  graph:      R=$R alpha=$ALPHA n_parts=$N_PARTS"
echo "  out:        $OUT"
for ds in $DATASETS; do
  spec=$(dataset_spec "$ds") || { echo "unknown dataset: $ds" >&2; exit 2; }
  IFS='|' read -r base q gt dtype dim <<<"$spec"
  printf "  %-10s dim=%-5s %s\n" "$ds" "$dim" "$dtype"
  for v in $VARIANTS; do
    idx=$(variant_index "$ds" "$v")
    printf "    %-12s %s%s\n" "$v" "$idx" "$([ -s "$idx" ] && echo ' (exists)' || echo ' (will build)')"
  done
  [ -r "$q" ]  || echo "    WARNING: queries missing: $q"
  [ -r "$gt" ] || echo "    WARNING: ground truth missing: $gt"
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
for l in $LS; do
  bd=$(build_dir "$l")
  cmake -B "$bd" -DJASPER_BUILD_CMD=ON -DJASPER_BUILD_TESTS=OFF -DJASPER_BUILD_FFI=OFF \
    -DJASPER_BUILD_L="$l" -DThrust_DIR="$THRUST_DIR" -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" >/dev/null
  cmake --build "$bd" --target create_diskann_index -j8 >/dev/null
  echo "  $bd (L=$l): ok"
done
# Search binaries do not depend on the build's L, so `build` keeps whatever L
# it was configured with.
cmake -B build -DJASPER_BUILD_CMD=ON -DJASPER_BUILD_TESTS=OFF -DJASPER_BUILD_FFI=OFF \
  -DThrust_DIR="$THRUST_DIR" -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" >/dev/null
cmake --build build --target distortion_test -j8 >/dev/null
mkdir -p "$WORK" "$INDEX_DIR" "$RAW"

# ── Build + measure ────────────────────────────────────────────────────────
for ds in $DATASETS; do
  IFS='|' read -r base q gt dtype dim <<<"$(dataset_spec "$ds")"
  gt_k=$(gt_width "$gt"); gt_k=${gt_k:-0}
  echo
  echo "######## $ds ########"
  for v in $VARIANTS; do
    idx=$(variant_index "$ds" "$v")
    l=$(variant_L "$v")
    blog="$RAW/${ds}_${v}_build.log"
    if [ -s "$idx" ]; then
      echo "[$(stamp)] $v: reusing $idx"
    else
      echo "[$(stamp)] $v: building -> $idx"
      two=(--one_pass); [ "$(variant_2pass "$v")" = 1 ] && two=()
      "./$(build_dir "$l")/bin/create_diskann_index" \
          -f "$base" -t "$dtype" -d l2 --dim "$dim" \
          --r_intermediate "$R" --r_final "$R" \
          --n_parts "$N_PARTS" --n_dup "$N_DUP" -a "$ALPHA" ${two[@]+"${two[@]}"} \
          -w "$WORKSPACE_BUDGET" -i "$idx" >| "$blog" 2>&1 \
        || { echo "  BUILD FAILED (see $blog) — skipping $ds $v"; continue; }
      echo "  $(grep -m1 'build (shard' "$blog" || true)"
    fi
    for topk in $K; do
      if [ "$topk" -gt "$gt_k" ]; then echo "  ${topk}@${topk}: SKIP (ground truth has $gt_k)"; continue; fi
      pts=$(POINTS_FOR "$topk")
      for m in $METHODS; do
        r=$K_RANKS; [ "$m" = conventional ] && r=4   # conventional ignores it, but needs a config
        name="${ds}_${v}_${m}_top${topk}"
        echo "[$(stamp)] $v $m ${topk}@${topk}"
        ./build/bin/distortion_test -i "$idx" --dim "$dim" -t "$dtype" -n "$R" -k "$topk" \
            -m "$m" -r "$r" -q "$q" -g "$gt" --pops "$RAW/$name.pops.csv" -b $pts \
            >| "$RAW/$name.pops.log" 2>&1 \
          || { echo "  POPS FAILED ($name) — see $RAW/$name.pops.log"; continue; }
        grep 'mean_pops=' "$RAW/$name.pops.log" | tail -1 | sed 's/^/  last: /'
      done
    done
  done
done

echo
summarize
echo "[$(stamp)] done"
