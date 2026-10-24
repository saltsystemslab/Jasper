#!/bin/bash
#SBATCH --partition=SaltSystemsLab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --nodelist=d4067
#SBATCH --mem=987842M
#SBATCH --gres=gpu:1
#SBATCH --job-name=adaptive_term
#SBATCH --output=adaptive_term_%j.log
#SBATCH --error=adaptive_term_%j.err
#SBATCH --time=24:00:00
#
# Pareto comparison of four searches on one plain index, each over its whole
# parameter space:
#
#   conventional   exact-distance beam search            beam width
#   lsh            directional search, CPH/LSH codes     beam width, and γ
#   rabitq         directional search, RaBitQ codes      beam width, and γ
#   pq             directional search, PQ codes          beam width, and γ
#
# METHODS picks any subset. scripts/plot_pareto.py draws one Pareto curve per
# method (QPS on device and host, nodes expanded and vectors read, each against
# recall), and the summary compares the two stop rules within each DBS method.
#
# Adaptive Beam Search (Al-Jazzazi et al. 2025, arXiv:2505.15636, eq. 3)
# swaps the stop rule "the beam_width best candidates are all expanded" for
# "the next candidate is at least (1+γ)× farther than d_k, the k-th best of the
# other discovered candidates". In the directional kernel both sides are edge
# ESTIMATES (CPH / PQ / RaBitQ), and γ = 0 stops where beam_width = k does
# (up to ties) — see step 2b of graph_beam_search.cuh. This job
# The job measures each (dataset, method, k_ranks, k) cell and placement,
# where k is the search k of recall k@k (K, default 10 and 100):
#
#   throughput   run_query: QPS and recall@K at every point
#   pops         distortion_test --pops: nodes popped per query and recall@K
#                (the same searches with visited logging on, untimed)
#
# Both rules go through ONE invocation of each binary, so they search the same
# graph with the same codes, and a recall difference is the stop rule alone:
#
#   beam      every BEAM_WIDTHS entry, as beam_width:LIMIT. By default a
#             geometric grid from 16 to 1983 (the widest point the dispatch
#             compiles, beam_width + 64 < 2048) in steps of BEAM_RATIO, plus
#             the widest width of every compiled tier (63 191 447 959 1983).
#             Beam search has one knob, so its curve is only as fine as this
#             grid; a coarse one makes it reach each target recall with a
#             wider beam than it needs and understates it.
#   adaptive  (DBS methods only; conventional search keeps the beam-width
#             rule) every CAPACITIES x GAMMAS pair, as capacity:LIMIT:gamma. The
#             capacity is passed as beam_width and only caps the candidate list.
#             CAPACITIES spans 16 through 1983 on a coarser grid: γ already
#             gives this side a second axis, and the grid multiplies by it.
#             It carries the same tier tops as the beam grid (63 191 447 959
#             1983), so both rules get the cheapest width per slot.
#   rerank    (DBS methods only) beam search at every RERANK_WIDTHS entry
#             with a re-rank of M = m x k candidates for each m in
#             RERANK_MULTS, as width:LIMIT:-:M: the list keeps M more than the
#             beam_width it expands, and when the search stops the best M
#             unexpanded ones get an exact distance (a vector read each, no
#             expansion) and compete for the top-k. The list then holds
#             width + M, which picks the compiled tier, so width + M + 64 must
#             stay under 2048, and each M also runs the widths that fill a tier
#             to its top (tier top - M). RERANK_MULTS="" turns this off.
#
# ── Wide tier ──────────────────────────────────────────────────────────────
# WIDE_TIER=1 builds with -DJASPER_WIDE_TIER=ON (in build_wide/, so the default
# build is not recompiled), which gives lsh and rabitq a 4096-entry list:
# their points may then reach beam_width + M + 64 < 4096. Beam adds WIDE_WIDTHS
# (default 2380 2856 3427 4031: the 20% steps past 1983, then the top),
# adaptive adds WIDE_CAPACITIES (4031), and re-rank adds WIDE_WIDTHS and 4031 - M.
# Conventional and pq keep 2048: conventional reaches every target recall
# within it. The 4096 instance runs 256 threads (WIDE_TIER_BLOCK; 0 = 128 like
# the other widths): at 128 the per-pop list work outgrew the PCIe reads, and
# 256 gave 27% more QPS at width 4031 on sra50, same recall. The build step
# prints each list width's registers, stack and spills. ONLY_WIDE=1 runs just the
# points whose list lands in the 4096 width, for a quick check of whether the
# recall ceiling moves (cells left with no points are skipped).
#
# ── Placements ─────────────────────────────────────────────────────────────
#   device   the graph is resident on the card
#   host     the graph lives in pinned host memory (--on_host), so every
#            vector, edge list and code read of a pop crosses PCIe
# Each placement runs both steps. Pops and recall belong to the search, not to
# where the graph lives, so they should match across placements (the summary
# checks); only QPS moves. On host the per-pop cost is mostly the PCIe reads,
# so the adaptive/beam QPS ratio should follow the pop ratio more closely.
#
# ── Reading the capacity axis ──────────────────────────────────────────────
# Per-pop cost is set by the compiled search width the dispatch derives from
# beam_width (the first of 128/256/512/1024/2048[/4096] above beam_width + 64), not by
# the rule. Adaptive at capacity C and beam at width C run on the same tier and
# pay the same per pop, so there the QPS gap is the pop gap. A larger capacity
# costs more per pop; a smaller one saturates: once the list holds fewer
# in-slack candidates than the query needs, the beam-width rule fires before γ
# does and recall plateaus as γ grows. The summary's "max recall" line shows
# where each capacity tops out.
#
# Each k gets its own point list: a beam or capacity narrower than k cannot
# hold the k results, so widths below k are dropped for that k (100@100 starts
# at 100). A k is skipped for a dataset whose ground truth has fewer neighbors.
#
# LIMIT caps pops for every point and is set high so that neither rule stops on
# it, even at width 1983; the summary warns if any query did (capped_frac > 0).
# The pops run keeps a visited log of n_queries x (LIMIT+1) x 8 bytes, on the
# device and again on the host (~650 MB each for 10k queries at 8192).
#
# ── Outputs, under $OUT ────────────────────────────────────────────────────
#   raw/<ds>_<method>_k<k_ranks>_top<k>_<placement>.qps.log   run_query output
#   raw/<ds>_<method>_k<k_ranks>_top<k>_<placement>.pops.csv  distortion_test
#                                                  --pops (stop,gamma columns)
#   raw/<ds>_<method>_k<k_ranks>_top<k>_<placement>.pops.log  its console output
#   adaptive_<ds>.csv     both joined, one row per (k, placement, point)
#   adaptive_<ds>.txt     per k, placement and target recall: best QPS and fewest
#                         pops under each rule, and the ratios
#                         (scripts/adaptive_summary.py)
#   figures/pareto_<ds>_top<k>.pdf   one Pareto curve per method per panel
#   pareto_<ds>.csv       the points on those curves (the figures' table view)
#   pareto_<ds>.txt       per placement and target recall: each method's best
#                         QPS, nodes expanded and vectors read, and its config
#                         (scripts/plot_pareto.py)
#
# conventional cells are named with k_ranks 0 (k0): it has no codes.
#
# Indexes: $INDEX_DIR/<ds>_r64_64_L<BUILD_L>[_2pass]_plain.bin (default
# /scratch/$USER, L192 two-pass, the same names bench_graph_build.sbatch
# writes); a missing one is built in one part. The build settings are in the
# name so a changed default never silently reuses an old graph. INDEX_<ds>
# overrides the path. L2 only: distortion_test compiles L2 configs only, and
# every dataset below is L2.
#
# Query chunks: QUERY_CHUNK_<placement> (device | host) runs run_query with
# --query_chunk, bounding the searches in flight, and
# QUERY_CHUNK_<placement>_<method> overrides it per method. With the graph in
# host memory too many concurrent searches cost conventional search most of its
# PCIe bandwidth (sra50: 12-23 GB/s at 10k in flight vs 43-48 at 500), so a
# fair host comparison runs each method at its own best chunk. 0 = all at once.
#
#   DRY_RUN=1 ./sbatch/bench_adaptive_termination.sbatch     # plan, no GPU
#   sbatch sbatch/bench_adaptive_termination.sbatch
#   DATASETS="deep10M sra50" K_LIST=16 sbatch sbatch/bench_adaptive_termination.sbatch
#   METHODS="conventional lsh" K_LIST="4 16" sbatch sbatch/bench_adaptive_termination.sbatch
#   CAPACITIES=256 GAMMAS="0 0.05 0.1 0.2" sbatch sbatch/bench_adaptive_termination.sbatch
#   RERANK_MULTS="1 4" METHODS="lsh rabitq" sbatch sbatch/bench_adaptive_termination.sbatch
#   QUERY_BATCH=100000 sbatch sbatch/bench_adaptive_termination.sbatch  # tile for QPS
#   PLACEMENTS=host RUN_POPS=0 sbatch sbatch/bench_adaptive_termination.sbatch
#   SUMMARY_ONLY=1 ./sbatch/bench_adaptive_termination.sbatch   # re-join raw/, no GPU
#   WIDE_TIER=1 ONLY_WIDE=1 DATASETS=sra50 METHODS="lsh rabitq" K_LIST=16 \
#     OUT=$HOME/jasper/wide_trial sbatch sbatch/bench_adaptive_termination.sbatch

set -euo pipefail

# ── Tunables ───────────────────────────────────────────────────────────────
DATASETS="${DATASETS:-deep10M}"
METHODS="${METHODS:-conventional lsh rabitq pq}"
K_LIST="${K_LIST:-8}"                  # DBS k_ranks; 8 is valid for every method and dim below
K="${K:-10 100}"                       # search k, one cell per value (recall k@k)
BEAM_MIN="${BEAM_MIN:-16}"
BEAM_MAX="${BEAM_MAX:-1983}"
BEAM_RATIO="${BEAM_RATIO:-1.2}"        # beam grid step; 1 = every integer width
CAPACITIES="${CAPACITIES:-16 32 63 128 191 256 447 959 1983}"
GAMMAS="${GAMMAS:-0 0.02 0.04 0.06 0.08 0.1 0.12 0.15 0.2 0.25 0.3 0.4 0.5 0.75 1}"
RERANK_MULTS="${RERANK_MULTS-1 2 4}"   # re-rank M = m x k per m; "" = no re-rank points
RERANK_WIDTHS="${RERANK_WIDTHS:-$CAPACITIES}"
LIMIT="${LIMIT:-8192}"
REPEATS="${REPEATS:-3}"                # timed passes over the point list; median QPS
QUERY_BATCH="${QUERY_BATCH:-0}"        # tile queries for QPS; pops use the original set
PLACEMENTS="${PLACEMENTS:-device host}" # host = graph in pinned host memory (--on_host)
RUN_QPS="${RUN_QPS:-1}"
RUN_POPS="${RUN_POPS:-1}"
TARGETS="${TARGETS:-0.9 0.95 0.98 0.99 0.995}"
R_INT="${R_INT:-64}"
R_FIN="${R_FIN:-64}"                   # dim > 128 configs are compiled only at R=64
N_PARTS="${N_PARTS:-1}"
N_DUP="${N_DUP:-2}"
ALPHA="${ALPHA:-1.2}"
WORKSPACE_BUDGET="${WORKSPACE_BUDGET:-10GB}"
BUILD_L="${BUILD_L:-192}"              # create_diskann_index's build L (cmake -DJASPER_BUILD_L)
TWO_PASS="${TWO_PASS:-1}"              # 0 = --one_pass
QUERY_CHUNK_device="${QUERY_CHUNK_device:-0}"
QUERY_CHUNK_host="${QUERY_CHUNK_host:-0}"
WIDE_TIER="${WIDE_TIER:-0}"            # 1 = lsh/rabitq get a 4096-entry list (see "Wide tier")
WIDE_TIER_BLOCK="${WIDE_TIER_BLOCK:-256}" # its block size; 0 = the other widths' (128)
WIDE_WIDTHS="${WIDE_WIDTHS:-2380 2856 3427 4031}"
WIDE_CAPACITIES="${WIDE_CAPACITIES-4031}"
ONLY_WIDE="${ONLY_WIDE:-0}"            # 1 = only points whose list lands in the 4096 width
BUILD_DIR="${BUILD_DIR:-build$([ "$WIDE_TIER" != 1 ] || echo "_wide$([ "$WIDE_TIER_BLOCK" = 256 ] || echo "$WIDE_TIER_BLOCK")")}"
DRY_RUN="${DRY_RUN:-0}"
SUMMARY_ONLY="${SUMMARY_ONLY:-0}"

ANN_DATA="${ANN_DATA:-/projects/SaltSystemsLab/ann_data}"
ANN_GT="${ANN_GT:-/projects/SaltSystemsLab/ann_gt}"
LOCALE="${LOCALE:-/tmp2/ann_data/locale}"
REPO="${REPO:-$HOME/jasper}"
WORK="${WORK:-/projects/SaltSystemsLab/${USER:-$(id -un)}/distortion}"
INDEX_DIR="${INDEX_DIR:-/scratch/${USER:-$(id -un)}}"
BUILD_TAG="L${BUILD_L}$([ "$TWO_PASS" != 1 ] || echo _2pass)"
OUT="${OUT:-$REPO/adaptive_termination}"
RAW="$OUT/raw"

# name -> base|queries|gt|dtype|dim   (same table as distortion_sweep.sbatch)
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

# Registers and stack of the directional kernels, by compiled list width: the
# largest Lj<n>E template argument of the mangled name (block size, degree,
# k_ranks and tile are all <= 256, so widths 128 and 256 show as "<=256").
# STACK is the merge window's per-thread arrays (2 x width/block x 8 bytes, in
# local memory); LOCAL above 0 is a register spill. Never fails the job.
kernel_resources() {                   # $1 = binary, $2 = its build dir
  local nvcc_dir objdump
  # The cuda module is unloaded by now; cuobjdump sits next to the nvcc cmake used.
  nvcc_dir=$(dirname "$(sed -n 's/^CMAKE_CUDA_COMPILER:[A-Z]*=//p' "$2/CMakeCache.txt" 2>/dev/null)")
  objdump=$(command -v cuobjdump || { [ -x "$nvcc_dir/cuobjdump" ] && echo "$nvcc_dir/cuobjdump"; } || true)
  [ -n "$objdump" ] || { echo "  (no cuobjdump on PATH or in $nvcc_dir: register check skipped)"; return 0; }
  echo "  directional kernels by list width (max over configs):"
  { "$objdump" -res-usage "$1" 2>/dev/null || true; } | awk '
    /graph_beam_search_kernel/ {
      w = 0; name = $0
      while (match(name, /Lj[0-9]+E/)) {
        v = substr(name, RSTART + 2, RLENGTH - 3) + 0
        if (v > w) w = v
        name = substr(name, RSTART + RLENGTH)
      }
      w = (w > 256) ? w : "<=256"; want = 1; next
    }
    want && /REG:/ {
      for (i = 1; i <= NF; i++) {
        split($i, kv, ":")
        if (kv[1] == "REG"   && kv[2] + 0 > reg[w]) reg[w] = kv[2] + 0
        if (kv[1] == "STACK" && kv[2] + 0 > stk[w]) stk[w] = kv[2] + 0
        if (kv[1] == "LOCAL" && kv[2] + 0 > lcl[w]) lcl[w] = kv[2] + 0
      }
      n[w]++; want = 0
    }
    END {
      for (w in n) printf "    width %-6s %3d kernels   REG %-4d STACK %-5d LOCAL %d\n", w, n[w], reg[w], stk[w], lcl[w]
      if (length(n) == 0) print "    (no graph_beam_search_kernel found)"
    }' | sort -k2 -n || true
}

# Default beam sweep: BEAM_RATIO steps from BEAM_MIN to BEAM_MAX (at least +1
# each), plus each compiled tier's widest width — one wider moves the search to
# the next compiled search width and a dearer pop, so those are the cheapest
# beams per slot and the baseline should have them.
beam_grid() {
  awk -v lo="$BEAM_MIN" -v hi="$BEAM_MAX" -v r="$BEAM_RATIO" 'BEGIN {
    for (w = lo; w < hi; ) { print w; n = int(w * r + 0.5); w = (n > w) ? n : w + 1 }
    print hi
    n = split("63 191 447 959 1983", tier, " ")
    for (i = 1; i <= n; i++) if (tier[i] >= lo && tier[i] <= hi) print tier[i]
  }' | sort -n -u | tr '\n' ' '
}
BEAM_WIDTHS="${BEAM_WIDTHS:-$(beam_grid)}"
BEAM_WIDTHS="${BEAM_WIDTHS% }"

summarize() {                          # $1 dataset
  python3 -I "$REPO/scripts/adaptive_summary.py" "$RAW" --dataset "$1" \
      --out "$OUT/adaptive_$1.csv" --targets $TARGETS \
    | tee "$OUT/adaptive_$1.txt" \
    || echo "  SUMMARY FAILED for $1"
  # Needs matplotlib, which the job's conda env has; a failure only costs the
  # figures, the joined CSV is already written.
  python3 -I "$REPO/scripts/plot_pareto.py" "$OUT/adaptive_$1.csv" \
      --out "$OUT" --targets $TARGETS \
    | tee "$OUT/pareto_$1.txt" \
    || echo "  PARETO PLOT FAILED for $1"
}

if [ "$SUMMARY_ONLY" != "0" ]; then
  for ds in $DATASETS; do summarize "$ds"; done
  exit 0
fi

# ── Points ─────────────────────────────────────────────────────────────────
# Built per search k and method: the searches dispatch on list + 64 < 2048
# (4096 for lsh and rabitq under WIDE_TIER=1), where the list is beam_width,
# plus M with re-rank; and a width under k cannot hold the k results.
max_list() {                           # $1 = method
  if [ "$WIDE_TIER" = 1 ] && { [ "$1" = lsh ] || [ "$1" = rabitq ]; }; then echo 4096; else echo 2048; fi
}
fits() {                               # $1 = list, $2 = search k, $3 = method
  [ "$1" -ge "$2" ] && [ $(( $1 + 64 )) -lt "$(max_list "$3")" ] &&
    { [ "$ONLY_WIDE" = 0 ] || [ $(( $1 + 64 )) -ge 2048 ]; }
}
build_points() {                       # $1 = search k, $2 = method
  local topk=$1 method=${2:-lsh} w g mult rr
  BEAM_POINTS=""; ADAPT_POINTS=""; RERANK_POINTS=""
  for w in $BEAM_WIDTHS $WIDE_WIDTHS; do
    if fits "$w" "$topk" "$method"; then BEAM_POINTS="$BEAM_POINTS $w:$LIMIT"; fi
  done
  for w in $CAPACITIES $WIDE_CAPACITIES; do
    [ "$method" = "conventional" ] && break   # beam-width rule only
    fits "$w" "$topk" "$method" || continue
    for g in $GAMMAS; do ADAPT_POINTS="$ADAPT_POINTS $w:$LIMIT:$g"; done
  done
  # Re-rank on top of the beam-width rule. The list holds w + M, and that is
  # what picks the tier, so each M also gets the widths that fill a tier to its
  # top (63 191 447 959 1983 [4031], less M): the cheapest per slot, as on the
  # beam grid.
  [ "$method" = "conventional" ] || for mult in $RERANK_MULTS; do
    rr=$(( mult * topk ))
    for w in $(printf '%s\n' $RERANK_WIDTHS $WIDE_WIDTHS $(( 63 - rr )) $(( 191 - rr )) $(( 447 - rr )) \
                 $(( 959 - rr )) $(( 1983 - rr )) $(( 4031 - rr )) | sort -n -u); do
      [ "$w" -ge "$topk" ] || continue
      if fits $(( w + rr )) "$topk" "$method"; then RERANK_POINTS="$RERANK_POINTS $w:$LIMIT:-:$rr"; fi
    done
  done
  POINTS="${BEAM_POINTS# } ${ADAPT_POINTS# } ${RERANK_POINTS# }"
  POINTS=$(tr -s ' ' <<<"$POINTS"); POINTS="${POINTS# }"; POINTS="${POINTS% }"
  # Whole passes rather than back-to-back repeats, so clock or thermal drift
  # spreads over every point instead of landing on a few.
  QPS_POINTS=""
  for _ in $(seq "$REPEATS"); do QPS_POINTS="$QPS_POINTS $POINTS"; done
  N_BEAM=$(wc -w <<<"$BEAM_POINTS" | tr -d ' ')
  N_ADAPT=$(wc -w <<<"$ADAPT_POINTS" | tr -d ' ')
  N_RERANK=$(wc -w <<<"$RERANK_POINTS" | tr -d ' ')
}

for topk in $K; do
  case "$topk" in ''|*[!0-9]*|0) echo "bad search k: $topk" >&2; exit 2;; esac
  build_points "$topk"
  [ -n "$POINTS" ] || { echo "no points to run at k=$topk (need $topk <= width <= 1983, or 4031 with WIDE_TIER=1)" >&2; exit 2; }
done

# Neighbors per query in a ground-truth file: the second int32 of its header.
gt_width() { od -An -tu4 -j4 -N4 "$1" 2>/dev/null | tr -d ' '; }

for place in $PLACEMENTS; do
  case "$place" in device|host) ;; *) echo "unknown placement: $place (device | host)" >&2; exit 2;; esac
done
N_PLACE=$(wc -w <<<"$PLACEMENTS" | tr -d ' ')
[ "$N_PLACE" -gt 0 ] || { echo "no placements" >&2; exit 2; }

# ── Plan ───────────────────────────────────────────────────────────────────
echo "=== Pareto: conventional vs DBS (beam width x γ) ==="
echo "  host:       $(hostname)"
echo "  datasets:   $DATASETS"
echo "  methods:    $METHODS    DBS k_ranks: $K_LIST"
echo "  beam:       widths"
fold -s -w 96 <<<"$BEAM_WIDTHS" | sed 's/^/                /'
echo "  adaptive:   capacities $CAPACITIES"
echo "              x gammas $GAMMAS"
echo "  rerank:     $([ -n "${RERANK_MULTS// }" ] && echo "widths $RERANK_WIDTHS" || echo off)"
[ -n "${RERANK_MULTS// }" ] && echo "              x M = (${RERANK_MULTS}) x k"
for topk in $K; do
  build_points "$topk" conventional; n_conv=$N_BEAM
  build_points "$topk"
  printf "  %-11s %s beam + %s adaptive + %s rerank points per lsh/rabitq cell, %s for conventional\n" \
    "${topk}@${topk}:" "$N_BEAM" "$N_ADAPT" "$N_RERANK" "$n_conv"
done
echo "  limit:      $LIMIT pops    qps passes: $REPEATS    query batch: $([ "$QUERY_BATCH" = 0 ] && echo untiled || echo "$QUERY_BATCH")"
echo "  chunks:     device $QUERY_CHUNK_device, host $QUERY_CHUNK_host (0 = all at once; per-method QUERY_CHUNK_<placement>_<method> overrides)"
echo "  build:      $BUILD_TAG -> $INDEX_DIR"
echo "  wide tier:  $([ "$WIDE_TIER" = 1 ] && echo "on (lsh, rabitq; block $([ "$WIDE_TIER_BLOCK" = 0 ] && echo 128 || echo "$WIDE_TIER_BLOCK")), widths $WIDE_WIDTHS, capacities ${WIDE_CAPACITIES:-none}$([ "$ONLY_WIDE" = 1 ] && echo ', ONLY those points')" || echo off)    binaries: $BUILD_DIR"
echo "  placements: $PLACEMENTS"
echo "  steps:      $([ "$RUN_QPS" = 1 ] && echo 'run_query ')$([ "$RUN_POPS" = 1 ] && echo 'distortion_test --pops')"
echo "  graph:      R_int=$R_INT R_fin=$R_FIN dist=l2"
echo "  out:        $OUT"
echo
CELLS=""
TOTAL_POINTS=0
for ds in $DATASETS; do
  combos=""
  spec=$(dataset_spec "$ds") || { echo "unknown dataset: $ds" >&2; exit 2; }
  IFS='|' read -r base q gt dtype dim <<<"$spec"
  padded=$(( (dim + 7) / 8 * 8 ))
  ivar="INDEX_$ds"; idx="${!ivar:-$INDEX_DIR/${ds}_r${R_INT}_${R_FIN}_${BUILD_TAG}_plain.bin}"
  printf "  %-10s dim=%-5s %-6s index=%s%s\n" "$ds" "$dim" "$dtype" "$idx" \
    "$([ -s "$idx" ] && echo ' (exists)' || echo ' (will build)')"
  [ -s "$idx" ] || [ -r "$base" ] || echo "    WARNING: no index and no base at $base"
  [ -r "$q" ]  || echo "    WARNING: queries missing: $q"
  [ -r "$gt" ] || { echo "    SKIP: no ground truth at $gt (recall is the comparison axis)"; continue; }
  gt_k=$(gt_width "$gt"); gt_k=${gt_k:-0}
  if [ "$dim" -gt 128 ] && [ "$R_FIN" != "64" ]; then
    echo "    ERROR: dim > 128 needs R_FIN=64 (uint16-packed configs)" >&2; exit 2
  fi
  for m in $METHODS; do
    case "$m" in conventional|lsh|pq|rabitq) ;;
      *) echo "unknown method: $m (conventional | lsh | pq | rabitq)" >&2; exit 2;; esac
    # Conventional search has no codes, so it is one cell whatever K_LIST is.
    if [ "$m" = "conventional" ]; then combos="$combos conventional:0"; continue; fi
    for k in $K_LIST; do
      # The compiled config tables (run_query, distortion_test) stop here.
      case "$k" in 4|8|16) ;; *) echo "    $m k$k: SKIP (k_ranks must be 4, 8 or 16)"; continue;; esac
      if [ "$m" = "pq" ] && [ $(( padded % k )) -ne 0 ]; then
        echo "    $m k$k: SKIP (padded_dim=$padded not divisible by $k)"; continue
      fi
      [ "$m" = "rabitq" ] && [ $(( 8 * k )) -gt "$padded" ] && \
        echo "    $m k$k: note — rq_dim=$(( 8 * k )) > dim, codes all $padded dims (no projection)"
      combos="$combos $m:$k"
    done
  done
  for topk in $K; do
    if [ "$topk" -gt "$gt_k" ]; then
      echo "    ${topk}@${topk}: SKIP (ground truth has $gt_k neighbors per query)"; continue
    fi
    for c in $combos; do
      build_points "$topk" "${c%%:*}"
      [ -n "$POINTS" ] || { echo "    ${c%%:*} ${topk}@${topk}: SKIP (no points)"; continue; }
      CELLS="$CELLS $ds:$c:$topk"
      TOTAL_POINTS=$(( TOTAL_POINTS + N_BEAM + N_ADAPT + N_RERANK ))
    done
  done
done
CELLS="${CELLS# }"
echo
[ -n "$CELLS" ] || { echo "No runnable cells." >&2; exit 1; }
echo "  cells: $(wc -w <<<"$CELLS" | tr -d ' ') x $N_PLACE placement(s); per placement" \
     "$TOTAL_POINTS points ($(( TOTAL_POINTS * REPEATS )) timed searches + $TOTAL_POINTS logged)"

if [ "$DRY_RUN" != "0" ]; then
  echo
  echo "DRY_RUN set — nothing run."
  exit 0
fi

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
  -DJASPER_BUILD_L="$BUILD_L" \
  -DJASPER_WIDE_TIER="$([ "$WIDE_TIER" = 1 ] && echo ON || echo OFF)" \
  -DJASPER_WIDE_TIER_BLOCK="$WIDE_TIER_BLOCK" \
  -DThrust_DIR="$THRUST_DIR" -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" >/dev/null
cmake --build "$BUILD_DIR" --target create_diskann_index run_query distortion_test -j8 >/dev/null
echo "  ok"
if [ "$WIDE_TIER" = 1 ]; then kernel_resources "$BUILD_DIR/bin/run_query" "$BUILD_DIR"; fi
mkdir -p "$WORK" "$INDEX_DIR" "$RAW"

# ── Sweep ──────────────────────────────────────────────────────────────────
DONE_DS=""
for ds in $DATASETS; do
  case " $CELLS " in *" $ds:"*) ;; *) continue;; esac
  IFS='|' read -r base q gt dtype dim <<<"$(dataset_spec "$ds")"
  ivar="INDEX_$ds"; idx="${!ivar:-$INDEX_DIR/${ds}_r${R_INT}_${R_FIN}_${BUILD_TAG}_plain.bin}"

  echo
  echo "######## $ds ########"
  if [ -s "$idx" ]; then
    echo "[$(stamp)] reusing index $idx"
  else
    echo "[$(stamp)] building plain index -> $idx"
    "$BUILD_DIR"/bin/create_diskann_index \
        -f "$base" -t "$dtype" -d l2 --dim "$dim" \
        --r_intermediate "$R_INT" --r_final "$R_FIN" \
        --n_parts "$N_PARTS" --n_dup "$N_DUP" -a "$ALPHA" \
        $([ "$TWO_PASS" = 1 ] || echo --one_pass) \
        -w "$WORKSPACE_BUDGET" -i "$idx" >| "$RAW/${ds}_build.log" 2>&1 \
      || { echo "  BUILD FAILED (see $RAW/${ds}_build.log) — skipping $ds"; continue; }
  fi

  # Tiled queries feed the timed runs only: a pop count is per query, so
  # tiling cannot change it, and the logged runs stay small.
  qq="$q"; qg="$gt"
  if [ "$QUERY_BATCH" != "0" ] && [ "$RUN_QPS" = "1" ]; then
    tq="$WORK/${ds}_x${QUERY_BATCH}_query.fbin"
    tg="$WORK/${ds}_x${QUERY_BATCH}_gt.ibin"
    if [ ! -s "$tq" ] || [ ! -s "$tg" ]; then
      python3 -I "$REPO/scripts/tile_queries.py" --target "$QUERY_BATCH" \
          --query "$q" --out-query "$tq" --gt "$gt" --out-gt "$tg" \
        || { echo "  tiling failed; timing the untiled query set"; rm -f "$tq" "$tg"; }
    fi
    [ -s "$tq" ] && [ -s "$tg" ] && { qq="$tq"; qg="$tg"; }
  fi

  for cell in $CELLS; do
    IFS=':' read -r cds m k topk <<<"$cell"
    [ "$cds" = "$ds" ] || continue
    build_points "$topk" "$m"
    # run_query picks the search by flag; conventional is the default path.
    # distortion_test always needs a k_ranks to pick a compiled config, which
    # conventional search ignores.
    if [ "$m" = "conventional" ]; then qps_mode=(); pops_k=4
    else qps_mode=("--$m" --k_ranks "$k"); pops_k=$k; fi
    for place in $PLACEMENTS; do
      host_arg=()
      [ "$place" = "host" ] && host_arg=(--on_host)
      cvar="QUERY_CHUNK_${place}_${m}"; pvar="QUERY_CHUNK_${place}"
      chunk="${!cvar:-${!pvar:-0}}"
      name="${ds}_${m}_k${k}_top${topk}_${place}"
      echo "[$(stamp)] $ds $m$([ "$m" = conventional ] || echo " k$k") ${topk}@${topk} on $place"

      if [ "$RUN_QPS" = "1" ]; then
        # No --prepopulated: codes are built at load from the plain index, the
        # same way distortion_test builds them below.
        "$BUILD_DIR"/bin/run_query -i "$idx" --dim "$dim" -d l2 -t "$dtype" -n "$R_FIN" -k "$topk" \
            ${qps_mode[@]+"${qps_mode[@]}"} ${host_arg[@]+"${host_arg[@]}"} \
            --query_chunk "$chunk" \
            -q "$qq" -g "$qg" --beam_limits $QPS_POINTS >| "$RAW/$name.qps.log" 2>&1 \
          || echo "  QPS FAILED ($name) — see $RAW/$name.qps.log"
        echo "  qps:  $(grep -c 'throughput=' "$RAW/$name.qps.log" || true) timed searches -> raw/$name.qps.log"
      fi

      if [ "$RUN_POPS" = "1" ]; then
        "$BUILD_DIR"/bin/distortion_test -i "$idx" --dim "$dim" -t "$dtype" -n "$R_FIN" -k "$topk" \
            -m "$m" -r "$pops_k" ${host_arg[@]+"${host_arg[@]}"} \
            -q "$q" -g "$gt" --pops "$RAW/$name.pops.csv" -b $POINTS >| "$RAW/$name.pops.log" 2>&1 \
          || echo "  POPS FAILED ($name) — see $RAW/$name.pops.log"
        echo "  pops: $(grep -c 'mean_pops=' "$RAW/$name.pops.log" || true) logged searches -> raw/$name.pops.csv"
      fi
    done
  done
  DONE_DS="$DONE_DS $ds"
done

# ── Summary ────────────────────────────────────────────────────────────────
for ds in $DONE_DS; do summarize "$ds"; done

echo
echo "=== done: $(stamp) ==="
echo "raw outputs in $RAW; joined CSV + rule comparison in $OUT/adaptive_<dataset>.{csv,txt}"
echo "Pareto figures in $OUT/figures; frontier points + method comparison in $OUT/pareto_<dataset>.{csv,txt}"
exit 0
