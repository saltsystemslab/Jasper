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
# Beam-width vs distance-adaptive termination for directional beam search.
#
# Adaptive Beam Search (Al-Jazzazi et al. 2025, arXiv:2505.15636, eq. 3)
# swaps the stop rule "the beam_width best candidates are all expanded" for
# "the next candidate is at least (1+γ)× farther than the k-th best result".
# In the directional kernel the candidate's distance is its edge ESTIMATE
# (CPH / PQ / RaBitQ) and d_k is EXACT — see graph_beam_search.cuh. This job
# measures what that buys, per (dataset, method, k_ranks) cell and placement:
#
#   throughput   run_query: QPS and recall@K at every point
#   pops         distortion_test --pops: nodes popped per query and recall@K
#                (the same searches with visited logging on, untimed)
#
# Both rules go through ONE invocation of each binary, so they search the same
# graph with the same codes, and a recall difference is the stop rule alone:
#
#   beam      every BEAM_WIDTHS entry, as beam_width:LIMIT, up to 1983 — the
#             widest point the dispatch compiles (beam_width + 64 < 2048)
#   adaptive  every CAPACITIES x GAMMAS pair, as capacity:LIMIT:gamma. The
#             capacity is passed as beam_width and only caps the candidate list.
#             CAPACITIES defaults to BEAM_WIDTHS, so every γ runs at every
#             width the beam sweep does, 16 through 1983.
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
# beam_width (the first of 128/256/512/1024/2048 above beam_width + 64), not by
# the rule. Adaptive at capacity C and beam at width C run on the same tier and
# pay the same per pop, so there the QPS gap is the pop gap. A larger capacity
# costs more per pop; a smaller one saturates: once the list holds fewer
# in-slack candidates than the query needs, the beam-width rule fires before γ
# does and recall plateaus as γ grows. The summary's "max recall" line shows
# where each capacity tops out.
#
# LIMIT caps pops for every point and is set high so that neither rule stops on
# it, even at width 1983; the summary warns if any query did (capped_frac > 0).
# The pops run keeps a visited log of n_queries x (LIMIT+1) x 8 bytes, on the
# device and again on the host (~650 MB each for 10k queries at 8192).
#
# ── Outputs, under $OUT ────────────────────────────────────────────────────
#   raw/<ds>_<method>_k<k>_<placement>.qps.log    run_query output
#   raw/<ds>_<method>_k<k>_<placement>.pops.csv   distortion_test --pops
#                                                 (stop,gamma columns)
#   raw/<ds>_<method>_k<k>_<placement>.pops.log   its console output
#   adaptive_<ds>.csv     both joined, one row per (placement, point)
#   adaptive_<ds>.txt     per placement and target recall: best QPS and fewest
#                         pops under each rule, and the ratios
#                         (scripts/adaptive_summary.py)
#
# Plain indexes are shared with distortion_sweep.sbatch / profile_pcie.sbatch
# ($WORK/<ds>_r64_64_plain.bin); a missing one is built in one part. L2 only:
# distortion_test compiles L2 configs only, and every dataset below is L2.
#
#   DRY_RUN=1 ./sbatch/bench_adaptive_termination.sbatch     # plan, no GPU
#   sbatch sbatch/bench_adaptive_termination.sbatch
#   DATASETS="deep10M sra50" METHODS=lsh K_LIST="4 16" sbatch sbatch/bench_adaptive_termination.sbatch
#   CAPACITIES=256 GAMMAS="0 0.05 0.1 0.2" sbatch sbatch/bench_adaptive_termination.sbatch
#   QUERY_BATCH=100000 sbatch sbatch/bench_adaptive_termination.sbatch  # tile for QPS
#   PLACEMENTS=host RUN_POPS=0 sbatch sbatch/bench_adaptive_termination.sbatch
#   SUMMARY_ONLY=1 ./sbatch/bench_adaptive_termination.sbatch   # re-join raw/, no GPU

set -euo pipefail

# ── Tunables ───────────────────────────────────────────────────────────────
DATASETS="${DATASETS:-deep10M}"
METHODS="${METHODS:-lsh pq rabitq}"
K_LIST="${K_LIST:-8}"                  # 8 is valid for every method at every dim below
K="${K:-10}"
BEAM_WIDTHS="${BEAM_WIDTHS:-16 24 32 48 64 96 128 192 256 384 512 768 1024 1536 1983}"
CAPACITIES="${CAPACITIES:-$BEAM_WIDTHS}"  # adaptive beam_width (list cap) per gamma
GAMMAS="${GAMMAS:-0 0.02 0.04 0.06 0.08 0.1 0.12 0.15 0.2 0.25 0.3 0.4 0.5 0.75 1}"
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
DRY_RUN="${DRY_RUN:-0}"
SUMMARY_ONLY="${SUMMARY_ONLY:-0}"

ANN_DATA="${ANN_DATA:-/projects/SaltSystemsLab/ann_data}"
ANN_GT="${ANN_GT:-/projects/SaltSystemsLab/ann_gt}"
LOCALE="${LOCALE:-/tmp2/ann_data/locale}"
REPO="${REPO:-$HOME/jasper}"
WORK="${WORK:-/projects/SaltSystemsLab/${USER:-$(id -un)}/distortion}"
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

summarize() {                          # $1 dataset
  python3 -I "$REPO/scripts/adaptive_summary.py" "$RAW" --dataset "$1" \
      --out "$OUT/adaptive_$1.csv" --targets $TARGETS \
    | tee "$OUT/adaptive_$1.txt" \
    || echo "  SUMMARY FAILED for $1"
}

if [ "$SUMMARY_ONLY" != "0" ]; then
  for ds in $DATASETS; do summarize "$ds"; done
  exit 0
fi

# ── Points ─────────────────────────────────────────────────────────────────
# The searches dispatch on beam_width + 64 < 2048, so 1983 is the widest point;
# a width under K could not even hold the k results.
fits() { [ "$1" -ge "$K" ] && [ $(( $1 + 64 )) -lt 2048 ]; }
BEAM_POINTS=""
for bw in $BEAM_WIDTHS; do
  fits "$bw" || { echo "dropping beam width $bw (need $K <= width <= 1983)"; continue; }
  BEAM_POINTS="$BEAM_POINTS $bw:$LIMIT"
done
ADAPT_POINTS=""
for cap in $CAPACITIES; do
  fits "$cap" || { echo "dropping capacity $cap (need $K <= capacity <= 1983)"; continue; }
  for g in $GAMMAS; do ADAPT_POINTS="$ADAPT_POINTS $cap:$LIMIT:$g"; done
done
POINTS="${BEAM_POINTS# } ${ADAPT_POINTS# }"
POINTS="${POINTS# }"
[ -n "$POINTS" ] || { echo "no points to run" >&2; exit 2; }
# Whole passes rather than back-to-back repeats, so clock or thermal drift
# spreads over every point instead of landing on a few.
QPS_POINTS=""
for _ in $(seq "$REPEATS"); do QPS_POINTS="$QPS_POINTS $POINTS"; done
N_BEAM=$(wc -w <<<"$BEAM_POINTS" | tr -d ' ')
N_ADAPT=$(wc -w <<<"$ADAPT_POINTS" | tr -d ' ')

for place in $PLACEMENTS; do
  case "$place" in device|host) ;; *) echo "unknown placement: $place (device | host)" >&2; exit 2;; esac
done
N_PLACE=$(wc -w <<<"$PLACEMENTS" | tr -d ' ')
[ "$N_PLACE" -gt 0 ] || { echo "no placements" >&2; exit 2; }

# ── Plan ───────────────────────────────────────────────────────────────────
echo "=== beam-width vs adaptive termination ==="
echo "  host:       $(hostname)"
echo "  datasets:   $DATASETS"
echo "  methods:    $METHODS    k_ranks: $K_LIST    k: $K"
echo "  beam:       $N_BEAM points, widths $BEAM_WIDTHS"
echo "  adaptive:   $N_ADAPT points, capacities $CAPACITIES x gammas $GAMMAS"
echo "  limit:      $LIMIT pops    qps passes: $REPEATS    query batch: $([ "$QUERY_BATCH" = 0 ] && echo untiled || echo "$QUERY_BATCH")"
echo "  placements: $PLACEMENTS"
echo "  steps:      $([ "$RUN_QPS" = 1 ] && echo 'run_query ')$([ "$RUN_POPS" = 1 ] && echo 'distortion_test --pops')"
echo "  graph:      R_int=$R_INT R_fin=$R_FIN dist=l2"
echo "  out:        $OUT"
echo
CELLS=""
for ds in $DATASETS; do
  spec=$(dataset_spec "$ds") || { echo "unknown dataset: $ds" >&2; exit 2; }
  IFS='|' read -r base q gt dtype dim <<<"$spec"
  padded=$(( (dim + 7) / 8 * 8 ))
  ivar="INDEX_$ds"; idx="${!ivar:-$WORK/${ds}_r${R_INT}_${R_FIN}_plain.bin}"
  printf "  %-10s dim=%-5s %-6s index=%s%s\n" "$ds" "$dim" "$dtype" "$idx" \
    "$([ -s "$idx" ] && echo ' (exists)' || echo ' (will build)')"
  [ -s "$idx" ] || [ -r "$base" ] || echo "    WARNING: no index and no base at $base"
  [ -r "$q" ]  || echo "    WARNING: queries missing: $q"
  [ -r "$gt" ] || { echo "    SKIP: no ground truth at $gt (recall is the comparison axis)"; continue; }
  if [ "$dim" -gt 128 ] && [ "$R_FIN" != "64" ]; then
    echo "    ERROR: dim > 128 needs R_FIN=64 (uint16-packed configs)" >&2; exit 2
  fi
  for m in $METHODS; do
    case "$m" in lsh|pq|rabitq) ;; *) echo "unknown method: $m (lsh | pq | rabitq)" >&2; exit 2;; esac
    for k in $K_LIST; do
      # The compiled config tables (run_query, distortion_test) stop here.
      case "$k" in 4|8|16) ;; *) echo "    $m k$k: SKIP (k_ranks must be 4, 8 or 16)"; continue;; esac
      if [ "$m" = "pq" ] && [ $(( padded % k )) -ne 0 ]; then
        echo "    $m k$k: SKIP (padded_dim=$padded not divisible by $k)"; continue
      fi
      [ "$m" = "rabitq" ] && [ $(( 8 * k )) -gt "$padded" ] && \
        echo "    $m k$k: note — rq_dim=$(( 8 * k )) > dim, codes all $padded dims (no projection)"
      CELLS="$CELLS $ds:$m:$k"
    done
  done
done
CELLS="${CELLS# }"
echo
[ -n "$CELLS" ] || { echo "No runnable cells." >&2; exit 1; }
echo "  cells: $(wc -w <<<"$CELLS" | tr -d ' ') x $N_PLACE placement(s), each" \
     "$(( N_BEAM + N_ADAPT )) points ($(( (N_BEAM + N_ADAPT) * REPEATS )) timed searches" \
     "+ $(( N_BEAM + N_ADAPT )) logged)"

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
cmake -B build -DJASPER_BUILD_CMD=ON -DJASPER_BUILD_TESTS=OFF -DJASPER_BUILD_FFI=OFF \
  -DThrust_DIR="$THRUST_DIR" -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" >/dev/null
cmake --build build --target create_diskann_index run_query distortion_test -j8 >/dev/null
echo "  ok"
mkdir -p "$WORK" "$RAW"

# ── Sweep ──────────────────────────────────────────────────────────────────
DONE_DS=""
for ds in $DATASETS; do
  case " $CELLS " in *" $ds:"*) ;; *) continue;; esac
  IFS='|' read -r base q gt dtype dim <<<"$(dataset_spec "$ds")"
  ivar="INDEX_$ds"; idx="${!ivar:-$WORK/${ds}_r${R_INT}_${R_FIN}_plain.bin}"

  echo
  echo "######## $ds ########"
  if [ -s "$idx" ]; then
    echo "[$(stamp)] reusing index $idx"
  else
    echo "[$(stamp)] building plain index -> $idx"
    ./build/bin/create_diskann_index \
        -f "$base" -t "$dtype" -d l2 --dim "$dim" \
        --r_intermediate "$R_INT" --r_final "$R_FIN" \
        --n_parts "$N_PARTS" --n_dup "$N_DUP" -a "$ALPHA" \
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
    IFS=':' read -r cds m k <<<"$cell"
    [ "$cds" = "$ds" ] || continue
    for place in $PLACEMENTS; do
      host_arg=()
      [ "$place" = "host" ] && host_arg=(--on_host)
      name="${ds}_${m}_k${k}_${place}"
      echo "[$(stamp)] $ds $m k$k on $place"

      if [ "$RUN_QPS" = "1" ]; then
        # No --prepopulated: codes are built at load from the plain index, the
        # same way distortion_test builds them below.
        ./build/bin/run_query -i "$idx" --dim "$dim" -d l2 -t "$dtype" -n "$R_FIN" -k "$K" \
            "--$m" --k_ranks "$k" ${host_arg[@]+"${host_arg[@]}"} \
            -q "$qq" -g "$qg" --beam_limits $QPS_POINTS >| "$RAW/$name.qps.log" 2>&1 \
          || echo "  QPS FAILED ($name) — see $RAW/$name.qps.log"
        echo "  qps:  $(grep -c 'throughput=' "$RAW/$name.qps.log" || true) timed searches -> raw/$name.qps.log"
      fi

      if [ "$RUN_POPS" = "1" ]; then
        ./build/bin/distortion_test -i "$idx" --dim "$dim" -t "$dtype" -n "$R_FIN" -k "$K" \
            -m "$m" -r "$k" ${host_arg[@]+"${host_arg[@]}"} \
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
echo "raw outputs in $RAW; joined CSV + matched-recall report in $OUT/adaptive_<dataset>.{csv,txt}"
exit 0
