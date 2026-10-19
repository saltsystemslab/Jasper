#!/bin/bash
#SBATCH --partition=SaltSystemsLab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --nodelist=d4067
#SBATCH --mem=987842M
#SBATCH --gres=gpu:1
#SBATCH --job-name=deep_ksweep
#SBATCH --output=deep_ksweep_%j.log
#SBATCH --error=deep_ksweep_%j.err
#SBATCH --time=24:00:00
#
# k_ranks sweep for every quantization method, on one DEEP index.
#
#   estimators: lsh, pq, rabitq        k_ranks: 4, 8, 16
#
# k_ranks is the one knob all three share — LSH rank count, PQ subquantizer
# count M, and RaBitQ projected dim / 8 — so sweeping it says how each method
# spends a per-edge byte budget, and where each stops paying for more.
#
# ── Why this builds ONE index ──────────────────────────────────────────────
# run_query WITHOUT --prepopulated builds the codes at load time: it rotates
# the stored vectors, trains whatever model the estimator needs, populates the
# per-edge codes, and only then searches. So a plain index (no trailer) serves
# every (estimator, k_ranks) pair, and this script builds it once.
#
# That is not just cheaper than nine builds. It is the only version of this
# experiment that answers the question: all nine points then run against a
# BIT-IDENTICAL graph, so a difference in recall is the estimator and nothing
# else. Nine separate create_diskann_index runs would each derive their own
# shard count from their own payload size (see bench_deep_scaling.sbatch), and
# a different shard count is a different graph.
#
# It also sidesteps a table mismatch: at dim 96, run_query compiles all six
# (R, k_ranks) combinations, while create_diskann_index has only k_ranks=4 at
# R=64. Populating at load means only run_query's table has to cover the sweep.
#
# ── Read before launching ──────────────────────────────────────────────────
#   DRY_RUN=1 ./sbatch/bench_deep_ksweep.sbatch     # plan + per-cell budget
#   sbatch sbatch/bench_deep_ksweep.sbatch
#   K_LIST="4 16" METHODS="pq rq" sbatch sbatch/bench_deep_ksweep.sbatch
#   INDEX=/path/to/plain.bin sbatch sbatch/bench_deep_ksweep.sbatch
#   SCALE=1B R_INT=32 R_FIN=32 RUN_POPS=1 sbatch sbatch/bench_deep_ksweep.sbatch
#
# RUN_POPS=1 also records nodes popped per query (recall vs nodes explored)
# for every estimator cell: see run_pops below.
#
# ⚠ THROUGHPUT COMPARABILITY. A loaded index that exceeds the card is searched
# over PCIe (--on_host), which is far slower per hop. At 100M and R_fin=64 the
# bigger payloads cross that line while k_ranks=4 does not, so the QPS column
# would be measuring the storage path rather than the estimator. The plan table
# below flags every cell that lands on the host. If any do, either accept that
# recall is still comparable and QPS is not, or drop to R=32 (halves every
# payload, keeps the whole sweep device-resident):
#
#   R_INT=32 R_FIN=32 sbatch sbatch/bench_deep_ksweep.sbatch
#
# ⚠ Each cell reloads the index and re-populates its codes. That is the
# dominant per-cell cost, not the search.

set -euo pipefail

# ── Tunables ───────────────────────────────────────────────────────────────
SCALE="${SCALE:-100M}"
METHODS="${METHODS:-lsh pq rq}"        # any subset, in any order
K_LIST="${K_LIST:-4 8 16}"
R_INT="${R_INT:-64}"
R_FIN="${R_FIN:-64}"
N_DUP="${N_DUP:-2}"
ALPHA="${ALPHA:-1.2}"
DIST="${DIST:-l2}"                     # DEEP is an L2 benchmark
WORKSPACE_BUDGET="${WORKSPACE_BUDGET:-10GB}"
DEVICE_FREE_MIB="${DEVICE_FREE_MIB:-0}"
SHARD_FILL="${SHARD_FILL:-70}"
N_PARTS="${N_PARTS:-0}"
K="${K:-10}"                           # search k; one value keeps the grid small
# Same tail as bench_locale / bench_deep_scaling. The searches dispatch on
# beam_width + 64 < 2048 (largest compiled search width), so 1982 is the top.
BEAM_LIMITS="${BEAM_LIMITS:-16:128 32:128 64:128 128:256 256:512 512:1024 1024:2048 1536:2048 1982:2048}"
QUERY_BATCH="${QUERY_BATCH:-0}"        # tile queries for a saturated QPS number
ON_HOST="${ON_HOST:-auto}"             # auto | 1 | 0
RUN_BASELINE="${RUN_BASELINE:-1}"      # also run conventional search once
RUN_POPS="${RUN_POPS:-0}"              # also run distortion_test --pops per cell
DRY_RUN="${DRY_RUN:-0}"
DISK_BUDGET="${DISK_BUDGET:-0}"        # GiB, 0 = ask df

DATA="${DATA:-/projects/SaltSystemsLab/ann_data/deep}"
GT="${GT:-/projects/SaltSystemsLab/ann_gt}"
QUERIES="${QUERIES:-$DATA/deep10kquery}"
# /tmp is node-local and cleared at job end, so the index cannot outlive this
# job — which is fine, because the whole sweep runs inside it.
WORK="${WORK:-/tmp/${USER:-$(id -un)}/deep_ksweep}"
REPO="${REPO:-$HOME/jasper}"
LOGDIR="${LOGDIR:-$REPO/deep_ksweep_logs}"
POPS_DIR="${POPS_DIR:-$LOGDIR/pops}"   # <dataset>_<method>_k<k>.csv (plot_pops.py)
INDEX="${INDEX:-}"                     # reuse an existing PLAIN index

DIM=96
SRC_DTYPE=float
PADDED=$(( (DIM + 7) / 8 * 8 ))
PACKED_BYTES=$([ "$DIM" -le 128 ] && echo 1 || echo 2)
PACKED_T=$([ "$DIM" -le 128 ] && echo uint8_t || echo uint16_t)
FUNC_TOKEN=$([ "$DIST" = "ip" ] && echo INNER_PRODUCT || echo L2)

scale_n()    { case "$1" in 10M) echo 10000000;; 100M) echo 100000000;;
                            200M) echo 200000000;; 1B) echo 1000000000;;
                            *) echo 0;; esac; }
scale_base() { case "$1" in 1B) echo "$DATA/base.1B.fbin";; *) echo "$DATA/deep$1";; esac; }
scale_gt()   { case "$1" in 1B) echo "$GT/GT_1B/deep-1B";; *) echo "$GT/GT_$1/deep-$1";; esac; }

gib()    { awk -v b="$1" 'BEGIN{printf "%.1f", b/1073741824}'; }
mnodes() { awk -v n="$1" 'BEGIN{printf "%.1fM", n/1000000}'; }

parse_size() {
  awk -v s="$1" 'BEGIN{ n=s+0; u=toupper(s); sub(/^[0-9.]+/,"",u); m=1;
    if(u=="KB")m=1024; else if(u=="MB")m=1024^2; else if(u=="GB")m=1024^3;
    else if(u=="TB")m=1024^4; printf "%d", n*m; }'
}
resolve_device_free() {
  if [ "$DEVICE_FREE_MIB" != "0" ]; then echo $(( DEVICE_FREE_MIB * 1048576 )); return; fi
  local mib
  mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null \
        | head -1 | tr -d ' ') || mib=""
  [ -n "$mib" ] || mib=96683
  echo $(( mib * 1048576 ))
}
mem_available_bytes() {
  if [ -n "${SLURM_MEM_PER_NODE:-}" ]; then echo $(( SLURM_MEM_PER_NODE * 1048576 )); return; fi
  awk '/^MemAvailable:/ {print $2 * 1024; exit}' /proc/meminfo 2>/dev/null || echo 0
}
disk_avail_bytes() {
  local v; v=$(df -B1 --output=avail "$1" 2>/dev/null | tail -1 | tr -d ' ')
  [ -n "$v" ] || v=0; echo "$v"
}

# ── Per-edge payload, by method and k ──────────────────────────────────────
# Mirrors the structs: edge_lsh_list (packed words + bf16 mag²), edge_pq_list
# (one byte per subquantizer), edge_rq_list (32-bit code words + fp16 rescale
# + bf16 mag², over rq_dim = 8*k dimensions at 1 bit each).
payload_per_node() {       # $1 = method, $2 = k, $3 = degree
  local m=$1 k=$2 r=$3
  case "$m" in
    lsh) echo $(( r * k * PACKED_BYTES + r * 2 ));;
    pq)  echo $(( r * k + 4 ));;                       # codes + exact ||v||²
    rq)  echo $(( r * (4 * (((8 * k) + 31) / 32) + 4) ));;
    *)   echo 0;;
  esac
}
# What run_query holds resident per node once the codes are populated.
resident_per_node() {      # $1 = method, $2 = k
  echo $(( PADDED * 2 + R_FIN * 6 + 1 + 4 + $(payload_per_node "$1" "$2" "$R_FIN") ))
}

# ── Compiled-config preflight ──────────────────────────────────────────────
# run_query's directional table is fixed at compile time; an uncompiled cell
# throws "Unsupported config" only after the index is loaded. Read the table
# out of the source so this cannot drift from what is actually built.
query_config_supported() {   # $1 = k_ranks
  local f="$REPO/cmd/run_query.cu"
  [ -r "$f" ] || return 2
  sed 's/[[:space:]]//g' "$f" \
    | grep -q -- "uint32_t,${R_FIN},__half,float,jasper::distance_func::${FUNC_TOKEN},$1,${PACKED_T}"
}

N=$(scale_n "$SCALE")
[ "$N" != "0" ] || { echo "unknown SCALE: $SCALE" >&2; exit 2; }
BASE=$(scale_base "$SCALE")
GTFILE=$(scale_gt "$SCALE")

echo "=== deep k_ranks sweep ==="
echo "  host:        $(hostname)"
echo "  scale:       $SCALE ($N vectors, dim $DIM, packed_t=$PACKED_T)"
echo "  methods:     $METHODS"
echo "  k_ranks:     $K_LIST"
echo "  degrees:     R_int=$R_INT R_fin=$R_FIN  n_dup=$N_DUP  alpha=$ALPHA  dist=$DIST"
echo "  search k:    $K"
if [ "$RUN_POPS" = "1" ] && [ "$DIST" != "l2" ]; then
  echo "  pops:        disabled — distortion_test compiles L2 configs only"
  RUN_POPS=0
fi
[ "$RUN_POPS" = "1" ] && echo "  pops:        distortion_test --pops per cell -> $POPS_DIR"
echo "  workdir:     $WORK"
echo "  logs:        $LOGDIR"

mkdir -p "$WORK" "$LOGDIR"
MEM_AVAIL=$(mem_available_bytes)
DISK_AVAIL=$(disk_avail_bytes "$WORK")
[ "$DISK_BUDGET" = "0" ] || DISK_AVAIL=$(( DISK_BUDGET * 1073741824 ))
DEVICE_FREE_BYTES=$(resolve_device_free)
WORKSPACE_BYTES=$(parse_size "$WORKSPACE_BUDGET")

echo "  host RAM:    $(gib "$MEM_AVAIL") GiB"
echo "  disk:        $(gib "$DISK_AVAIL") GiB at $WORK"
echo "  device:      $(gib "$DEVICE_FREE_BYTES") GiB free"
echo

# ── The plain index ────────────────────────────────────────────────────────
# No estimator flags, so no prerotation either: run_query rotates at load and
# populates from there. Its name carries only what changes the GRAPH.
IDX="${INDEX:-$WORK/deep${SCALE}_r${R_INT}_${R_FIN}_plain.bin}"

# Plain build: no directional payload on host or device.
BUILD_PER_VEC=$(( PADDED * 2 + R_FIN * 6 + 1 + N_DUP * 4 ))
DEVICE_PER_SLOT=$(( PADDED * 2 + R_INT * 6 + 1 + 4 + 1 ))
DISK_PER_VEC=$(( DIM * 2 + 1 + R_FIN * 6 ))

max_shard() {
  local per_seg avail n_seg
  per_seg=$(( DEVICE_PER_SLOT * 1048576 ))
  avail=$(( DEVICE_FREE_BYTES - WORKSPACE_BYTES - 4 * 1073741824 ))
  [ "$avail" -lt 0 ] && { echo 0; return; }
  n_seg=$(( avail / per_seg ))
  echo $(( n_seg * 1048576 ))
}
MAX_SHARD=$(max_shard)
[ "$MAX_SHARD" -gt 0 ] || { echo "Nothing fits on the device." >&2; exit 1; }

n_parts_for() {
  local n=$1 cap=$(( 2 * DIM )) target p
  if [ "$N_PARTS" != "0" ]; then echo "$N_PARTS"; return; fi
  target=$(( MAX_SHARD * SHARD_FILL / 100 ))
  [ "$target" -lt 1 ] && { echo 0; return; }
  # A shard never holds more than n distinct vectors: the n_dup duplicates
  # collapse when they land together, so n*N_DUP only binds once it is spread.
  if [ "$n" -le "$target" ]; then echo 1; return; fi
  p=$(( (n * N_DUP + target - 1) / target ))
  [ "$p" -lt 1 ] && p=1
  [ "$p" -gt "$cap" ] && p=$cap
  echo "$p"
}
PARTS=$(n_parts_for "$N")

BUILD_RAM=$(( N * BUILD_PER_VEC ))
BUILD_DISK=$(( N * DISK_PER_VEC ))
echo "  plain index: $IDX"
printf "    %s parts, %s/shard, %s GiB host, %s GiB disk\n" \
  "$PARTS" "$(mnodes $(( N * N_DUP / (PARTS > 0 ? PARTS : 1) > N ? N : N * N_DUP / (PARTS > 0 ? PARTS : 1) )))" \
  "$(gib "$BUILD_RAM")" "$(gib "$BUILD_DISK")"
[ "$BUILD_RAM" -le "$MEM_AVAIL" ] || echo "    WARNING: build needs more RAM than this job has"
[ "$BUILD_DISK" -le "$DISK_AVAIL" ] || echo "    WARNING: build needs more disk than $WORK has"
echo

# ── Plan table ─────────────────────────────────────────────────────────────
printf "  %-4s %-4s %9s %10s %10s %s\n" method k B/edge B/node resident where
CELLS=""
ANY_HOST=0
ANY_DEVICE=0
for m in $METHODS; do
  case "$m" in lsh|pq|rq) ;; *) echo "  unknown method: $m" >&2; exit 2;; esac
  for k in $K_LIST; do
    rc=0; query_config_supported "$k" || rc=$?
    if [ "$rc" = "1" ]; then
      printf "  %-4s %-4s %9s %10s %10s %s\n" "$m" "$k" - - - \
        "SKIP: run_query has no R=$R_FIN k_ranks=$k $DIST $PACKED_T config"
      continue
    fi
    # rq_dim = 8k wider than the data is allowed, but there is nothing to
    # project onto: it degenerates to plain 1-bit RaBitQ over all padded_dim
    # coordinates, with the surplus bits stored at zero. Same recall as the
    # largest k that fits, at strictly more bytes — so flag it rather than
    # letting the B/edge column imply the extra bits bought something.
    rq_note=""
    if [ "$m" = "rq" ] && [ $(( 8 * k )) -gt "$PADDED" ]; then
      rq_note="(no projection: codes $PADDED dims, $(( 8 * k - PADDED )) bits/edge idle)"
    fi
    # PQ splits padded_dim into k subspaces; it must divide evenly.
    if [ "$m" = "pq" ] && [ $(( PADDED % k )) -ne 0 ]; then
      printf "  %-4s %-4s %9s %10s %10s %s\n" "$m" "$k" - - - \
        "SKIP: padded_dim=$PADDED not divisible by pq_m=$k"
      continue
    fi
    pe=$(( $(payload_per_node "$m" "$k" "$R_FIN") ))
    pn=$(resident_per_node "$m" "$k")
    res=$(( N * pn ))
    where="device"
    case "$ON_HOST" in
      1|yes|true|on|force) where="host (forced)"; ANY_HOST=1;;
      0|no|false|off|device) where="device (forced)"; ANY_DEVICE=1;;
      *) if [ "$res" -gt "$DEVICE_FREE_BYTES" ]; then where="HOST (PCIe)"; ANY_HOST=1
         else ANY_DEVICE=1; fi;;
    esac
    printf "  %-4s %-4s %9s %10s %10s %s\n" \
      "$m" "$k" "$(( pe / R_FIN ))" "$pn" "$(gib "$res") GiB" "$where $rq_note"
    CELLS="$CELLS $m:$k"
  done
done
echo
CELLS="${CELLS# }"
[ -n "$CELLS" ] || { echo "No runnable cells." >&2; exit 1; }
echo "  cells: $(echo "$CELLS" | wc -w | tr -d ' ') run_query invocations, each reloading"
echo "         the index and re-populating its codes"
if [ "$ANY_HOST" = "1" ] && [ "$ANY_DEVICE" = "1" ]; then
  echo
  echo "  ⚠ THIS SWEEP STRADDLES THE DEVICE/HOST LINE. Some cells are searched on"
  echo "    the card and others over PCIe, so their QPS numbers are not comparable"
  echo "    to each other — recall still is. R_INT=32 R_FIN=32 halves every"
  echo "    payload and keeps the whole grid device-resident."
fi

if [ "$DRY_RUN" != "0" ]; then
  echo
  echo "DRY_RUN set — nothing built."
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

echo "=== building binaries ==="
cmake -B build -DJASPER_BUILD_CMD=ON -DJASPER_BUILD_TESTS=OFF -DJASPER_BUILD_FFI=OFF \
  -DThrust_DIR="$THRUST_DIR" -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" >/dev/null
cmake --build build --target create_diskann_index run_query \
  $([ "$RUN_POPS" = "1" ] && echo distortion_test) -j8 >/dev/null
[ "$RUN_POPS" = "1" ] && mkdir -p "$POPS_DIR"
echo "  ok"
echo

stamp() { date '+%Y-%m-%d %H:%M:%S'; }
BUILD_LOG="$LOGDIR/deep${SCALE}_r${R_INT}_${R_FIN}_plain_build.log"

if [ -s "$IDX" ]; then
  echo "[$( stamp )] plain index exists, reusing: $IDX ($(gib "$(stat -c %s "$IDX")") GiB)"
else
  [ -r "$BASE" ] || { echo "base missing: $BASE" >&2; exit 1; }
  echo "[$( stamp )] building plain index -> $IDX"
  # No --lsh/--pq/--rabitq: a bare graph, which run_query then rotates and
  # populates per cell.
  /usr/bin/time -v ./build/bin/create_diskann_index \
      -f "$BASE" -t "$SRC_DTYPE" -d "$DIST" --dim "$DIM" \
      --r_intermediate "$R_INT" --r_final "$R_FIN" \
      --n_parts "$PARTS" --n_dup "$N_DUP" -a "$ALPHA" \
      -w "$WORKSPACE_BUDGET" -i "$IDX" 2>&1 | tee -a "$BUILD_LOG" \
    || { echo "BUILD FAILED (see $BUILD_LOG)" >&2; exit 1; }
  echo "[$( stamp )] built, $(gib "$(stat -c %s "$IDX")") GiB"
fi
echo

# ── Queries ────────────────────────────────────────────────────────────────
[ -r "$QUERIES" ] || { echo "queries missing: $QUERIES" >&2; exit 1; }
Q="$QUERIES"
G="$GTFILE"
gt_arg=()
if [ -r "$G" ]; then gt_arg=(-g "$G"); else
  echo "NOTE: no ground truth at $G — latency only."
fi

if [ "$QUERY_BATCH" != "0" ]; then
  TQ="$WORK/deep_x${QUERY_BATCH}_query.fbin"
  TG="$WORK/deep_x${QUERY_BATCH}_gt.ibin"
  if [ ! -s "$TQ" ]; then
    tile_gt_arg=()
    [ -r "$G" ] && tile_gt_arg=(--gt "$G" --out-gt "$TG")
    python3 "$REPO/scripts/tile_queries.py" --target "$QUERY_BATCH" \
        --query "$Q" --out-query "$TQ" ${tile_gt_arg[@]+"${tile_gt_arg[@]}"} \
      || { echo "tiling failed; using the untiled query set"; TQ=""; }
  fi
  if [ -s "$TQ" ]; then
    Q="$TQ"
    [ -s "$TG" ] && { G="$TG"; gt_arg=(-g "$TG"); }
  fi
fi

# ── Nodes popped per query (RUN_POPS=1) ────────────────────────────────────
# run_query reports throughput, not how many nodes each search popped. With
# RUN_POPS=1 every estimator cell is followed by distortion_test --pops on the
# SAME plain index, with the same --on_host decision, queries, ground truth and
# beam:limit pairs, so the QPS curve and the nodes-explored curve come from one
# graph and one build. Costs a second index load + code build per cell.
run_pops() {                       # $1 method, $2 k, $3 log, then --on_host if any
  local m=$1 k=$2 log=$3; shift 3
  local dm=$m; [ "$m" = "rq" ] && dm=rabitq
  local out="$POPS_DIR/deep${SCALE}_${dm}_k${k}.csv"
  [ -r "$G" ] || { echo "  pops: skipped (no ground truth)"; return 0; }
  echo "[$( stamp )] pops $dm k$k -> $out"
  ./build/bin/distortion_test -i "$IDX" --dim "$DIM" -t "$SRC_DTYPE" -n "$R_FIN" \
      -m "$dm" -r "$k" -q "$Q" -g "$G" -k "$K" "$@" --pops "$out" -b $BEAM_LIMITS 2>&1 \
    | tee -a "$log" \
    | grep -E "mean_pops" \
    || echo "  POPS FAILED ($dm k$k) — see $log"
}

run_cell() {                       # $1 = method, $2 = k_ranks ("" = baseline)
  local m=$1 k=${2:-}
  local mode_arg=() label host_arg=() res
  case "$m" in
    conventional) mode_arg=(); label="conventional"; res=$(( N * (PADDED * 2 + R_FIN * 6 + 1 + 4) ));;
    lsh) mode_arg=(--lsh    --k_ranks "$k"); label="lsh_k$k";  res=$(( N * $(resident_per_node lsh "$k") ));;
    pq)  mode_arg=(--pq     --k_ranks "$k"); label="pq_k$k";   res=$(( N * $(resident_per_node pq  "$k") ));;
    rq)  mode_arg=(--rabitq --k_ranks "$k"); label="rq_k$k";   res=$(( N * $(resident_per_node rq  "$k") ));;
  esac

  case "$ON_HOST" in
    1|yes|true|on|force)   host_arg=(--on_host);;
    0|no|false|off|device) ;;
    *) [ "$res" -gt "$DEVICE_FREE_BYTES" ] && host_arg=(--on_host);;
  esac

  local log="$LOGDIR/deep${SCALE}_r${R_INT}_${R_FIN}_${label}.log"
  echo "========== $label ($(gib "$res") GiB resident${host_arg[0]+, --on_host}) =========="
  echo "[$( stamp )] $label"
  # No --prepopulated: the codes are built at load from the plain index, which
  # is what lets every cell share one graph.
  ./build/bin/run_query -i "$IDX" --dim "$DIM" -d "$DIST" -t "$SRC_DTYPE" \
      -n "$R_FIN" -k "$K" ${mode_arg[@]+"${mode_arg[@]}"} ${host_arg[@]+"${host_arg[@]}"} \
      -q "$Q" ${gt_arg[@]+"${gt_arg[@]}"} --beam_limits $BEAM_LIMITS 2>&1 \
    | tee -a "$log" \
    | grep -E "beam_width|Recall|populate|generate|Rotating|Training" \
    || echo "  CELL FAILED ($label) — see $log"
  if [ "$RUN_POPS" = "1" ] && [ "$m" != "conventional" ]; then
    run_pops "$m" "$k" "$log" ${host_arg[@]+"${host_arg[@]}"}
  fi
  echo
}

if [ "$RUN_BASELINE" = "1" ]; then
  # One conventional run: the ceiling every estimator is a fraction of.
  run_cell conventional
fi
for cell in $CELLS; do
  run_cell "${cell%%:*}" "${cell##*:}"
done

echo "=== done: $(stamp) ==="
echo "Per-cell logs in $LOGDIR/deep${SCALE}_r${R_INT}_${R_FIN}_<method>_k<k>.log"
[ "$RUN_POPS" = "1" ] && echo "Nodes-popped CSVs in $POPS_DIR/deep${SCALE}_<method>_k<k>.csv" || true
