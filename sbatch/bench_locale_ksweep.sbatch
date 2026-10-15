#!/bin/bash
#SBATCH --partition=SaltSystemsLab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --nodelist=d4067
#SBATCH --mem=987842M
#SBATCH --gres=gpu:1
#SBATCH --job-name=locale_ksweep
#SBATCH --output=locale_ksweep_%j.log
#SBATCH --error=locale_ksweep_%j.err
#SBATCH --time=24:00:00
#
# k_ranks sweep for every quantization method, on one LOCALE index.
#
#   estimators: lsh, pq, rabitq        k_ranks: 4, 8, 16
#
# The sibling of bench_deep_ksweep.sbatch, for the SRA genomic sketch sets.
# Same shape, and worth running for a reason the DEEP sweep cannot cover: at
# dim 768 every cell actually projects (rq_dim = 8k <= 128 < 768) and all nine
# fit on the card, so this is the complete grid at one throughput regime. On
# DEEP (dim 96) RaBitQ's k16 runs but degenerates — rq_dim = 128 outruns the
# data, so it codes all 96 dims and idles 32 bits/edge — and at R=64 most
# cells spill to PCIe.
#
# ── Why this builds ONE index ──────────────────────────────────────────────
# run_query WITHOUT --prepopulated builds the codes at load: it rotates the
# stored vectors, trains whatever model the estimator needs, populates the
# per-edge codes, then searches. So a plain index serves every (method,
# k_ranks) pair, and all cells run against a BIT-IDENTICAL graph — which is the
# only way the comparison means anything. Nine create_diskann_index runs would
# each derive their own shard count from their own payload size, and a
# different shard count is a different graph.
#
# It also sidesteps the build table, which at R=64 has k_ranks=4 for dim <= 128
# and only the one l2/k16 row above it. Populating at load means only
# run_query's table has to cover the sweep.
#
# ── R is fixed at 64 here ──────────────────────────────────────────────────
# Unlike the DEEP sweep, R=32 is NOT an option: at dim > 128 the coordinate
# index needs 15 bits, so the configs are uint16-packed, and run_query compiles
# those only at R=64. The script rejects anything else rather than letting it
# fail after the build. That is fine for this dataset — every cell fits on the
# card at R=64 anyway.
#
# ── Read before launching ──────────────────────────────────────────────────
#   DRY_RUN=1 ./sbatch/bench_locale_ksweep.sbatch     # plan + per-cell budget
#   sbatch sbatch/bench_locale_ksweep.sbatch
#   METHODS="pq rq" K_LIST="4 16" sbatch sbatch/bench_locale_ksweep.sbatch
#   MUTATIONS="0.00 0.05 0.10" sbatch sbatch/bench_locale_ksweep.sbatch
#   DATASET=sra500 sbatch sbatch/bench_locale_ksweep.sbatch
#   DATASET=sra500 RUN_POPS=1 sbatch sbatch/bench_locale_ksweep.sbatch
#
# RUN_POPS=1 also records nodes popped per query (recall vs nodes explored)
# for every estimator cell: see run_pops below.
#
# Queries are tiled to QUERY_BATCH by default: the locale sets are ~500, the
# beam search runs one block per query, and 500 blocks leaves the GPU idle
# enough that the QPS column measures launch overhead. Tiling repeats the batch
# a whole number of times, so recall is unchanged (scripts/tile_queries.py).

set -euo pipefail

# ── Tunables ───────────────────────────────────────────────────────────────
DATASET="${DATASET:-sra50}"
METHODS="${METHODS:-lsh pq rq}"        # any subset, in any order
K_LIST="${K_LIST:-4 8 16}"
MUTATIONS="${MUTATIONS:-0.00}"         # add 0.05 0.10 to sweep drift as well
R_INT="${R_INT:-64}"                   # see "R is fixed at 64" above
R_FIN="${R_FIN:-64}"
N_DUP="${N_DUP:-2}"
ALPHA="${ALPHA:-1.2}"
DIST="${DIST:-l2}"
WORKSPACE_BUDGET="${WORKSPACE_BUDGET:-10GB}"
DEVICE_FREE_MIB="${DEVICE_FREE_MIB:-0}"
SHARD_FILL="${SHARD_FILL:-70}"
N_PARTS="${N_PARTS:-0}"
K="${K:-10}"                           # search k; the gt files carry k=100
BEAM_LIMITS="${BEAM_LIMITS:-16:128 32:128 64:128 128:256 256:512 512:1024 1024:2048}"
QUERY_BATCH="${QUERY_BATCH:-10000}"    # 0 disables tiling
ON_HOST="${ON_HOST:-auto}"             # auto | 1 | 0
RUN_BASELINE="${RUN_BASELINE:-1}"      # also run conventional search once
RUN_POPS="${RUN_POPS:-0}"              # also run distortion_test --pops per cell
DRY_RUN="${DRY_RUN:-0}"
DISK_BUDGET="${DISK_BUDGET:-0}"        # GiB, 0 = ask df

# sra500 is converted from the Hugging Face bundle in /scratch by
# scripts/locale_bundle_to_fbin.py (the old /tmp2 copy is gone); sra50 still
# lives in /tmp2. Either way the layout is $DATA/$DATASET/{base,query_mut*,gt_mut*}.
if [ "$DATASET" = "sra500" ]; then
  DATA="${DATA:-/scratch/${USER:-$(id -un)}/locale}"
else
  DATA="${DATA:-/tmp2/ann_data/locale}"
fi
# /tmp is node-local and cleared at job end, so the index cannot outlive this
# job — fine, because the whole sweep runs inside it.
WORK="${WORK:-/tmp/${USER:-$(id -un)}/locale_ksweep}"
REPO="${REPO:-$HOME/jasper}"
LOGDIR="${LOGDIR:-$REPO/locale_ksweep_logs}"
POPS_DIR="${POPS_DIR:-$LOGDIR/pops}"   # <dataset>_<method>_k<k>.csv (plot_pops.py)
INDEX="${INDEX:-}"                     # reuse an existing PLAIN index

SRC_DTYPE=float

ds_base()  { echo "$DATA/$DATASET/base.fbin"; }
ds_query() { echo "$DATA/$DATASET/query_mut$1.fbin"; }
ds_gt()    { echo "$DATA/$DATASET/gt_mut$1.ibin"; }

fbin_u32() { od -An -tu4 -N4 -j"$2" "$1" 2>/dev/null | tr -d ' \n'; }
fbin_n()   { fbin_u32 "$1" 0; }
fbin_dim() { fbin_u32 "$1" 4; }

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

# ── Probe the dataset ──────────────────────────────────────────────────────
BASE=$(ds_base)
[ -r "$BASE" ] || { echo "no base at $BASE" >&2; exit 1; }
N=$(fbin_n "$BASE")
DIM=$(fbin_dim "$BASE")
[ -n "$N" ] && [ "$N" -gt 0 ] 2>/dev/null || { echo "bad header in $BASE" >&2; exit 2; }
PADDED=$(( (DIM + 7) / 8 * 8 ))
PACKED_BYTES=$([ "$DIM" -le 128 ] && echo 1 || echo 2)
PACKED_T=$([ "$DIM" -le 128 ] && echo uint8_t || echo uint16_t)
FUNC_TOKEN=$([ "$DIST" = "ip" ] && echo INNER_PRODUCT || echo L2)

# ── Per-edge payload, by method and k ──────────────────────────────────────
# Mirrors the structs: edge_lsh_list (packed words + bf16 mag²), edge_pq_list
# (one byte per subquantizer, plus the exact ||v||² the L2 estimator needs),
# edge_rq_list (32-bit code words + fp16 rescale + bf16 mag², over
# rq_dim = 8*k dimensions at 1 bit each).
payload_per_node() {       # $1 = method, $2 = k, $3 = degree
  local m=$1 k=$2 r=$3
  case "$m" in
    lsh) echo $(( r * k * PACKED_BYTES + r * 2 ));;
    pq)  echo $(( r * k + 4 ));;
    rq)  echo $(( r * (4 * (((8 * k) + 31) / 32) + 4) ));;
    *)   echo 0;;
  esac
}
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

echo "=== locale k_ranks sweep ==="
echo "  host:        $(hostname)"
echo "  dataset:     $DATASET ($N vectors, dim $DIM probed, packed_t=$PACKED_T)"
echo "  methods:     $METHODS"
echo "  k_ranks:     $K_LIST"
echo "  mutations:   $MUTATIONS"
echo "  degrees:     R_int=$R_INT R_fin=$R_FIN  n_dup=$N_DUP  alpha=$ALPHA  dist=$DIST"
echo "  search k:    $K"
if [ "$RUN_POPS" = "1" ] && [ "$DIST" != "l2" ]; then
  echo "  pops:        disabled — distortion_test compiles L2 configs only"
  RUN_POPS=0
fi
[ "$RUN_POPS" = "1" ] && echo "  pops:        distortion_test --pops per cell -> $POPS_DIR"
[ "$QUERY_BATCH" = "0" ] || echo "  query batch: tiled to >= $QUERY_BATCH (recall unchanged)"
echo "  workdir:     $WORK"
echo "  logs:        $LOGDIR"

# At dim > 128 the packed coordinate needs 15 bits, and run_query compiles
# those uint16 configs only at R=64. Catch a wrong R now, not after the build.
if [ "$PACKED_T" = "uint16_t" ] && { [ "$R_FIN" != "64" ] || [ "$R_INT" != "64" ]; }; then
  echo >&2
  echo "R_int=$R_INT R_fin=$R_FIN will not work at dim=$DIM: the uint16-packed" >&2
  echo "configs run_query compiles for dim > 128 exist only at R=64. Drop the" >&2
  echo "R_INT/R_FIN overrides, or add the rows to run_query's table." >&2
  exit 2
fi

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
IDX="${INDEX:-$WORK/${DATASET}_r${R_INT}_${R_FIN}_plain.bin}"

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
SHARD=$(( N * N_DUP / PARTS )); [ "$SHARD" -gt "$N" ] && SHARD=$N

BUILD_RAM=$(( N * BUILD_PER_VEC ))
BUILD_DISK=$(( N * DISK_PER_VEC ))
echo "  plain index: $IDX"
printf "    %s parts, %s/shard, %s GiB host, %s GiB disk\n" \
  "$PARTS" "$(mnodes "$SHARD")" "$(gib "$BUILD_RAM")" "$(gib "$BUILD_DISK")"
[ "$BUILD_RAM"  -le "$MEM_AVAIL"  ] || echo "    WARNING: build needs more RAM than this job has"
[ "$BUILD_DISK" -le "$DISK_AVAIL" ] || echo "    WARNING: build needs more disk than $WORK has"
echo

# ── Query-set inventory ────────────────────────────────────────────────────
echo "  query sets:"
for m in $MUTATIONS; do
  q=$(ds_query "$m"); g=$(ds_gt "$m")
  if [ -r "$q" ]; then
    printf "    mut%-5s %6s queries, gt k=%-5s %s\n" "$m" "$(fbin_n "$q")" \
      "$([ -r "$g" ] && fbin_dim "$g" || echo -)" \
      "$([ -r "$g" ] || echo '(no gt: latency only)')"
  else
    printf "    mut%-5s MISSING (%s)\n" "$m" "$q"
  fi
done
echo

# ── Plan table ─────────────────────────────────────────────────────────────
printf "  %-4s %-4s %9s %10s %10s %s\n" method k B/edge B/node resident where
CELLS=""
ANY_HOST=0; ANY_DEVICE=0
for meth in $METHODS; do
  case "$meth" in lsh|pq|rq) ;; *) echo "  unknown method: $meth" >&2; exit 2;; esac
  for k in $K_LIST; do
    rc=0; query_config_supported "$k" || rc=$?
    if [ "$rc" = "1" ]; then
      printf "  %-4s %-4s %9s %10s %10s %s\n" "$meth" "$k" - - - \
        "SKIP: run_query has no R=$R_FIN k_ranks=$k $DIST $PACKED_T config"
      continue
    fi
    # rq_dim = 8k wider than the data is allowed, but there is nothing to
    # project onto: it degenerates to plain 1-bit RaBitQ over all padded_dim
    # coordinates, with the surplus bits stored at zero. Same recall as the
    # largest k that fits, at strictly more bytes — so flag it rather than
    # letting the B/edge column imply the extra bits bought something.
    rq_note=""
    if [ "$meth" = "rq" ] && [ $(( 8 * k )) -gt "$PADDED" ]; then
      rq_note="(no projection: codes $PADDED dims, $(( 8 * k - PADDED )) bits/edge idle)"
    fi
    # PQ splits padded_dim into k subspaces; it must divide evenly.
    if [ "$meth" = "pq" ] && [ $(( PADDED % k )) -ne 0 ]; then
      printf "  %-4s %-4s %9s %10s %10s %s\n" "$meth" "$k" - - - \
        "SKIP: padded_dim=$PADDED not divisible by pq_m=$k"
      continue
    fi
    pe=$(payload_per_node "$meth" "$k" "$R_FIN")
    pn=$(resident_per_node "$meth" "$k")
    res=$(( N * pn ))
    where="device"
    case "$ON_HOST" in
      1|yes|true|on|force)   where="host (forced)";   ANY_HOST=1;;
      0|no|false|off|device) where="device (forced)"; ANY_DEVICE=1;;
      *) if [ "$res" -gt "$DEVICE_FREE_BYTES" ]; then where="HOST (PCIe)"; ANY_HOST=1
         else ANY_DEVICE=1; fi;;
    esac
    printf "  %-4s %-4s %9s %10s %10s %s\n" \
      "$meth" "$k" "$(( pe / R_FIN ))" "$pn" "$(gib "$res") GiB" "$where $rq_note"
    CELLS="$CELLS $meth:$k"
  done
done
echo
CELLS="${CELLS# }"
[ -n "$CELLS" ] || { echo "No runnable cells." >&2; exit 1; }
NCELL=$(echo "$CELLS" | wc -w | tr -d ' ')
NMUT=$(echo "$MUTATIONS" | wc -w | tr -d ' ')
echo "  cells: $NCELL x $NMUT mutation(s) = $(( NCELL * NMUT )) run_query invocations,"
echo "         each reloading the index and re-populating its codes"
if [ "$ANY_HOST" = "1" ] && [ "$ANY_DEVICE" = "1" ]; then
  echo
  echo "  ⚠ THIS SWEEP STRADDLES THE DEVICE/HOST LINE. Some cells are searched on"
  echo "    the card and others over PCIe, so their QPS numbers are not comparable"
  echo "    to each other — recall still is."
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
BUILD_LOG="$LOGDIR/${DATASET}_r${R_INT}_${R_FIN}_plain_build.log"

if [ -s "$IDX" ]; then
  echo "[$( stamp )] plain index exists, reusing: $IDX ($(gib "$(stat -c %s "$IDX")") GiB)"
else
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

# ── Query sets, tiled once per mutation ────────────────────────────────────
prep_queries() {           # $1 = mutation; sets QFILE / GFILE
  QFILE=$(ds_query "$1"); GFILE=$(ds_gt "$1")
  [ -r "$QFILE" ] || return 1
  [ -r "$GFILE" ] || GFILE=""
  if [ "$QUERY_BATCH" != "0" ]; then
    local tq="$WORK/${DATASET}_mut$1_x${QUERY_BATCH}_query.fbin"
    local tg="$WORK/${DATASET}_mut$1_x${QUERY_BATCH}_gt.ibin"
    if [ ! -s "$tq" ]; then
      local targ=()
      [ -n "$GFILE" ] && targ=(--gt "$GFILE" --out-gt "$tg")
      python3 "$REPO/scripts/tile_queries.py" --target "$QUERY_BATCH" \
          --query "$QFILE" --out-query "$tq" ${targ[@]+"${targ[@]}"} \
        || { echo "  tiling failed for mut$1; using the untiled set"; return 0; }
    fi
    [ -s "$tq" ] && QFILE="$tq"
    [ -s "$tg" ] && GFILE="$tg"
  fi
  return 0
}

# ── Nodes popped per query (RUN_POPS=1) ────────────────────────────────────
# run_query reports throughput, not how many nodes each search popped. With
# RUN_POPS=1 every estimator cell is followed by distortion_test --pops on the
# SAME plain index, with the same --on_host decision, queries, ground truth and
# beam:limit pairs, so the QPS curve and the nodes-explored curve come from one
# graph and one build. Costs a second index load + code build per cell.
run_pops() {               # $1 method, $2 k, $3 mutation, $4 log, then --on_host if any
  local m=$1 k=$2 mut=$3 log=$4; shift 4
  local dm=$m; [ "$m" = "rq" ] && dm=rabitq
  local name="$DATASET"; [ "$mut" = "0.00" ] || name="${DATASET}_mut${mut}"
  local out="$POPS_DIR/${name}_${dm}_k${k}.csv"
  [ -n "$GFILE" ] || { echo "  pops: skipped (no ground truth for mut$mut)"; return 0; }
  echo "[$( stamp )] pops $dm k$k mut$mut -> $out"
  ./build/bin/distortion_test -i "$IDX" --dim "$DIM" -t "$SRC_DTYPE" -n "$R_FIN" \
      -m "$dm" -r "$k" -q "$QFILE" -g "$GFILE" -k "$K" "$@" --pops "$out" -b $BEAM_LIMITS 2>&1 \
    | tee -a "$log" \
    | grep -E "mean_pops" \
    || echo "  POPS FAILED ($dm k$k mut$mut) — see $log"
}

run_cell() {               # $1 = method, $2 = k_ranks ("" = baseline), $3 = mutation
  local meth=$1 k=${2:-} mut=$3
  local mode_arg=() host_arg=() gt_arg=() label res
  case "$meth" in
    conventional) mode_arg=(); label="conventional"
                  res=$(( N * (PADDED * 2 + R_FIN * 6 + 1 + 4) ));;
    lsh) mode_arg=(--lsh    --k_ranks "$k"); label="lsh_k$k"; res=$(( N * $(resident_per_node lsh "$k") ));;
    pq)  mode_arg=(--pq     --k_ranks "$k"); label="pq_k$k";  res=$(( N * $(resident_per_node pq  "$k") ));;
    rq)  mode_arg=(--rabitq --k_ranks "$k"); label="rq_k$k";  res=$(( N * $(resident_per_node rq  "$k") ));;
  esac

  case "$ON_HOST" in
    1|yes|true|on|force)   host_arg=(--on_host);;
    0|no|false|off|device) ;;
    *) [ "$res" -gt "$DEVICE_FREE_BYTES" ] && host_arg=(--on_host);;
  esac
  [ -n "$GFILE" ] && gt_arg=(-g "$GFILE")

  local log="$LOGDIR/${DATASET}_r${R_INT}_${R_FIN}_${label}_mut${mut}.log"
  echo "========== $label  mut$mut  ($(gib "$res") GiB resident${host_arg[0]+, --on_host}) =========="
  echo "[$( stamp )] $label mut$mut"
  # No --prepopulated: codes are built at load from the plain index, which is
  # what lets every cell share one graph.
  ./build/bin/run_query -i "$IDX" --dim "$DIM" -d "$DIST" -t "$SRC_DTYPE" \
      -n "$R_FIN" -k "$K" ${mode_arg[@]+"${mode_arg[@]}"} ${host_arg[@]+"${host_arg[@]}"} \
      -q "$QFILE" ${gt_arg[@]+"${gt_arg[@]}"} --beam_limits $BEAM_LIMITS 2>&1 \
    | tee -a "$log" \
    | grep -E "beam_width|Recall|populate|generate|Rotating|Training" \
    || echo "  CELL FAILED ($label mut$mut) — see $log"
  if [ "$RUN_POPS" = "1" ] && [ "$meth" != "conventional" ]; then
    run_pops "$meth" "$k" "$mut" "$log" ${host_arg[@]+"${host_arg[@]}"}
  fi
  echo
}

for mut in $MUTATIONS; do
  if ! prep_queries "$mut"; then
    echo "### no query file for mut$mut, skipping ###"; continue
  fi
  echo "######## mut$mut ($(fbin_n "$QFILE") queries) ########"
  # One conventional run per mutation: the ceiling each estimator is a
  # fraction of, measured on the same queries.
  [ "$RUN_BASELINE" = "1" ] && run_cell conventional "" "$mut"
  for cell in $CELLS; do
    run_cell "${cell%%:*}" "${cell##*:}" "$mut"
  done
done

echo "=== done: $(stamp) ==="
echo "Per-cell logs in $LOGDIR/${DATASET}_r${R_INT}_${R_FIN}_<method>_k<k>_mut<m>.log"
[ "$RUN_POPS" = "1" ] && echo "Nodes-popped CSVs in $POPS_DIR/${DATASET}_<method>_k<k>.csv" || true
