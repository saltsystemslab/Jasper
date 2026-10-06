#!/bin/bash
#SBATCH --partition=SaltSystemsLab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --nodelist=d4067
#SBATCH --mem=987842M
#SBATCH --gres=gpu:1
#SBATCH --job-name=locale
#SBATCH --output=locale_%j.log
#SBATCH --error=locale_%j.err
#SBATCH --time=24:00:00
#
# DiskANN-merge + directional index benchmark on LOCALE (SRA genomic sketches),
# across the two dataset sizes and all three query mutation rates.
#
#   $DATA/sra50/     ~10M  vectors    $DATA/sra500/    ~100M vectors
#     base.fbin                         base.fbin
#     query_mut{0.00,0.05,0.10}.fbin    (same three)
#     gt_mut{0.00,0.05,0.10}.ibin
#
# This is bench_deep_scaling.sbatch with one extra axis. There the ladder was
# over N; here N has only two rungs and the interesting sweep is MUTATION: the
# same index answered by queries that have drifted 0%, 5% and 10% from the base
# sequences. Recall at mut0.00 measures the index; the slope from 0.00 to 0.10
# measures how gracefully it degrades when the query is no longer a near-exact
# member of the base set — which for a graph index is a different question, and
# the one an estimator can plausibly get wrong.
#
# Each dataset builds ONE index and every (mutation x estimator x k) combination
# queries that same index, so any difference is attributable to the query set or
# the estimator, never to the graph.
#
# ── Before launching ───────────────────────────────────────────────────────
# Run DRY_RUN=1 first. It probes each base.fbin for its real N and dim, prints
# the host/device/disk budget per dataset, and exits without building.
#
#   DRY_RUN=1 ./sbatch/bench_locale.sbatch
#   sbatch sbatch/bench_locale.sbatch
#   DATASETS=sra50 ESTIMATORS=pq+rq sbatch sbatch/bench_locale.sbatch
#   MUTATIONS=0.10 K=10 sbatch sbatch/bench_locale.sbatch
#   MUTATIONS=0.00 QUERY_BATCH=10000 sbatch sbatch/bench_locale.sbatch  # for QPS
#   WORK=/tmp2/zikun/locale_bench sbatch sbatch/bench_locale.sbatch   # bigger fs
#
# Indexes land in $WORK, which defaults to node-local /tmp (~1 TB on d4067) —
# the shared filesystem is too small for sra500. /tmp is wiped at job end, so
# indexes are disposable and KEEP_INDEX defaults to 0. Logs always go to
# $LOGDIR, which is on persistent storage for exactly that reason.
#
# ⚠ Build and query run in ONE job and cannot be split. $WORK defaults to /tmp,
# which this cluster clears when the job ends, so an index does not outlive the
# job that made it — STAGE_MODE=build / =query is rejected outright unless you
# point WORK= at persistent storage. Budget the wall clock for both phases.
#
# ⚠ COST: run_query loads the whole index per invocation, and the mutation axis
# multiplies the invocation count by 3. The banner prints the total before
# anything runs — at sra500 with the default modes that is the dominant cost of
# the query phase, not the searching. Trim QUERY_MODES / MUTATIONS / K.
#
# ⚠ DIST defaults to l2. These are genomic sketch vectors and I have not
# verified which metric the ground truth was generated under — if recall at
# mut0.00 comes out implausibly low across every mode INCLUDING conventional,
# that is the first thing to check (DIST=ip).

set -euo pipefail

# ── Tunables (all overridable from the environment) ────────────────────────
DATASETS="${DATASETS:-sra50 sra500}"
MUTATIONS="${MUTATIONS:-0.00 0.05 0.10}"
# '+'-separated so any combination works: pq, lsh+pq, pq+rq, lsh+pq+rq, none.
# pq+rq is the useful default here: both are per-edge quantizers over the same
# graph, so running them together is the cheapest way to compare them.
ESTIMATORS="${ESTIMATORS:-pq+rq}"
QUERY_MODES="${QUERY_MODES:-conventional lsh pq rq}"   # unbuilt modes self-skip
STAGE_MODE="${STAGE_MODE:-build+query}"    # build+query | build | query
INDEX="${INDEX:-}"                         # query an explicit file instead of
                                           # the derived $WORK/<dataset>_... name
# See bench_deep_scaling.sbatch for why R_int and R_fin must move together: at
# R_int < R_fin the merge's "keep the R_fin closest" step has nothing to prune.
R_INT="${R_INT:-64}"
R_FIN="${R_FIN:-64}"
K_RANKS="${K_RANKS:-4}"
N_DUP="${N_DUP:-2}"
ALPHA="${ALPHA:-1.2}"
DIST="${DIST:-l2}"                   # see the warning above
WORKSPACE_BUDGET="${WORKSPACE_BUDGET:-10GB}"
DEVICE_FREE_MIB="${DEVICE_FREE_MIB:-0}"    # 0 = ask nvidia-smi, else override
SHARD_FILL="${SHARD_FILL:-70}"             # % of max_shard to aim for
N_PARTS="${N_PARTS:-0}"                    # 0 = derive from device capacity
# One k by default, unlike the deep ladder: the mutation axis already triples
# the number of index loads, and k=10 is the headline. read_groundtruth rejects
# a k larger than the gt file's own k, so check the .ibin before raising this.
K="${K:-10 100}"
BEAM_LIMITS="${BEAM_LIMITS:-16:128 32:128 64:128 128:256 256:512 512:1024 1024:2048 1536:2048 1982:2048}"
# Tile the query set (and its gt) up to at least this many queries before
# searching, rounded UP to a whole number of repeats. The locale query sets are
# ~500, and the beam search runs one block per query, so 500 queries leaves the
# GPU nearly idle and the QPS column measures launch overhead. Repeating the
# batch saturates it. Recall is unaffected: a whole number of copies means every
# query is weighted exactly as before (see scripts/tile_queries.py).
# 0 disables tiling and uses the query files as they are.
QUERY_BATCH="${QUERY_BATCH:-0}"
ON_HOST="${ON_HOST:-auto}"           # auto | 1 | 0
DRY_RUN="${DRY_RUN:-0}"
# Default 0 because $WORK is wiped when the job ends (see below): keeping an
# index buys nothing across jobs, and dropping it right after its queries means
# only the LARGEST index has to fit at once rather than their sum. Set 1 only
# when $WORK is persistent.
KEEP_INDEX="${KEEP_INDEX:-0}"        # 0 = delete each index after querying
# df reports the whole volume, which can be much larger than what this job may
# actually use (/tmp on d4067 is ~1 TB). Set this to cap the budget in GiB; 0
# trusts df.
DISK_BUDGET="${DISK_BUDGET:-0}"      # GiB, 0 = ask df

# /tmp2 is node-local scratch on d4067. Fine to read from, but don't assume it
# survives the job — point DATA= elsewhere if it has been cleaned.
DATA="${DATA:-/tmp2/ann_data/locale}"
# Indexes default to node-local scratch because the shared filesystem does not
# have room: /projects had ~86 GiB free and one sra500 index is ~410 GiB at
# R_fin=64, dim 768. The tradeoff is that /tmp is per-node and may be cleared
# between jobs, so an index here is a CACHE, not an artifact — "index exists,
# reusing" still works for a resubmission that lands on the same node, but do
# not count on it: on this cluster /tmp is cleared when the job exits, which is
# why build and query must run in the same job and why KEEP_INDEX defaults to 0.
# Set WORK= to persistent storage to keep an index. df reports the whole volume,
# so use DISK_BUDGET= if the real allowance is smaller (/tmp here is ~1 TB).
WORK="${WORK:-/tmp/${USER:-$(id -un)}/locale_bench}"
REPO="${REPO:-$HOME/jasper}"
# Logs stay on the shared filesystem on purpose: they are the part you cannot
# rebuild cheaply, and $WORK is now ephemeral.
LOGDIR="${LOGDIR:-$REPO/locale_logs}"

SRC_DTYPE=float                      # .fbin

# ── Dataset layout ─────────────────────────────────────────────────────────
ds_base()  { echo "$DATA/$1/base.fbin"; }
ds_query() { echo "$DATA/$1/query_mut$2.fbin"; }
ds_gt()    { echo "$DATA/$1/gt_mut$2.ibin"; }

# .fbin / .ibin both start with [int32 count][int32 dim-or-k], which is exactly
# what jasper's load_vectors_from_file_cast reads, so the file is authoritative
# about its own shape — no hardcoded N or dim here.
fbin_u32() { od -An -tu4 -N4 -j"$2" "$1" 2>/dev/null | tr -d ' \n'; }
fbin_n()   { fbin_u32 "$1" 0; }
fbin_dim() { fbin_u32 "$1" 4; }

# ── Estimator selection ────────────────────────────────────────────────────
BUILD_LSH=0; BUILD_PQ=0; BUILD_RQ=0
if [ "$ESTIMATORS" != "none" ]; then
  for tok in ${ESTIMATORS//+/ }; do
    case "$tok" in
      lsh)       BUILD_LSH=1;;
      pq)        BUILD_PQ=1;;
      rq|rabitq) BUILD_RQ=1;;
      *) echo "ESTIMATORS tokens must be lsh, pq, rq (or the whole value 'none');" \
              "got '$tok' in '$ESTIMATORS'" >&2; exit 2;;
    esac
  done
fi

# ── Probe the datasets ─────────────────────────────────────────────────────
# Every dataset must agree on dim: it selects the compiled config (packed_t is
# uint8 for dim <= 128, uint16 above) and drives the whole budget model below.
DIM=0
PROBED=""
for d in $DATASETS; do
  b=$(ds_base "$d")
  if [ ! -r "$b" ]; then
    echo "NOTE: no base at $b — '$d' will be skipped" >&2
    continue
  fi
  dm=$(fbin_dim "$b")
  [ -n "$dm" ] && [ "$dm" -gt 0 ] 2>/dev/null || {
    echo "Could not read a dim from $b (truncated, or not an .fbin)" >&2; exit 2; }
  if [ "$DIM" = "0" ]; then
    DIM=$dm
  elif [ "$dm" != "$DIM" ]; then
    echo "Datasets disagree on dim ($DIM vs $dm at $b); run them separately." >&2
    exit 2
  fi
  PROBED="$PROBED $d"
done
[ "$DIM" != "0" ] || { echo "No readable dataset under $DATA" >&2; exit 1; }
DATASETS="${PROBED# }"

PADDED=$(( (DIM + 7) / 8 * 8 ))
PACKED_BYTES=$([ "$DIM" -le 128 ] && echo 1 || echo 2)

# RaBitQ geometry, derived like the C++ side: rq_dim = 8*k_ranks at 1 bit per
# dimension, packed into 32-bit words, plus an fp16 rescale and a bf16 mag² per
# edge (see edge_rq_list in rabitq/edge_rq.cuh).
RQ_DIM=$(( 8 * K_RANKS ))
RQ_WORDS=$(( (RQ_DIM + 31) / 32 ))
RQ_PER_EDGE=$(( 4 * RQ_WORDS + 4 ))

# The codes truncate to the first rq_dim rotated coordinates, so the projection
# has to fit inside the data.
if [ "$BUILD_RQ" = "1" ] && [ "$RQ_DIM" -gt "$DIM" ]; then
  echo "ESTIMATORS includes rq but rq_dim=8*K_RANKS=$RQ_DIM exceeds dim=$DIM." >&2
  echo "Lower K_RANKS (4 gives rq_dim=32) or drop rq from ESTIMATORS." >&2
  exit 2
fi

# Modes that will actually run, so the banner can state the real invocation
# count rather than the nominal one.
active_modes() {
  local out=""
  for mode in $QUERY_MODES; do
    case "$mode" in
      conventional) out="$out $mode";;
      lsh) [ "$BUILD_LSH" = "1" ] && out="$out $mode";;
      pq)  [ "$BUILD_PQ"  = "1" ] && out="$out $mode";;
      rq|rabitq) [ "$BUILD_RQ" = "1" ] && out="$out $mode";;
    esac
  done
  echo "${out# }"
}
count() { echo $#; }

# ── Config-table preflight ─────────────────────────────────────────────────
# create_diskann_index and run_query each compile a FIXED set of
# (R, k_ranks, distance, packed_t) combinations; anything else throws
# "Unsupported config" at dispatch — which happens after the dataset load, so
# at these sizes it is minutes wasted to learn it. The tables differ: at the
# time of writing run_query has k_ranks 4/8/16 at R=64, while
# create_diskann_index only has k_ranks=4 there.
#
# Read the tables out of the source instead of duplicating them here, so this
# check cannot drift from what is actually compiled.
FUNC_TOKEN=$([ "$DIST" = "ip" ] && echo INNER_PRODUCT || echo L2)
PACKED_T=$([ "$DIM" -le 128 ] && echo uint8_t || echo uint16_t)

# 0 = present, 1 = absent, 2 = could not check
config_supported() {
  local f="$REPO/$1" needle="$2"
  [ -r "$f" ] || return 2
  sed 's/[[:space:]]//g' "$f" | grep -q -- "$needle"
}

# create_diskann_index: X(id, IDX, R_INT, R_FIN, DAT, DIST, FUNC, KR, PACKEDT)
BUILD_NEEDLE="uint32_t,${R_INT},${R_FIN},__half,float,jasper::distance_func::${FUNC_TOKEN},${K_RANKS},${PACKED_T}"
# run_query directional: X(id, IDX, R, DAT, DIST, FUNC, KR, PACKEDT)
QUERY_NEEDLE="uint32_t,${R_FIN},__half,float,jasper::distance_func::${FUNC_TOKEN},${K_RANKS},${PACKED_T}"

CFG_FATAL=0
if [ "$BUILD_LSH$BUILD_PQ$BUILD_RQ" != "000" ]; then
  # `cmd; rc=$?` does NOT survive set -e — a failing simple command exits the
  # shell before the assignment runs. The || form is exempt.
  rc=0; config_supported cmd/create_diskann_index.cu "$BUILD_NEEDLE" || rc=$?
  case $rc in
    1) echo "CONFIG: create_diskann_index has no directional config for" >&2
       echo "        R_int=$R_INT R_fin=$R_FIN k_ranks=$K_RANKS $DIST ${PACKED_T} (dim $DIM)." >&2
       echo "        Compiled combinations (grep JASPER_FOR_EACH_DIR_CONFIG):" >&2
       sed 's/[[:space:]]\+/ /g' "$REPO/cmd/create_diskann_index.cu" \
         | grep -oE 'X\(d_r[0-9]+_[0-9]+_[a-z0-9_]+' | sed 's/^X(/          /' >&2
       CFG_FATAL=1;;
    2) echo "CONFIG: cannot read \$REPO/cmd/create_diskann_index.cu — skipping the" >&2
       echo "        compiled-config check (set REPO=... if the build fails at dispatch)." >&2;;
  esac

  for mode in $(active_modes); do
    case "$mode" in lsh|pq|rq|rabitq) ;; *) continue;; esac
    rc=0; config_supported cmd/run_query.cu "$QUERY_NEEDLE" || rc=$?
    case $rc in
      1) echo "CONFIG: run_query has no directional config for R=$R_FIN k_ranks=$K_RANKS" >&2
         echo "        $DIST ${PACKED_T} — the '$mode' query mode cannot dispatch." >&2
         CFG_FATAL=1;;
    esac
    break
  done
fi
[ "$CFG_FATAL" = "0" ] || {
  echo >&2
  echo "Pick a compiled combination, or add the row to the table and rebuild." >&2
  exit 2
}

# ── Budget model (mirrors docs/design/directional-diskann-population.md §4) ──
host_per_vec() {
  local per=0
  per=$(( per + PADDED * 2 ))                       # dataset row (__half, pinned)
  per=$(( per + R_FIN * 6 ))                        # edge_list_t: 4B id + 2B bf16
  per=$(( per + 1 ))                                # edge_counts
  per=$(( per + N_DUP * 4 ))                        # shard_ids
  [ "$BUILD_LSH" = "1" ] && per=$(( per + R_FIN * K_RANKS * PACKED_BYTES + R_FIN * 2 ))
  [ "$BUILD_PQ"  = "1" ] && per=$(( per + R_FIN * K_RANKS + 4 ))   # codes + ||v||²
  [ "$BUILD_RQ"  = "1" ] && per=$(( per + R_FIN * RQ_PER_EDGE ))
  echo "$per"
}

device_per_slot() {
  local per=$(( PADDED * 2 + R_INT * 6 + 1 + 4 + 1 ))
  [ "$BUILD_LSH" = "1" ] && per=$(( per + R_INT * K_RANKS * PACKED_BYTES + R_INT * 2 ))
  [ "$BUILD_PQ"  = "1" ] && per=$(( per + R_INT * K_RANKS ))
  [ "$BUILD_RQ"  = "1" ] && per=$(( per + R_INT * RQ_PER_EDGE ))
  echo "$per"
}

merge_job_per_node() {
  local per=$(( R_INT * 6 + 1 + 4 ))
  [ "$BUILD_LSH" = "1" ] && per=$(( per + R_INT * K_RANKS * PACKED_BYTES + R_INT * 2 ))
  [ "$BUILD_PQ"  = "1" ] && per=$(( per + R_INT * K_RANKS ))
  [ "$BUILD_RQ"  = "1" ] && per=$(( per + R_INT * RQ_PER_EDGE ))
  echo "$per"
}

query_resident_per_node() {
  local per=$(( PADDED * 2 + R_FIN * 6 + 1 + 4 ))
  [ "$BUILD_LSH" = "1" ] && per=$(( per + R_FIN * K_RANKS * PACKED_BYTES + R_FIN * 2 ))
  [ "$BUILD_PQ"  = "1" ] && per=$(( per + R_FIN * K_RANKS + 4 ))
  [ "$BUILD_RQ"  = "1" ] && per=$(( per + R_FIN * RQ_PER_EDGE ))
  echo "$per"
}

base_disk_per_vec() { echo $(( DIM * 2 + 1 + R_FIN * 6 )); }

disk_per_vec() {
  local per
  per=$(base_disk_per_vec)
  [ "$BUILD_LSH" = "1" ] && per=$(( per + R_FIN * K_RANKS * PACKED_BYTES + R_FIN * 2 ))
  [ "$BUILD_PQ"  = "1" ] && per=$(( per + R_FIN * K_RANKS + 4 ))
  [ "$BUILD_RQ"  = "1" ] && per=$(( per + R_FIN * RQ_PER_EDGE ))
  echo "$per"
}

max_shard() {
  local per_slot per_seg n_seg avail
  per_slot=$(device_per_slot)
  per_seg=$(( per_slot * 1048576 ))
  avail=$(( DEVICE_FREE_BYTES - WORKSPACE_BYTES - 4 * 1073741824 ))
  [ "$avail" -lt 0 ] && { echo 0; return; }
  n_seg=$(( avail / per_seg ))
  echo $(( n_seg * 1048576 ))
}

# Fewest shards whose pieces still fit on the device — see the long note in
# bench_deep_scaling.sbatch; the 2*dim cap is a hard property of
# polytope_lsh_assign_kernel, not a heuristic.
n_parts_for() {
  local n=$1 cap=$(( 2 * DIM )) target p
  if [ "$N_PARTS" != "0" ]; then echo "$N_PARTS"; return; fi
  target=$(( $(max_shard) * SHARD_FILL / 100 ))
  [ "$target" -lt 1 ] && { echo 0; return; }
  # A shard can never hold more than n DISTINCT vectors — the n_dup duplicates
  # collapse when they land in the same shard — so the n*N_DUP work estimate
  # only binds once it is actually spread across parts. Without this check a
  # dataset that fits whole on the device still gets split, purely because
  # n*N_DUP exceeds the target, which costs recall for nothing.
  if [ "$n" -le "$target" ]; then echo 1; return; fi
  p=$(( (n * N_DUP + target - 1) / target ))
  [ "$p" -lt 1 ] && p=1
  [ "$p" -gt "$cap" ] && p=$cap
  echo "$p"
}

check_pinned_parts() {
  local n=$1 p=$2 shard ms
  [ "$N_PARTS" = "0" ] && return 0
  ms=$(max_shard)
  shard=$(( n * N_DUP / p ))
  [ "$shard" -gt "$n" ] && shard=$n
  if [ "$shard" -gt "$ms" ]; then
    echo "  WARNING: N_PARTS=$p gives $(mnodes "$shard")-node shards but the device"
    echo "           holds $(mnodes "$ms"); build() will split them into sub-shards."
  fi
  if [ "$p" -gt "$(( 2 * DIM ))" ]; then
    echo "  WARNING: N_PARTS=$p exceeds 2*dim=$(( 2 * DIM )); shards above that id"
    echo "           are never produced by the sharding kernel and stay empty."
  fi
}

gib() { awk -v b="$1" 'BEGIN{printf "%.1f", b/1073741824}'; }
mnodes() { awk -v n="$1" 'BEGIN{printf "%.1fM", n/1000000}'; }

parse_size() {
  awk -v s="$1" 'BEGIN{
    n = s + 0; u = toupper(s); sub(/^[0-9.]+/, "", u);
    m = 1;
    if (u == "KB") m = 1024;
    else if (u == "MB") m = 1024^2;
    else if (u == "GB") m = 1024^3;
    else if (u == "TB") m = 1024^4;
    printf "%d", n * m;
  }'
}

resolve_device_free() {
  if [ "$DEVICE_FREE_MIB" != "0" ]; then
    echo $(( DEVICE_FREE_MIB * 1048576 )); return
  fi
  local mib
  mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null \
        | head -1 | tr -d ' ') || mib=""
  [ -n "$mib" ] || mib=96683
  echo $(( mib * 1048576 ))
}

mem_available_bytes() {
  if [ -n "${SLURM_MEM_PER_NODE:-}" ]; then
    echo $(( SLURM_MEM_PER_NODE * 1048576 )); return
  fi
  if [ -n "${SLURM_MEM_PER_CPU:-}" ] && [ -n "${SLURM_CPUS_PER_TASK:-}" ]; then
    echo $(( SLURM_MEM_PER_CPU * SLURM_CPUS_PER_TASK * 1048576 )); return
  fi
  awk '/^MemAvailable:/ {print $2 * 1024; exit}' /proc/meminfo 2>/dev/null || echo 0
}
mem_total_bytes() {
  awk '/^MemTotal:/ {print $2 * 1024; exit}' /proc/meminfo 2>/dev/null || echo 0
}
# GNU df only. The `|| echo 0` on the pipeline is not enough — df failing still
# leaves tr succeeding with empty output, and an empty DISK_AVAIL turns every
# later comparison into a bash syntax error rather than a skip.
disk_avail_bytes() {
  local v
  v=$(df -B1 --output=avail "$1" 2>/dev/null | tail -1 | tr -d ' ')
  [ -n "$v" ] || v=0
  echo "$v"
}

echo "=== locale benchmark ==="
echo "  host:        $(hostname)"
echo "  datasets:    $DATASETS   (from $DATA)"
echo "  mutations:   $MUTATIONS"
echo "  estimators:  $ESTIMATORS (lsh=$BUILD_LSH pq=$BUILD_PQ rq=$BUILD_RQ,"
echo "               k_ranks=$K_RANKS$([ "$BUILD_RQ" = "1" ] && echo ", rq_dim=$RQ_DIM @ ${RQ_PER_EDGE}B/edge"))"
echo "  degrees:     R_int=$R_INT R_fin=$R_FIN  n_dup=$N_DUP  alpha=$ALPHA"
echo "  distance:    $DIST   dim=$DIM (probed; packed_t=${PACKED_BYTES}B)"
ACTIVE_MODES=$(active_modes)
LOADS=$(( $(count $K) * $(count $MUTATIONS) * $(count $ACTIVE_MODES) ))
echo "  query modes: ${ACTIVE_MODES:-none}   (k = $K)"
echo "  -> $LOADS run_query invocations per dataset, each reloading the index"
[ "$QUERY_BATCH" = "0" ] || \
  echo "  query batch: tiled to >= $QUERY_BATCH queries (whole repeats; recall unchanged)"
echo "  workdir:     $WORK"
echo "  logs:        $LOGDIR"

mkdir -p "$WORK" "$LOGDIR"
MEM_TOTAL=$(mem_total_bytes)
MEM_AVAIL=$(mem_available_bytes)
DISK_AVAIL=$(disk_avail_bytes "$WORK")
if [ "$DISK_BUDGET" != "0" ]; then
  DISK_AVAIL=$(( DISK_BUDGET * 1073741824 ))
fi
DEVICE_FREE_BYTES=$(resolve_device_free)
WORKSPACE_BYTES=$(parse_size "$WORKSPACE_BUDGET")
MAX_SHARD=$(max_shard)

if [ -n "${SLURM_MEM_PER_NODE:-}" ]; then
  echo "  host RAM:    $(gib "$MEM_AVAIL") GiB granted by SLURM (--mem);" \
       "node has $(gib "$MEM_TOTAL") GiB"
else
  echo "  host RAM:    $(gib "$MEM_AVAIL") GiB available of $(gib "$MEM_TOTAL") GiB"
fi
echo "  disk:        $(gib "$DISK_AVAIL") GiB available at $WORK"
echo "  device:      $(gib "$DEVICE_FREE_BYTES") GiB free, minus $WORKSPACE_BUDGET"
echo "               workspace + 4 GiB slack"
echo "  max_shard:   $(mnodes "$MAX_SHARD") nodes at $(device_per_slot) B/slot"
echo "               -> aiming for $(mnodes $(( MAX_SHARD * SHARD_FILL / 100 )))/shard (${SHARD_FILL}% fill)"
echo

# $WORK under /tmp is wiped at job end on this cluster, so an index built in one
# job is gone before a second job could query it. Catch that here rather than
# letting the query job rebuild from scratch (hours) or fail on a missing file.
case "$WORK" in
  /tmp/*|/var/tmp/*) WORK_EPHEMERAL=1;;
  *)                 WORK_EPHEMERAL=0;;
esac
if [ "$WORK_EPHEMERAL" = "1" ] && [ "$STAGE_MODE" != "build+query" ]; then
  echo "STAGE_MODE=$STAGE_MODE with WORK=$WORK cannot work: /tmp is cleared when" >&2
  echo "the job ends, so a build-only job leaves nothing for a query-only job to" >&2
  echo "find. Either keep STAGE_MODE=build+query (the default), or point WORK= at" >&2
  echo "persistent storage with room for the index." >&2
  exit 2
fi

[ "$MAX_SHARD" -gt 0 ] || {
  echo "Nothing fits on the device: $(device_per_slot) B/slot leaves no room after" >&2
  echo "the $WORKSPACE_BUDGET workspace + 4 GiB slack. Lower WORKSPACE_BUDGET." >&2
  exit 1
}

# ── Query-set inventory ────────────────────────────────────────────────────
# Missing query/gt files are the most likely reason a stage does less than you
# expect, so list what was actually found before spending hours on a build.
echo "  query sets:"
for d in $DATASETS; do
  for m in $MUTATIONS; do
    q=$(ds_query "$d" "$m"); g=$(ds_gt "$d" "$m")
    qn="-"; gk="-"
    [ -r "$q" ] && qn=$(fbin_n "$q")
    [ -r "$g" ] && gk=$(fbin_dim "$g")   # second header field of a gt file is k
    if [ -r "$q" ]; then
      printf "    %-8s mut%-5s %8s queries, gt k=%-6s %s\n" \
        "$d" "$m" "$qn" "$gk" "$([ -r "$g" ] || echo '(no gt: latency only)')"
    else
      printf "    %-8s mut%-5s MISSING (%s)\n" "$d" "$m" "$q"
    fi
  done
done
echo

# ── Budget table ───────────────────────────────────────────────────────────
HPV=$(host_per_vec)
DPV=$(disk_per_vec)
MJPN=$(merge_job_per_node)
printf "%-8s %11s %6s %8s %10s %10s %9s %10s %s\n" \
  dataset N parts shard host_GiB gather_GiB merge_GiB disk_GiB verdict
FEASIBLE=""
RETAINED=0
DISK_SHORT=0
RAM_SHORT=0
for d in $DATASETS; do
  n=$(fbin_n "$(ds_base "$d")")
  p=$(n_parts_for "$n")
  shard=$(( n * N_DUP / p ))
  [ "$shard" -gt "$n" ] && shard=$n

  hb=$(( n * HPV ))
  gb=$(( 2 * shard * PADDED * 2 ))
  chunk=$shard; [ "$chunk" -gt 1048576 ] && chunk=1048576
  mb=$(( 10 * chunk * MJPN ))
  db=$(( n * DPV ))
  ram_total=$(( hb + gb + mb ))

  disk_left=$DISK_AVAIL
  [ "$KEEP_INDEX" = "1" ] && disk_left=$(( DISK_AVAIL - RETAINED ))

  verdict="ok"
  if [ "$ram_total" -gt "$MEM_AVAIL" ]; then
    verdict="SKIP: needs $(gib "$ram_total") GiB RAM"
    RAM_SHORT=1
  elif [ "$(( ram_total * 100 / MEM_AVAIL ))" -ge 90 ]; then
    verdict="ok, but TIGHT: $(( ram_total * 100 / MEM_AVAIL ))% of RAM"
  elif [ "$db" -gt "$disk_left" ]; then
    verdict="SKIP: needs $(gib "$db") GiB disk, $(gib "$disk_left") GiB left"
    DISK_SHORT=1
  fi
  printf "%-8s %11s %6s %8s %10s %10s %9s %10s %s\n" \
    "$d" "$n" "$p" "$(mnodes "$shard")" "$(gib "$hb")" "$(gib "$gb")" \
    "$(gib "$mb")" "$(gib "$db")" "$verdict"
  check_pinned_parts "$n" "$p"
  case "$verdict" in
    ok*) FEASIBLE="$FEASIBLE $d"
         [ "$KEEP_INDEX" = "1" ] && RETAINED=$(( RETAINED + db ));;
  esac
done
echo
echo "  per-vector: host ${HPV}B, disk ${DPV}B; per queued merge node: ${MJPN}B"
echo "  feasible:${FEASIBLE:- none}"
for d in $FEASIBLE; do
  n=$(fbin_n "$(ds_base "$d")")
  r=$(( n * $(query_resident_per_node) ))
  case "$ON_HOST" in
    1|yes|true|on|force)
      echo "  $d: ON_HOST=$ON_HOST, queries forced onto the host ($(gib "$r") GiB, slow)";;
    0|no|false|off|device)
      [ "$r" -gt "$DEVICE_FREE_BYTES" ] && \
        echo "  $d: ON_HOST=$ON_HOST but the index is $(gib "$r") GiB > device — queries will OOM";;
    *)
      [ "$r" -gt "$DEVICE_FREE_BYTES" ] && \
        echo "  $d: loaded index is $(gib "$r") GiB > device, so queries use --on_host (slow)";;
  esac
done
if [ "$KEEP_INDEX" = "1" ] && [ "$RETAINED" -gt 0 ]; then
  echo "  indexes retained across the run: $(gib "$RETAINED") GiB"
fi
# A RAM skip while running inside a smaller allocation than this file's own
# #SBATCH --mem is almost always just that: an interactive shell, not a real
# limit. Worth saying, because the skip otherwise reads as "won't fit, period".
if [ "$RAM_SHORT" = "1" ]; then
  SB_MEM_MIB=$(awk -F'=' '/^#SBATCH --mem=/{print $2}' "$0" | tr -dc '0-9')
  if [ -n "$SB_MEM_MIB" ] && [ "$(( SB_MEM_MIB * 1048576 ))" -gt "$MEM_AVAIL" ]; then
    echo
    echo "  NOTE: this shell has $(gib "$MEM_AVAIL") GiB, but the #SBATCH header in this"
    echo "        file asks for $(gib $(( SB_MEM_MIB * 1048576 ))) GiB. Submitting it"
    echo "        (sbatch $0) rather than running it directly may clear the RAM skips"
    echo "        above — re-check with DRY_RUN=1 inside the batch job if unsure."
  fi
fi
if [ "$DISK_SHORT" = "1" ]; then
  echo
  echo "  Disk is the binding constraint. The floor at R_fin=$R_FIN is the base"
  echo "  section alone — $(base_disk_per_vec) B/vector, no estimator choice gets under it."
  echo "  Point WORK= at a filesystem with room (df -h), or use KEEP_INDEX=0."
fi

if [ "$DRY_RUN" != "0" ]; then
  echo
  echo "DRY_RUN set — nothing built."
  echo
  echo "Levers, cheapest first:"
  echo "  MUTATIONS=0.00      one query set instead of three (3x fewer loads)"
  echo "  QUERY_MODES=...     drop modes you aren't comparing"
  echo "  KEEP_INDEX=0        delete each index after querying"
  echo "  ESTIMATORS=pq       rq is the biggest payload (${RQ_PER_EDGE}B/edge vs PQ's"
  echo "                      $K_RANKS B), so drop it first if memory is tight"
  echo "  SHARD_FILL=40       smaller shards: less gather+merge RAM, some recall"
  exit 0
fi
[ -n "${FEASIBLE// /}" ] || { echo "No dataset fits on this node; aborting." >&2; exit 1; }

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

echo "=== building ==="
cmake -B build -DJASPER_BUILD_CMD=ON -DJASPER_BUILD_TESTS=OFF -DJASPER_BUILD_FFI=OFF \
  -DThrust_DIR="$THRUST_DIR" -DCMAKE_PREFIX_PATH="$CONDA_PREFIX" >/dev/null
cmake --build build --target create_diskann_index run_query -j8 >/dev/null
echo "  ok"
echo

# ── Stages ─────────────────────────────────────────────────────────────────
stamp() { date '+%Y-%m-%d %H:%M:%S'; }

run_stage() {
  local d=$1
  local n base p idx log
  base=$(ds_base "$d")
  n=$(fbin_n "$base")
  p=$(n_parts_for "$n")
  idx="${INDEX:-$WORK/${d}_r${R_INT}_${R_FIN}_${ESTIMATORS}_k${K_RANKS}.bin}"
  # Everything that changes the index goes in the name, so two runs that differ
  # only in degree or k_ranks don't tee -a into the same file.
  log="$LOGDIR/${d}_r${R_INT}_${R_FIN}_${ESTIMATORS}_k${K_RANKS}.log"

  local do_build=0 do_query=0
  case "$STAGE_MODE" in
    build+query) do_build=1; do_query=1;;
    build)       do_build=1;;
    query)       do_query=1;;
    *) echo "STAGE_MODE must be build+query, build, or query" >&2; return 1;;
  esac

  echo "############### $d ($n vectors, dim=$DIM, n_parts=$p, $STAGE_MODE) ###############"
  [ "$do_build" = "0" ] || [ -r "$base" ] || {
    echo "  base missing: $base — skipping"; return 0; }

  local est_arg=()
  [ "$BUILD_LSH" = "1" ] && est_arg+=(--lsh)
  [ "$BUILD_PQ"  = "1" ] && est_arg+=(--pq)
  [ "$BUILD_RQ"  = "1" ] && est_arg+=(--rabitq)

  if [ "$do_build" = "0" ]; then
    [ -s "$idx" ] || { echo "  STAGE_MODE=query but no index at $idx"; return 1; }
    echo "  [$( stamp )] querying existing index: $idx ($(gib "$(stat -c %s "$idx")") GiB)"
  elif [ -s "$idx" ]; then
    echo "  [$( stamp )] index exists, reusing: $idx"
  else
    echo "  [$( stamp )] building -> $idx"
    /usr/bin/time -v ./build/bin/create_diskann_index \
        -f "$base" -t "$SRC_DTYPE" -d "$DIST" --dim "$DIM" \
        --r_intermediate "$R_INT" --r_final "$R_FIN" \
        --n_parts "$p" --n_dup "$N_DUP" -a "$ALPHA" \
        -w "$WORKSPACE_BUDGET" ${est_arg[@]+"${est_arg[@]}"} --k_ranks "$K_RANKS" \
        -i "$idx" 2>&1 | tee -a "$log" \
      || { echo "  BUILD FAILED for $d (see $log)"; return 1; }
    echo "  [$( stamp )] built, $(gib "$(stat -c %s "$idx")") GiB on disk"
  fi

  local resident=$(( n * $(query_resident_per_node) ))
  local host_arg=()
  case "$ON_HOST" in
    1|yes|true|on|force)
      host_arg=(--on_host)
      echo "  ON_HOST=$ON_HOST: forcing --on_host ($(gib "$resident") GiB resident)"
      ;;
    0|no|false|off|device)
      echo "  ON_HOST=$ON_HOST: forcing a device-resident query"
      if [ "$resident" -gt "$DEVICE_FREE_BYTES" ]; then
        echo "  WARNING: the index needs $(gib "$resident") GiB but the device has"
        echo "           $(gib "$DEVICE_FREE_BYTES") GiB — this will almost certainly OOM"
      fi
      ;;
    *)
      if [ "$resident" -gt "$DEVICE_FREE_BYTES" ]; then
        host_arg=(--on_host)
        echo "  loaded index is $(gib "$resident") GiB vs $(gib "$DEVICE_FREE_BYTES") GiB of device"
        echo "  -> querying with --on_host (pinned host memory, PCIe-resident)"
      fi
      ;;
  esac

  [ "$do_query" = "1" ] || { echo "  (STAGE_MODE=$STAGE_MODE, no queries)"; echo; return 0; }

  # k outermost, then mutation, then mode — so every estimator is compared back
  # to back on the SAME query set, and a job that runs out of time still leaves
  # complete mutation groups rather than a ragged half of each.
  for kk in $K; do
    for m in $MUTATIONS; do
      local q g gt_arg=()
      q=$(ds_query "$d" "$m")
      g=$(ds_gt "$d" "$m")
      [ -r "$q" ] || { echo "  ===== k=$kk mut$m: no query file ($q), skipping"; continue; }

      # Tile once per (dataset, mutation) into $WORK and reuse across k/modes.
      if [ "$QUERY_BATCH" != "0" ]; then
        local tq="$WORK/${d}_mut${m}_x${QUERY_BATCH}_query.fbin"
        local tg="$WORK/${d}_mut${m}_x${QUERY_BATCH}_gt.ibin"
        if [ ! -s "$tq" ]; then
          local tile_gt_arg=()
          [ -r "$g" ] && tile_gt_arg=(--gt "$g" --out-gt "$tg")
          python3 "$REPO/scripts/tile_queries.py" --target "$QUERY_BATCH" \
              --query "$q" --out-query "$tq" ${tile_gt_arg[@]+"${tile_gt_arg[@]}"} \
            || { echo "  tiling failed for mut$m; using the untiled set"; tq=""; }
        fi
        if [ -s "$tq" ]; then
          q="$tq"
          [ -s "$tg" ] && g="$tg"
        fi
      fi

      if [ -r "$g" ]; then
        gt_arg=(-g "$g")
      else
        echo "  NOTE: no ground truth at $g — latency only for mut$m."
      fi

      echo "  ===== k=$kk  mut$m  ($(fbin_n "$q") queries) ====="
      for mode in $QUERY_MODES; do
        local mode_arg=()
        case "$mode" in
          conventional) mode_arg=();;
          lsh) [ "$BUILD_LSH" = "1" ] || { echo "  skip lsh query (not built)"; continue; }
               mode_arg=(--lsh --prepopulated --k_ranks "$K_RANKS");;
          pq)  [ "$BUILD_PQ" = "1" ]  || { echo "  skip pq query (not built)"; continue; }
               mode_arg=(--pq --prepopulated --k_ranks "$K_RANKS");;
          rq|rabitq)
               [ "$BUILD_RQ" = "1" ]  || { echo "  skip rq query (not built)"; continue; }
               mode_arg=(--rabitq --prepopulated --k_ranks "$K_RANKS");;
          *) echo "  unknown query mode: $mode"; continue;;
        esac
        echo "  [$( stamp )] query: $mode (k=$kk, mut$m)"
        # A failure here is usually the ground truth: read_groundtruth rejects a
        # k larger than the file's own gt_k. Non-fatal; the next one still runs.
        ./build/bin/run_query -i "$idx" --dim "$DIM" -d "$DIST" -t "$SRC_DTYPE" \
            -n "$R_FIN" -k "$kk" ${mode_arg[@]+"${mode_arg[@]}"} ${host_arg[@]+"${host_arg[@]}"} \
            -q "$q" ${gt_arg[@]+"${gt_arg[@]}"} --beam_limits $BEAM_LIMITS 2>&1 \
          | tee -a "$log" \
          | grep -E "beam_width|Recall|prerotated|rotating queries|trailer" \
          || echo "  QUERY FAILED ($mode, k=$kk, mut$m) — see $log"
      done
    done
  done

  if [ "$KEEP_INDEX" = "0" ]; then
    echo "  removing $idx (KEEP_INDEX=0)"
    rm -f "$idx"
  fi
  echo
}

for d in $FEASIBLE; do
  run_stage "$d" || echo "### dataset $d failed; continuing ###"
done

echo "=== done: $(stamp) ==="
echo "Per-dataset logs in $LOGDIR/<dataset>_r<rint>_<rfin>_<estimators>_k<k_ranks>.log"
