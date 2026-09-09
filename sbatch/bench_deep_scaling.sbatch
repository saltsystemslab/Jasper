#!/bin/bash
#SBATCH --partition=SaltSystemsLab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --nodelist=d4067
#SBATCH --mem=987842M
# ^ d4067's full RealMemory. SLURM's documented suffixes are K|M|G|T, so use M
# rather than "GB". --mem=0 (all memory on the node) works too. This matters:
# the cgroup limit, not the node's free memory, is what OOM-kills the job, and
# the budget table below reads SLURM_MEM_PER_NODE precisely so it checks against
# the number that can actually kill you.
#SBATCH --gres=gpu:1
#SBATCH --job-name=deep_scaling
#SBATCH --output=deep_scaling_%j.log
#SBATCH --error=deep_scaling_%j.err
#SBATCH --time=24:00:00
#
# DiskANN-merge + directional index scaling ladder on DEEP (dim 96), from 10M
# up to the full 1B.
#
# Each stage builds one index with create_diskann_index (per-shard LSH/PQ
# population, see docs/design/directional-diskann-population.md) and then
# queries it three ways: conventional beam search, LSH-directional, and
# PQ-directional. The point of the ladder is that the interesting quantity is
# a RATIO — how much recall the shard-and-merge build gives up relative to a
# single-graph build, and whether that ratio is stable as N grows.
#
# ── Read this before launching 1B ──────────────────────────────────────────
# Host RAM, not GPU memory, is the binding constraint. The builder holds, for
# every vector, the dataset row plus the final edge list plus (when requested)
# the per-edge LSH and PQ codes — see section 4 of the design doc. At the
# defaults (1B x d96, R=64, PQ) that is ~787 GiB resident, ~178 GiB of which is
# PINNED (load_vectors_from_file_cast uses cudaMallocHost), plus ~32 GiB of
# transient buffers. Adding LSH on top does NOT fit on this node.
#
# So: run with DRY_RUN=1 first. It prints the budget for every stage against
# what this node actually has and exits without building anything.
#
#   DRY_RUN=1 ./sbatch/bench_deep_scaling.sbatch          # just the budget table
#   sbatch sbatch/bench_deep_scaling.sbatch               # the whole ladder
#   SCALES="10M 100M" sbatch sbatch/bench_deep_scaling.sbatch
#   ESTIMATORS=pq SCALES=1B sbatch sbatch/bench_deep_scaling.sbatch
#
# Build and query can be split, which is worth doing at 1B where the build runs
# for hours and the query phase wants the node to itself:
#
#   SCALES=1B STAGE_MODE=build sbatch sbatch/bench_deep_scaling.sbatch
#   SCALES=1B STAGE_MODE=query sbatch sbatch/bench_deep_scaling.sbatch
#   SCALES=1B STAGE_MODE=query INDEX=/path/to/other.bin ./sbatch/...   # any file
#
# Every stage is independent and skipped if its index already exists, so a
# timed-out or OOM-killed job can be resubmitted without redoing earlier work.

set -euo pipefail

# ── Tunables (all overridable from the environment) ────────────────────────
SCALES="${SCALES:-10M 100M 200M 1B}"
# pq rather than lsh+pq by default: at R_fin=64 the LSH codes add 384 B/vector,
# which puts 1B past this node's memory (see the R64 rows in section 4 of the
# design doc). PQ also measured ~2x LSH's recall on text2image at fewer bytes
# per node. ESTIMATORS=lsh is fine at R_fin=32.
ESTIMATORS="${ESTIMATORS:-pq}"       # lsh | pq | lsh+pq | none
QUERY_MODES="${QUERY_MODES:-conventional lsh pq}"
STAGE_MODE="${STAGE_MODE:-build+query}"    # build+query | build | query
INDEX="${INDEX:-}"                         # query an explicit file instead of
                                           # the derived $WORK/deep<scale>_... name
# R_int == R_fin == 64. Both matter, and they have to move together:
#   R_fin=64 is the degree the final graph is searched at.
#   R_int=64 is what makes that degree mean anything. Each shard contributes at
#   most R_int neighbors per node, so with R_int=32 and n_dup=2 the union is
#   <= 64 == R_fin and the merge's "keep the R_fin closest" step selects
#   nothing — you'd pay for degree 64 and get whatever the shards happened to
#   find. At R_int=64 the union is up to 128 and the merge actually prunes.
R_INT="${R_INT:-64}"
R_FIN="${R_FIN:-64}"                 # see doc section 4.2 for the RAM cost
K_RANKS="${K_RANKS:-4}"
N_DUP="${N_DUP:-2}"
ALPHA="${ALPHA:-1.2}"
DIST="${DIST:-l2}"                   # DEEP is an L2 benchmark
WORKSPACE_BUDGET="${WORKSPACE_BUDGET:-10GB}"
DEVICE_FREE_MIB="${DEVICE_FREE_MIB:-0}"    # 0 = ask nvidia-smi, else override
SHARD_FILL="${SHARD_FILL:-70}"             # % of max_shard to aim for; see below
N_PARTS="${N_PARTS:-0}"                    # 0 = derive from device capacity.
                                           # Pin it to compare two builds whose
                                           # payloads differ: bytes/slot changes
                                           # with the estimators, so the derived
                                           # value would differ too and you'd be
                                           # comparing different graphs.
K="${K:-10}"
BEAM_LIMITS="${BEAM_LIMITS:-16:128 32:128 64:128 128:256 256:512 512:1024 1024:2048}"
DRY_RUN="${DRY_RUN:-0}"
KEEP_INDEX="${KEEP_INDEX:-1}"        # 0 = delete each index after querying

DATA="${DATA:-/projects/SaltSystemsLab/ann_data/deep}"
GT="${GT:-/projects/SaltSystemsLab/ann_gt}"
QUERIES="${QUERIES:-$DATA/deep10kquery}"
WORK="${WORK:-/projects/SaltSystemsLab/zikun/deep_scaling}"
REPO="${REPO:-$HOME/jasper}"
# Separate from WORK on purpose: WORK may be node-local scratch that the job
# loses on exit, and the numbers are the part you can't rebuild cheaply. The
# SBATCH --output file lands in the submit directory and already captures
# everything, so this is a second, per-config copy.
LOGDIR="${LOGDIR:-$REPO/deep_scaling_logs}"

DIM=96
SRC_DTYPE=float

# Per-scale inputs. base.1B.fbin is the full set; the rest are prebuilt
# prefixes. Ground truth follows the GT_<scale>/<dataset>-<scale> convention
# used by the other sbatch scripts here; a stage whose GT is missing still runs,
# it just reports latency without recall.
scale_n()    { case "$1" in 10M) echo 10000000;; 100M) echo 100000000;;
                            200M) echo 200000000;; 1B) echo 1000000000;;
                            *) echo 0;; esac; }
scale_base() { case "$1" in 1B) echo "$DATA/base.1B.fbin";;
                            *) echo "$DATA/deep$1";; esac; }
scale_gt()   { case "$1" in 1B) echo "$GT/GT_1B/deep-1B";;
                            *) echo "$GT/GT_$1/deep-$1";; esac; }

case "$ESTIMATORS" in
  lsh)     BUILD_LSH=1; BUILD_PQ=0;;
  pq)      BUILD_LSH=0; BUILD_PQ=1;;
  lsh+pq)  BUILD_LSH=1; BUILD_PQ=1;;
  none)    BUILD_LSH=0; BUILD_PQ=0;;
  *) echo "ESTIMATORS must be one of: lsh pq lsh+pq none" >&2; exit 2;;
esac

# ── Budget model (mirrors docs/design/directional-diskann-population.md §4) ──
# Bytes of host memory the builder holds resident, per vector. Keep this in sync
# with the struct sizes: an over-optimistic estimate here means an OOM hours in.
PADDED=$DIM                          # dim 96 is already a multiple of 8
PACKED_BYTES=$([ "$DIM" -le 128 ] && echo 1 || echo 2)

host_per_vec() {
  local per=0
  per=$(( per + PADDED * 2 ))                       # dataset row (__half, pinned)
  per=$(( per + R_FIN * 6 ))                        # edge_list_t: 4B id + 2B bf16
  per=$(( per + 1 ))                                # edge_counts
  per=$(( per + N_DUP * 4 ))                        # shard_ids
  if [ "$BUILD_LSH" = "1" ]; then
    per=$(( per + R_FIN * K_RANKS * PACKED_BYTES + R_FIN * 2 ))   # edge_lsh_list
  fi
  if [ "$BUILD_PQ" = "1" ]; then
    per=$(( per + R_FIN * K_RANKS ))                # edge_pq_list
    per=$(( per + 4 ))                              # exact ||v||^2
  fi
  echo "$per"
}

# Device bytes per slot of the shard graph, mirroring build()'s per_vec: a
# graph_segment always allocates vectors + edges + edge_counts + slot_to_id +
# a deletion bit, and the directional payload buffers are lazily added on top.
device_per_slot() {
  local per=$(( PADDED * 2 + R_INT * 6 + 1 + 4 + 1 ))
  [ "$BUILD_LSH" = "1" ] && per=$(( per + R_INT * K_RANKS * PACKED_BYTES + R_INT * 2 ))
  [ "$BUILD_PQ"  = "1" ] && per=$(( per + R_INT * K_RANKS ))
  echo "$per"
}

# Host bytes per node held by ONE queued merge job: the shard's edge rows, its
# counts, its global ids, and its payload rows.
merge_job_per_node() {
  local per=$(( R_INT * 6 + 1 + 4 ))
  [ "$BUILD_LSH" = "1" ] && per=$(( per + R_INT * K_RANKS * PACKED_BYTES + R_INT * 2 ))
  [ "$BUILD_PQ"  = "1" ] && per=$(( per + R_INT * K_RANKS ))
  echo "$per"
}

# Largest shard the device can build, in the builder's own terms: whole
# vectors_per_segment segments out of (free - workspace - 4 GiB slack).
max_shard() {
  local per_slot per_seg n_seg avail
  per_slot=$(device_per_slot)
  per_seg=$(( per_slot * 1048576 ))
  avail=$(( DEVICE_FREE_BYTES - WORKSPACE_BYTES - 4 * 1073741824 ))
  [ "$avail" -lt 0 ] && { echo 0; return; }
  n_seg=$(( avail / per_seg ))
  echo $(( n_seg * 1048576 ))
}

# What a loaded index occupies at QUERY time, per node. Unlike the on-disk
# figure this includes slot_to_id and the norms, because load_graph_from_file
# materializes them. run_query puts all of this on the device unless told
# otherwise, so this is what decides --on_host.
query_resident_per_node() {
  local per=$(( PADDED * 2 + R_FIN * 6 + 1 + 4 ))
  [ "$BUILD_LSH" = "1" ] && per=$(( per + R_FIN * K_RANKS * PACKED_BYTES + R_FIN * 2 ))
  [ "$BUILD_PQ"  = "1" ] && per=$(( per + R_FIN * K_RANKS + 4 ))
  echo "$per"
}

# Base section: vectors + edge counts + edge lists, at the packed dim. This is
# the floor for an index of this N and R_fin — no choice of estimators reduces
# it, which is worth knowing before hunting for savings in the trailer.
base_disk_per_vec() { echo $(( DIM * 2 + 1 + R_FIN * 6 )); }

disk_per_vec() {
  local per
  per=$(base_disk_per_vec)
  if [ "$BUILD_LSH" = "1" ]; then
    per=$(( per + R_FIN * K_RANKS * PACKED_BYTES + R_FIN * 2 ))
  fi
  if [ "$BUILD_PQ" = "1" ]; then per=$(( per + R_FIN * K_RANKS + 4 )); fi
  echo "$per"
}

# n_parts: the FEWEST shards whose pieces still fit on the device.
#
# Fewer, larger shards is strictly better for recall — a shard's Vamana graph is
# built over only its own members, so a larger shard sees more of the dataset
# and produces better edges. Splitting exists solely because a shard has to fit
# in device memory, so n_parts should be driven by max_shard and nothing else.
#
# SHARD_FILL leaves headroom because cross-polytope shards are NOT equal-sized.
# A shard over max_shard is not an error — build() slices it into sub-shards —
# but a sub-shard is an arbitrary slice of a bucket group rather than a
# locality-based partition, so it costs quality. Aiming at ~70% of max_shard
# keeps ordinary imbalance from triggering that.
#
# The 2*dim cap is a hard property of the sharding kernel, not a heuristic:
# polytope_lsh_assign_kernel buckets a vector as (2*coord + sign) % n_parts with
# coord < dim, so beyond 2*dim the higher shard ids are never produced and those
# shards come out empty.
n_parts_for() {
  local n=$1 cap=$(( 2 * DIM )) target p
  if [ "$N_PARTS" != "0" ]; then echo "$N_PARTS"; return; fi
  target=$(( $(max_shard) * SHARD_FILL / 100 ))
  [ "$target" -lt 1 ] && { echo 0; return; }   # nothing fits on the device
  p=$(( (n * N_DUP + target - 1) / target ))
  [ "$p" -lt 1 ] && p=1
  [ "$p" -gt "$cap" ] && p=$cap
  echo "$p"
}

# A pinned N_PARTS is not checked against the device by n_parts_for, so verify
# it here: too few shards and every one of them gets sliced into sub-shards,
# which costs recall silently.
check_pinned_parts() {
  local n=$1 p=$2 shard ms
  [ "$N_PARTS" = "0" ] && return 0
  ms=$(max_shard)
  shard=$(( n * N_DUP / p ))
  [ "$shard" -gt "$n" ] && shard=$n
  if [ "$shard" -gt "$ms" ]; then
    echo "  WARNING: N_PARTS=$p gives $(mnodes "$shard")-node shards but the device"
    echo "           holds $(mnodes "$ms"); build() will split them into sub-shards,"
    echo "           which partitions by slice rather than by locality."
  fi
  if [ "$p" -gt "$(( 2 * DIM ))" ]; then
    echo "  WARNING: N_PARTS=$p exceeds 2*dim=$(( 2 * DIM )); shards above that id"
    echo "           are never produced by the sharding kernel and stay empty."
  fi
}

gib() { awk -v b="$1" 'BEGIN{printf "%.1f", b/1073741824}'; }
mnodes() { awk -v n="$1" 'BEGIN{printf "%.1fM", n/1000000}'; }

# "10GB" / "512MB" -> bytes, matching create_diskann_index's parse_size, which
# treats the suffixes as powers of 1024.
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

# Free device memory, which is what build() actually budgets against. Prefer
# asking the GPU; fall back to the documented 96 GiB card minus a little for
# context/driver overhead (an idle H100 NVL reports ~96683 MiB free of 97250).
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

# Under SLURM the job's cgroup limit binds, NOT the node's free memory: exceed
# --mem and the job is OOM-killed even though /proc/meminfo showed room. Prefer
# what SLURM granted (SLURM_MEM_PER_NODE is in MiB).
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
disk_avail_bytes() { df -B1 --output=avail "$1" 2>/dev/null | tail -1 | tr -d ' ' || echo 0; }

echo "=== deep scaling ladder ==="
echo "  host:        $(hostname)"
echo "  scales:      $SCALES"
echo "  estimators:  $ESTIMATORS (lsh=$BUILD_LSH pq=$BUILD_PQ, k_ranks=$K_RANKS)"
echo "  degrees:     R_int=$R_INT R_fin=$R_FIN  n_dup=$N_DUP  alpha=$ALPHA"
echo "  distance:    $DIST   dim=$DIM (packed_t=${PACKED_BYTES}B)"
echo "  query modes: $QUERY_MODES"
echo "  workdir:     $WORK"
echo "  logs:        $LOGDIR"

mkdir -p "$WORK" "$LOGDIR"
MEM_TOTAL=$(mem_total_bytes)
MEM_AVAIL=$(mem_available_bytes)
DISK_AVAIL=$(disk_avail_bytes "$WORK")
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

[ "$MAX_SHARD" -gt 0 ] || {
  echo "Nothing fits on the device: $(device_per_slot) B/slot leaves no room after" >&2
  echo "the $WORKSPACE_BUDGET workspace + 4 GiB slack. Lower WORKSPACE_BUDGET." >&2
  exit 1
}

# ── Budget table ───────────────────────────────────────────────────────────
HPV=$(host_per_vec)
DPV=$(disk_per_vec)
MJPN=$(merge_job_per_node)
printf "%-6s %11s %6s %8s %10s %10s %9s %10s %s\n" \
  scale N parts shard host_GiB gather_GiB merge_GiB disk_GiB verdict
FEASIBLE=""
RETAINED=0      # disk already committed to indexes this run keeps around
DISK_SHORT=0
for s in $SCALES; do
  n=$(scale_n "$s")
  [ "$n" = "0" ] && { echo "unknown scale: $s" >&2; exit 2; }
  p=$(n_parts_for "$n")
  # n_dup duplicates are dropped when they land in the same shard, so a shard
  # can never hold more than N distinct vectors — which is the binding limit
  # once n_parts approaches n_dup.
  shard=$(( n * N_DUP / p ))
  [ "$shard" -gt "$n" ] && shard=$n

  hb=$(( n * HPV ))                          # resident for the whole build
  gb=$(( 2 * shard * PADDED * 2 ))           # gather buffer + its prefetch
  # Merge jobs are one GRAPH SEGMENT each (2^20 rows), not one shard, so this is
  # a constant rather than something that scales with shard size:
  # MERGE_QCAP=8 queued, +1 being filled, +1 in the worker.
  chunk=$shard; [ "$chunk" -gt 1048576 ] && chunk=1048576
  mb=$(( 10 * chunk * MJPN ))
  db=$(( n * DPV ))
  ram_total=$(( hb + gb + mb ))

  # With KEEP_INDEX=1 the stages accumulate on disk, so each one is checked
  # against what earlier stages have left, not against the original free space.
  disk_left=$DISK_AVAIL
  [ "$KEEP_INDEX" = "1" ] && disk_left=$(( DISK_AVAIL - RETAINED ))

  verdict="ok"
  if [ "$ram_total" -gt "$MEM_AVAIL" ]; then
    verdict="SKIP: needs $(gib "$ram_total") GiB RAM"
  elif [ "$(( ram_total * 100 / MEM_AVAIL ))" -ge 90 ]; then
    verdict="ok, but TIGHT: $(( ram_total * 100 / MEM_AVAIL ))% of RAM"
  elif [ "$db" -gt "$disk_left" ]; then
    verdict="SKIP: needs $(gib "$db") GiB disk, $(gib "$disk_left") GiB left"
    DISK_SHORT=1
  fi
  printf "%-6s %11s %6s %8s %10s %10s %9s %10s %s\n" \
    "$s" "$n" "$p" "$(mnodes "$shard")" "$(gib "$hb")" "$(gib "$gb")" \
    "$(gib "$mb")" "$(gib "$db")" "$verdict"
  check_pinned_parts "$n" "$p"
  case "$verdict" in
    ok*) FEASIBLE="$FEASIBLE $s"
         [ "$KEEP_INDEX" = "1" ] && RETAINED=$(( RETAINED + db ));;
  esac
done
echo
echo "  per-vector: host ${HPV}B, disk ${DPV}B; per queued merge node: ${MJPN}B"
echo "  feasible stages:${FEASIBLE:- none}"
for s in $FEASIBLE; do
  n=$(scale_n "$s")
  r=$(( n * $(query_resident_per_node) ))
  [ "$r" -gt "$DEVICE_FREE_BYTES" ] && \
    echo "  $s: loaded index is $(gib "$r") GiB > device, so queries use --on_host (slow)"
done
if [ "$KEEP_INDEX" = "1" ] && [ "$RETAINED" -gt 0 ]; then
  echo "  indexes retained across the run: $(gib "$RETAINED") GiB"
  echo "  (KEEP_INDEX=0 deletes each one after querying, so only the largest"
  echo "   stage has to fit at any moment)"
fi
if [ "$DISK_SHORT" = "1" ]; then
  echo
  echo "  Disk is the binding constraint. The floor for one index at R_fin=$R_FIN is the"
  echo "  base section alone — $(base_disk_per_vec) B/vector, i.e. no choice of estimators"
  echo "  gets you under it:"
  for s in $SCALES; do
    n=$(scale_n "$s")
    echo "    $s: $(gib $(( n * $(base_disk_per_vec) ))) GiB base + $(gib $(( n * (DPV - $(base_disk_per_vec)) ))) GiB trailer"
  done
  echo "  Point WORK= at a filesystem with room (df -h), or use KEEP_INDEX=0."
fi

if [ "$DRY_RUN" != "0" ]; then
  echo
  echo "DRY_RUN set — nothing built."
  echo
  echo "Levers, cheapest first:"
  echo "  KEEP_INDEX=0        delete each index after querying, so only the"
  echo "                      largest stage needs to fit on disk at once"
  echo "  WORK=<big fs>       somewhere with room; df -h to find one"
  echo "  ESTIMATORS=pq       drops the LSH trailer (PQ beat LSH by 2x on"
  echo "                      text2image at less bytes/node)"
  echo "  SHARD_FILL=40       smaller shards: less gather+merge RAM, some recall"
  echo "  WORKSPACE_BUDGET=5GB  bigger max_shard, so fewer shards"
  echo
  echo "merge_GiB is a constant, not a function of shard size: a merge job is one"
  echo "2^20-row graph segment, so the queue costs the same whether shards are 10M"
  echo "or 100M nodes. That is what lets n_parts be chosen purely from device"
  echo "capacity. gather_GiB does still scale with shard size — graph::reload"
  echo "wants a shard's vectors as one contiguous upload."
  exit 0
fi
[ -n "${FEASIBLE// /}" ] || { echo "No stage fits on this node; aborting." >&2; exit 1; }

# ── Environment + build (mirrors the other sbatch scripts here) ────────────
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
  local s=$1
  local n base gt p idx log
  n=$(scale_n "$s")
  base=$(scale_base "$s")
  gt=$(scale_gt "$s")
  p=$(n_parts_for "$n")
  idx="${INDEX:-$WORK/deep${s}_r${R_INT}_${R_FIN}_${ESTIMATORS}_k${K_RANKS}.bin}"
  # Log name carries the estimators too, so separate pq / lsh runs don't
  # interleave in one file.
  log="$LOGDIR/deep${s}_${ESTIMATORS}.log"

  local do_build=0 do_query=0
  case "$STAGE_MODE" in
    build+query) do_build=1; do_query=1;;
    build)       do_build=1;;
    query)       do_query=1;;
    *) echo "STAGE_MODE must be build+query, build, or query" >&2; return 1;;
  esac

  echo "############### $s ($n vectors, n_parts=$p, $STAGE_MODE) ###############"
  [ "$do_build" = "0" ] || [ -r "$base" ] || {
    echo "  base missing: $base — skipping"; return 0; }
  [ "$do_query" = "0" ] || [ -r "$QUERIES" ] || {
    echo "  queries missing: $QUERIES — skipping"; return 0; }

  local gt_arg=()
  if [ -r "$gt" ]; then
    gt_arg=(-g "$gt")
  else
    echo "  NOTE: no ground truth at $gt — latency only, no recall."
    echo "        Set GT=... or drop a file there to get recall numbers."
  fi

  local est_arg=()
  [ "$BUILD_LSH" = "1" ] && est_arg+=(--lsh)
  [ "$BUILD_PQ"  = "1" ] && est_arg+=(--pq)

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
        -w "$WORKSPACE_BUDGET" "${est_arg[@]}" --k_ranks "$K_RANKS" \
        -i "$idx" 2>&1 | tee -a "$log" \
      || { echo "  BUILD FAILED for $s (see $log)"; return 1; }
    echo "  [$( stamp )] built, $(gib "$(stat -c %s "$idx")") GiB on disk"
  fi

  # run_query materializes the whole index in device memory unless told
  # otherwise. Past ~130M nodes at these degrees that exceeds a 96 GB card, so
  # fall back to --on_host, which keeps the graph in pinned host memory and
  # reaches it over PCIe. Correct, but far slower per hop — and page-locking
  # hundreds of GiB takes minutes on its own, so expect a long startup.
  local resident=$(( n * $(query_resident_per_node) ))
  local host_arg=()
  if [ "$resident" -gt "$DEVICE_FREE_BYTES" ]; then
    host_arg=(--on_host)
    echo "  loaded index is $(gib "$resident") GiB vs $(gib "$DEVICE_FREE_BYTES") GiB of device"
    echo "  -> querying with --on_host (pinned host memory, PCIe-resident)"
  fi

  # Conventional search is the baseline the directional numbers are a fraction
  # of. It works on a prerotated index because run_query reads the rotation out
  # of the trailer and rotates queries to match.
  #
  # Each mode is a separate run_query process, so each one re-reads the whole
  # index from disk and (on the --on_host path) page-locks it again. At 1B that
  # is the dominant cost of the query phase, not the search itself.
  [ "$do_query" = "1" ] || { echo "  (STAGE_MODE=$STAGE_MODE, no queries)"; echo; return 0; }
  for mode in $QUERY_MODES; do
    local mode_arg=()
    case "$mode" in
      conventional) mode_arg=();;
      lsh) [ "$BUILD_LSH" = "1" ] || { echo "  skip lsh query (not built)"; continue; }
           mode_arg=(--lsh --prepopulated --k_ranks "$K_RANKS");;
      pq)  [ "$BUILD_PQ" = "1" ]  || { echo "  skip pq query (not built)"; continue; }
           mode_arg=(--pq --prepopulated --k_ranks "$K_RANKS");;
      *) echo "  unknown query mode: $mode"; continue;;
    esac
    echo "  [$( stamp )] query: $mode"
    ./build/bin/run_query -i "$idx" --dim "$DIM" -d "$DIST" -t "$SRC_DTYPE" \
        -n "$R_FIN" -k "$K" "${mode_arg[@]}" "${host_arg[@]}" \
        -q "$QUERIES" "${gt_arg[@]}" --beam_limits $BEAM_LIMITS 2>&1 \
      | tee -a "$log" \
      | grep -E "beam_width|Recall|prerotated|rotating queries|trailer" \
      || echo "  QUERY FAILED ($mode) — see $log"
  done

  if [ "$KEEP_INDEX" = "0" ]; then
    echo "  removing $idx (KEEP_INDEX=0)"
    rm -f "$idx"
  fi
  echo
}

for s in $FEASIBLE; do
  run_stage "$s" || echo "### stage $s failed; continuing ###"
done

echo "=== done: $(stamp) ==="
echo "Per-stage logs in $LOGDIR/deep<scale>_<estimators>.log"
echo
echo "What to look at:"
echo "  1. Recall@10 of 'conventional' vs the single-graph create_lsh_index"
echo "     baseline at the same beam width. That ratio is the cost of the"
echo "     shard-and-merge build, separate from any estimator."
echo "  2. lsh and pq recall as a FRACTION of this index's own conventional"
echo "     recall — that isolates estimator quality from graph quality."
echo "  3. Whether either ratio degrades as N grows. Both should be roughly"
echo "     flat; a ratio that falls with N points at the merge (try N_DUP=3"
echo "     or a larger R_INT) rather than at the codes."
echo "  4. The 'phase totals' line in the build log — populate and rotate are"
echo "     the phases this work added; everything else predates it."
