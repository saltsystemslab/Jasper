#!/bin/bash
#SBATCH --partition=SaltSystemsLab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --nodelist=d4067
#SBATCH --mem=987842M
#SBATCH --gres=gpu:1
#SBATCH --job-name=profile_pcie
#SBATCH --output=profile_pcie_%j.log
#SBATCH --error=profile_pcie_%j.err
#SBATCH --time=08:00:00
#
# PCIe bytes per query, per pop and per vector fetched, measured with hardware
# counters (Nsight Compute) on distortion_test --pops.
#
# For each (method, k_ranks, placement), `distortion_test --pops` searches once
# per BEAM_LIMITS pair with visited logging on, and ncu profiles exactly those
# search launches. So the PCIe counter and the pop / vector counts describe the
# SAME searches; scripts/ncu_pcie_summary.py pairs them:
#
#   $OUT/pcie_<dataset>.csv   one row per (method, k, placement, beam width):
#                             recall, kernel_ms, pops_per_query,
#                             vectors_per_query, pcie_bytes_per_query,
#                             pcie_bytes_per_pop, pcie_bytes_per_vector,
#                             dram_bytes_per_query
#   $OUT/raw/                 the ncu CSV, --pops CSV and log of every run
#
#   vectors_per_query   full vectors read: one per pop for lsh / pq / rabitq,
#                       one per appended neighbor (+ the medoid) for
#                       conventional — counted from the visited log
#   placement=host      the graph lives in pinned host memory (--on_host), so
#                       every vector / edge list / code read crosses PCIe
#   placement=device    the graph is on the card: PCIe bytes drop to ~0 and the
#                       traffic shows up as dram_bytes_per_query instead
#
# PCIe bytes include the edge lists and estimator codes as well as vectors, so
# pcie_bytes_per_vector is "traffic per vector the search had to look at", not
# the size of one vector. PQ's exact ||v||^2 lives in device memory, so those
# reads count as dram even when the graph is on the host. The visited log adds
# one small device-memory write per pop; it does not touch PCIe.
#
# Queries are NOT tiled here on purpose: tiled copies of the same query hit the
# same nodes and inflate L2 hits, which would understate bytes per query. QPS
# printed under ncu is meaningless (kernels are serialized and replayed) — take
# QPS from the ksweep runs.
#
# Uses the plain index from distortion_sweep.sbatch if it exists, otherwise
# builds one (single part, like that script). deep1B / sra500 indexes only live
# inside a ksweep job, so they are not handled here.
#
#   sbatch sbatch/profile_pcie.sbatch                       # sra50, CPH r=16
#   METHODS="lsh conventional" PLACEMENTS=host sbatch sbatch/profile_pcie.sbatch
#   METHODS="lsh pq rabitq conventional" K_LIST="4 16" sbatch sbatch/profile_pcie.sbatch
#   DATASET=deep10M sbatch sbatch/profile_pcie.sbatch
#   PLACEMENTS=host BEAM_LIMITS="64:256 1024:2048" sbatch sbatch/profile_pcie.sbatch
#   NCU=/path/to/ncu sbatch sbatch/profile_pcie.sbatch      # if ncu is not found

set -euo pipefail

# ── Tunables ───────────────────────────────────────────────────────────────
DATASET="${DATASET:-sra50}"                  # sra50 | deep10M
METHODS="${METHODS:-lsh}"                    # any of: lsh pq rabitq conventional
K_LIST="${K_LIST:-16}"                       # ignored for conventional
PLACEMENTS="${PLACEMENTS:-host device}"
BEAM_LIMITS="${BEAM_LIMITS:-16:128 64:256 256:512 1024:2048 1982:2048}"
K="${K:-10}"
NCU="${NCU:-}"                               # empty = search for it
NCU_MODULE="${NCU_MODULE:-Nsight/2024.7.1}"  # module that puts ncu on PATH
NCU_EXTRA="${NCU_EXTRA:-}"                   # extra ncu flags, if needed

REPO="${REPO:-$HOME/jasper}"
WORK="${WORK:-/projects/SaltSystemsLab/${USER:-$(id -un)}/distortion}"
OUT="${OUT:-$REPO/pcie_profile}"
ANN_DATA="${ANN_DATA:-/projects/SaltSystemsLab/ann_data}"
ANN_GT="${ANN_GT:-/projects/SaltSystemsLab/ann_gt}"
LOCALE="${LOCALE:-/tmp2/ann_data/locale}"

case "$DATASET" in
  sra50)   BASE="$LOCALE/sra50/base.fbin"; Q="$LOCALE/sra50/query_mut0.00.fbin"
           GT="$LOCALE/sra50/gt_mut0.00.ibin"; DIM=768;;
  deep10M) BASE="$ANN_DATA/deep/deep10M"; Q="$ANN_DATA/deep/deep10kquery"
           GT="$ANN_GT/GT_10M/deep-10M"; DIM=96;;
  *) echo "unknown DATASET=$DATASET (sra50 | deep10M)" >&2; exit 2;;
esac
DTYPE=float; R=64
IDX="${INDEX:-$WORK/${DATASET}_r${R}_${R}_plain.bin}"

# Base metric names; whichever this GPU supports get collected (.sum).
# (The L2-sector host-memory metric reads 0 for zero-copy loads on this GPU,
# job 11156006, so it is not collected.)
CANDIDATES="pcie__read_bytes dram__bytes_read gpu__time_duration"

stamp() { date '+%Y-%m-%d %H:%M:%S'; }

echo "=== PCIe profile: $DATASET ==="
echo "  host:        $(hostname)"
echo "  methods:     $METHODS   k_ranks: $K_LIST   placements: $PLACEMENTS"
echo "  beam_limits: $BEAM_LIMITS"
echo "  index:       $IDX"
echo "  queries:     $Q"
echo "  out:         $OUT"
for f in "$Q" "$GT"; do [ -r "$f" ] || { echo "missing: $f" >&2; exit 1; }; done
echo

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
cmake --build build --target create_diskann_index distortion_test -j8 >/dev/null
echo "  ok"
mkdir -p "$OUT/raw"

# ── Find ncu without changing this shell's CUDA environment ────────────────
# the binaries were built against the conda toolchain; loading the cuda module here
# could swap the runtime under it, so only borrow the module's ncu path.
find_ncu() {
  local c
  for c in "$(command -v ncu 2>/dev/null || true)" \
           "$CONDA_PREFIX/bin/ncu" $(ls -d "$CONDA_PREFIX"/nsight-compute*/ncu 2>/dev/null) \
           "$( (module load cuda >/dev/null 2>&1 && command -v ncu) 2>/dev/null || true)" \
           $(ls -d /usr/local/cuda*/bin/ncu /opt/nvidia/nsight-compute/*/ncu 2>/dev/null); do
    [ -n "$c" ] && [ -x "$c" ] && { echo "$c"; return 0; }
  done
  return 1
}
# On Explorer the Nsight module points at a full CUDA install
# (cuda/12.8.0/bin/ncu, Nsight Compute 2025.1 — new enough for sm_120). Loading
# it here could also change LD_LIBRARY_PATH under the binaries, so, as with the
# cuda module, only borrow the ncu path from a subshell.
if [ -z "$NCU" ] && [ -n "$NCU_MODULE" ]; then
  NCU=$( (module load "$NCU_MODULE" >/dev/null 2>&1 && command -v ncu) 2>/dev/null || true)
  [ -n "$NCU" ] && echo "  ncu from module $NCU_MODULE" \
                || echo "  module $NCU_MODULE gave no ncu; searching elsewhere"
fi
[ -n "$NCU" ] || NCU=$(find_ncu) || {
  echo "ncu not found. Check with: module show $NCU_MODULE ; then pass NCU=/path/to/ncu or NCU_MODULE=<name>." >&2
  exit 1; }
echo "  ncu:         $NCU ($("$NCU" --version 2>/dev/null | tail -1))"

# ── Which metrics does this GPU have? ──────────────────────────────────────
# The listing is advisory only. It is used to skip metrics it positively does
# not contain, but if it confirms neither the host-memory nor the PCIe metric
# (an error, a changed output format, a new flag name), every candidate is
# tried and profile_cell drops whatever ncu actually rejects. Job 11155566
# stopped here with every metric "missing", gpu__time_duration included,
# which is a listing failure, not a GPU without the metrics.
#
# Feed $AVAIL through here-strings, never `echo "$AVAIL" | ...`: the listing is
# huge, and under pipefail a reader that stops early (head, grep -q) SIGPIPEs
# the echo. That made every present metric test as missing (job 11155566) and
# killed the script outright at the head (job 11156006).
AVAIL=$("$NCU" --query-metrics --query-metrics-mode all 2>&1 || "$NCU" --query-metrics 2>&1 || true)
echo "  ncu --query-metrics: $(wc -l <<<"$AVAIL") lines, first ones:"
head -6 <<<"$AVAIL" | sed 's/^/    | /'
METRICS=""
for m in $CANDIDATES; do
  if grep -qE "(^|[[:space:]])${m}(\.sum)?([[:space:]]|$)" <<<"$AVAIL"; then
    METRICS="${METRICS:+$METRICS,}${m}.sum"
  fi
done
case "$METRICS" in
  *pcie__*) echo "  listing confirms: $METRICS";;
  *) echo "  listing did not confirm the PCIe metric; trying every candidate"
     METRICS=$(echo $CANDIDATES | sed 's/ /.sum,/g; s/$/.sum/');;
esac
echo "  metrics:     $METRICS"
echo

# ── Plain index ────────────────────────────────────────────────────────────
if [ -s "$IDX" ]; then
  echo "[$(stamp)] reusing index $IDX"
else
  [ -r "$BASE" ] || { echo "no index and no base at $BASE" >&2; exit 1; }
  mkdir -p "$(dirname "$IDX")"
  echo "[$(stamp)] building plain index -> $IDX"
  ./build/bin/create_diskann_index -f "$BASE" -t "$DTYPE" -d l2 --dim "$DIM" \
      --r_intermediate "$R" --r_final "$R" --n_parts 1 --n_dup 2 -a 1.2 \
      -w 10GB -i "$IDX" >| "$OUT/raw/${DATASET}_build.log" 2>&1 \
    || { echo "BUILD FAILED (see $OUT/raw/${DATASET}_build.log)" >&2; exit 1; }
fi
echo

# ── Profile ────────────────────────────────────────────────────────────────
SUMMARY="$OUT/pcie_${DATASET}.csv"
# Keep an earlier summary instead of appending rows of a different layout to it.
if [ -s "$SUMMARY" ]; then
  mv "$SUMMARY" "$OUT/pcie_${DATASET}.before_${SLURM_JOB_ID:-$(date +%s)}.csv"
  echo "  previous summary kept as $(basename "$OUT/pcie_${DATASET}.before_${SLURM_JOB_ID:-x}.csv")"
fi
profile_cell() {           # $1 method, $2 k ("" for conventional), $3 placement
  local m=$1 k=$2 place=$3 host_arg=() kernel
  case "$m" in
    lsh|pq|rabitq) kernel=graph_beam_search_kernel;;
    conventional)  kernel=beam_search_kernel;;
    *) echo "unknown method: $m" >&2; return 1;;
  esac
  [ "$place" = "host" ] && host_arg=(--on_host)
  local cell="${DATASET}_${m}${k:+_k$k}_${place}"
  local raw="$OUT/raw/${cell}.ncu.csv" log="$OUT/raw/${cell}.log" pops="$OUT/raw/${cell}.pops.csv"
  echo "[$(stamp)] $cell"
  local attempt bad
  for attempt in 1 2 3 4; do
    rm -f "$raw" "$pops"
    # distortion_test has no warmup launch, so the profiled launches are exactly
    # the BEAM_LIMITS pairs, in order, matching the --pops rows. --print-units
    # base keeps ncu from rescaling (bytes stay bytes, time stays ns).
    # Conventional search ignores k_ranks; 4 just picks a compiled config.
    "$NCU" --metrics "$METRICS" -k "regex:${kernel}" \
        --print-units base --csv --page raw --log-file "$raw" $NCU_EXTRA \
      ./build/bin/distortion_test -i "$IDX" --dim "$DIM" -t "$DTYPE" -n "$R" -k "$K" \
        -m "$m" -r "${k:-4}" ${host_arg[@]+"${host_arg[@]}"} \
        -q "$Q" -g "$GT" --pops "$pops" -b $BEAM_LIMITS >| "$log" 2>&1 \
      || echo "  run exited non-zero — see $log and $raw"
    if grep -q "ERR_NVGPUCTRPERM" "$raw" "$log" 2>/dev/null; then
      echo "Profiling counters are restricted to admins on this node (ERR_NVGPUCTRPERM)." >&2
      echo "Ask for NVreg_RestrictProfilingToAdminUsers=0 on d4067, or use nvidia-smi dmon -s t." >&2
      exit 1
    fi
    # Blackwell (sm_120) needs a recent Nsight Compute; an older one runs the
    # app but profiles nothing.
    if grep -qiE "device.*not supported|unsupported (device|gpu)|no kernels were profiled" "$raw" 2>/dev/null; then
      echo "ncu ($("$NCU" --version 2>/dev/null | tail -1)) did not profile the kernel:" >&2
      grep -iE "not supported|unsupported|no kernels" "$raw" | head -3 >&2 || true
      echo "This GPU (sm_120) likely needs a newer Nsight Compute; try NCU_MODULE=<newer> or NCU=<path>." >&2
      exit 1
    fi
    # A metric the listing claimed but ncu rejects: drop it for every later
    # run too, and redo this one.
    bad=$(grep -hiE "failed to find metric|invalid metric|unknown metric|metric.*not (found|available|supported)" \
            "$raw" "$log" 2>/dev/null \
          | grep -oE '[a-z0-9_]+__[a-z0-9_]+(\.sum)?' | head -1 || true)
    [ -n "$bad" ] || break
    bad="${bad%.sum}.sum"
    METRICS=$(echo ",$METRICS," | sed "s/,${bad},/,/; s/^,//; s/,$//")
    echo "  ncu rejected $bad; retrying with: $METRICS"
    [ -n "$METRICS" ] || { echo "no usable metrics left" >&2; exit 1; }
  done
  [ -s "$pops" ] || { echo "  no --pops output for $cell — see $log"; tail -5 "$log" | sed 's/^/    | /'; return 0; }
  python3 "$REPO/scripts/ncu_pcie_summary.py" --ncu "$raw" --pops "$pops" --queries "$Q" \
      --out "$SUMMARY" --dataset "$DATASET" --method "$m" --k_ranks "$k" --placement "$place" \
    || { echo "  no counters parsed for $cell; ncu said:"
         grep -hE "==(ERROR|WARNING)==" "$raw" 2>/dev/null | head -8 | sed 's/^/    | /' || true
         tail -3 "$log" | sed 's/^/    distortion_test| /'; }
}

for place in $PLACEMENTS; do
  for m in $METHODS; do
    if [ "$m" = "conventional" ]; then
      profile_cell conventional "" "$place"
    else
      for k in $K_LIST; do profile_cell "$m" "$k" "$place"; done
    fi
  done
done

echo
echo "=== done: $(stamp) ==="
echo "summary: $SUMMARY"
[ -s "$SUMMARY" ] && column -s, -t < "$SUMMARY" | cut -c1-200 || true
