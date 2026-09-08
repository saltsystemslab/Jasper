# Directional quantization population inside the DiskANN-style build

Status: **proposal**. Nothing here is in the tree yet. The two halves being
joined both exist: `include/jasper/index/construct_diskann.cuh` (shard + merge
build that does not require the dataset to fit in VRAM) and the directional
estimator artifacts driven from `include/jasper/index/graph.cuh`
(`generate_lsh_globals`, `populate_edge_lsh`, `generate_pq_codebooks`,
`populate_edge_pq`, `compute_vector_norms`).

## 1. Problem

Directional beam search (`beam_search/directional_beam_search.cuh`,
`beam_search/pq_beam_search.cuh`) scores a popped node's out-edges from
per-edge quantized residuals instead of fetching neighbor vectors. It needs
four artifacts:

| artifact | producer | what it reads |
| --- | --- | --- |
| `lsh_globals` (per-rank calibration) | `graph.cuh:1308` | sampled edges + both endpoints' vectors |
| `pq_codebooks` (M x 256 centroids) | `graph.cuh:1206` | sampled edge residuals |
| `edge_lshs` / `edge_pqs` (per-edge codes) | `graph.cuh:1153`, `graph.cuh:1277` | **for every edge (u -> nb): x_u and x_nb** |
| `d_vector_norms` | `graph.cuh:1142` | one pass over the stored vectors |

Row 3 is the blocker. The DiskANN builder's output is host-resident edge lists
over **global** ids (`construct_diskann.cuh:86`) with no vectors resident at
all — that is the whole point of the distance-only merge
(`construct_diskann.cuh:12-16`). Calling `populate_edge_lsh()` on that result
would require random access to the entire dataset, reintroducing exactly the
residency assumption the builder exists to avoid.

Today the only way to search a DiskANN-built index directionally is to
repopulate at query time (`cmd/run_query.cu:547-552`), which loads the whole
graph and runs the full-graph populate kernels. That works for benchmarking and
is useless for a large index.

## 2. Design

The per-edge code for (u, nb) is a function of `x_nb - x_u` and nothing else.
Every edge in the final merged graph was first produced inside some shard, and
inside a shard **both endpoints are on the device**. So:

> Populate each shard's edges while the shard graph is still resident and its
> neighbor ids are still shard-local, then carry the codes as payload through
> the distance-only merge.

Consequences:

- No extra data movement. Population is pure compute on top of vectors that are
  already in VRAM for the build.
- Scale-independent: no step ever touches a vector outside the resident shard.
- Bit-exact. The payload equals what a whole-graph populate would compute for
  that (u, nb) pair under the same globals/codebooks; the merge only permutes
  and truncates.

The ordering constraint is strict and silent if violated: population must run
**after** `construct_graph_into` and **before** `remap_shard_edges_kernel`
(`construct_diskann.cuh:495`), because after the remap the stored neighbor ids
are global and out of the shard graph's index range.

## 3. Work plan

### Step 0 — prerequisite fixes (land first, testable alone)

**0a. `populate_lsh` ignores `edge_count`.** `lsh_kernels.cuh:46` loops over all
`n_neighbors` slots and dereferences `g.get_vector(g.get_neighbor(...))` for
slots past the real degree. On the shard path this is fatal: the shard graph is
reused across shards via `graph::reload()`, which does not clear `edges`, so
slot `>= cnt` holds a stale id from a possibly *larger* previous shard and
indexes out of range. Add a `cnt` guard and zero-fill the unused slots.
`pq_populate` (`pq_kernels.cuh:273`) does break at `cnt` but leaves the tail
stale — zero-fill it too, so the on-disk trailer is deterministic.

**0b. The directional trailer is currently unreadable.**
`load_directional_graph_from_file` (`graph.cuh:1961`) computes
`base_section_size` from the *legacy* `bytes_per_node`, but `save_graph_to_file`
(`graph.cuh:1680`) writes an inline `slot_to_id` column **plus** a packed
deletion bitmask and a trailing `next_id`. The loader seeks into the middle of
the node records, the magic check fails, and it silently returns a base-only
bundle (`has_lsh = has_pq = false`). Factor the layout math into one helper that
derives the columns from the header's `bytes_per_node` the way the base loader
already does (`graph.cuh:1466-1473`) and use it in both saver and loader. The
new DiskANN saver has to interoperate with this loader, so it cannot rely on the
accident that its own base layout happens to match the stale formula.

### Step 1 — directional plumbing in `diskann_intermediate_graph`

- `static_assert` that INT_CFG and FIN_CFG agree on `use_lsh`, `k_ranks`,
  `packed_t` and `pq_m`, alongside the existing shared-type asserts
  (`construct_diskann.cuh:76-83`).
- New optional members: `edge_lshs`, `edge_pqs`, `norms` (host vectors), plus
  the trained `globals` / `codebooks` and `built_lsh` / `built_pq` flags.
  Allocated only when requested.
- New `directional_opts { build_lsh, build_pq, lsh_samples, pq_train, pq_iters,
  seed }` parameter on `build()`.
- **Fix the `max_shard` sizing formula** (`construct_diskann.cuh:388`). It
  currently counts `padded_dim * sizeof(data_t) + sizeof(edge_list_t) +
  sizeof(uint8_t)` and omits `slot_to_id` (4 B/slot) and `deleted_bits`
  (1/8 B/slot) that `graph_segment::allocate` always allocates. With directional
  storage it must also include `sizeof(edge_lsh_list_t) + sizeof(edge_pq_list_t)`,
  rounded up to the `1<<20`-slot segment granularity the lazy allocators use.
  Without this the builder picks a shard 1.3-1.8x too large (see section 4.3) and
  OOMs when the payload buffers are lazily allocated mid-build.

### Step 2 — train globals + codebooks once, on a pilot shard

Both artifacts must be identical across all shards or the codes are not
mutually comparable at query time. Cheapest correct point: immediately after the
**first** `construct_graph_into` (tasks are sorted largest-first,
`construct_diskann.cuh:423`, so task 0 is the biggest shard) and before its
remap, call `generate_lsh_globals(lsh_samples)` and
`generate_pq_codebooks(pq_train)` on the resident shard graph, cache both, and
reuse them for every shard including that one. A shard's Vamana edges are drawn
from the same local-neighbor residual distribution as the final graph's, so the
sample is representative.

Validate `padded_dim % pq_m == 0` up front rather than throwing from inside
`generate_pq_codebooks` mid-build.

If pilot-shard codebooks measurably cost recall, the escalation is to accumulate
training samples across the first few shards before populating any of them —
which means holding those shards' graphs or rebuilding them. Only do that if
measurements demand it.

### Step 3 — populate in-shard, before remap

In `build_from_buf` (`construct_diskann.cuh:467`), between
`construct_graph_into` and `remap_shard_edges_kernel`: `g.populate_edge_lsh()`
and `g.populate_edge_pq(cb)`, then copy each segment's `edge_lshs` / `edge_pqs`
slice to host alongside the existing `edges` / `edge_counts` copies
(`construct_diskann.cuh:500-517`). Comment the populate-before-remap invariant
at both sites.

### Step 4 — carry payloads through `merge_row`

`merge_row` (`construct_diskann.cuh:272`) becomes payload-aware: each candidate
carries a `(source, slot)` tag, and after the sort + dedup the selected
payloads are written back.

The in-place write is **not** safe for payloads. It works today for ids and
distances only because those were copied into `cid[]` / `cd[]` first; `out`'s
payload array is simultaneously source and destination, and sorted order can
overwrite payload slot *j* before the candidate that needs slot *j* is reached.
Copy the selected payloads into a stack scratch first, then store:
`MAXC * (K_RANKS * sizeof(packed_t) + 2 + M)` bytes, i.e. 640 B for
(R_int 32, R_fin 32, K 4, uint8) and 1.3 KB for (32, 64, 4, uint16) — stack-safe.

Zero payload slots at or past the new degree.

### Step 5 — per-vector norms

`compute_vector_norms` (`graph.cuh:1142`) wants the whole graph on device. Here,
run a chunked pass over the dataset (same chunking as `assign_shards`) or an
OpenMP host loop, into `ig.norms`. Only needed when PQ is requested; norms must
be of the **stored** (rotated, see Step 7) vectors.

### Step 6 — save

New `save_directional_to_file(data, fname)` on `diskann_intermediate_graph`:
today's base section (`construct_diskann.cuh:633`) followed by the trailer in
exactly the byte order `save_directional_graph_to_file` writes
(`graph.cuh:1860-1930`) — magic / version / flags, then
`[k_ranks, packed_size, globals, edge_lshs]`, then
`[M, K, dsub, centroids, edge_pqs]`, then `[norms]`. Writing the flat host
arrays in one shot is byte-identical to the loader's segment-wise reads, since
segments are consecutive `1<<20` chunks in global order. Use the Step-0b offset
helper so saver and loader agree by construction.

### Step 7 — global prerotation (an accuracy prerequisite, not optional)

Cross-polytope LSH is only well calibrated in an isotropic space;
`create_lsh_index` gets that from `params.prerotate = true`
(`cmd/create_lsh_index.cu:494`). The DiskANN path deliberately sets
`prerotate = false` per shard (`construct_diskann.cuh:479`, shards must share
one space), and `construct_graph`'s prerotate stages the entire dataset on the
device (`construct.cuh:1046-1058`) — unusable at scale.

Add a chunked `prerotate_dataset_in_place(view, seed, chunk_bytes)`. The
chunked centre + GEMM loop already exists inside `assign_shards` for its
temporary hashing copy (`construct_diskann.cuh:204-256`); factor it out. Run it
once at the top of `build()` when directional artifacts are requested, then drop
the now-redundant second rotation inside `assign_shards` (keep the
mean-centring). All shards, the stored vectors and the codes then live in one
rotated space, and queries rotate with the same seed.

### Step 8 — persist prerotate in the trailer (recommended)

The footgun documented at `graph.cuh:1801-1813` (rotation state is not
persisted; get it wrong and queries silently search unrotated against a rotated
index — wrong results, no error) gets worse once a build path *requires*
rotation. Bump `DIRECTIONAL_TRAILER_VERSION` to 2 with `[u8 prerotate][u32
seed]`; the loader accepts both 1 and 2.

### Step 9 — CLI and consumers

- `cmd/create_diskann_index.cu`: add `--lsh`, `--pq`, `--k_ranks`,
  `--prerotate` / `--prerotate_seed`, `--lsh_samples`, `--pq_train`; add
  directional configs `graph_config<..., true, KR, PACKEDT>` for both INT and
  FIN. Keep the table small — `k_ranks` in {4} x packed {uint8 for dim <= 128,
  uint16 otherwise} x {l2, ip} x (R_int, R_fin) in {(32,32), (32,64)}. Every
  entry instantiates the whole construct + populate stack in a single TU; split
  per-config like `ffi/` if nvcc time bites.
- `cmd/run_query.cu`: add a path that loads via
  `load_directional_graph_from_file` and skips query-time training/population
  (and skips `rotate_graph_vectors`) when a trailer is present. Today `--lsh`
  always repopulates.
- `ffi/` and `python/` already have a directional load path
  (`LoadDirectionalGraph_*`); it starts working once Step 0b lands.

### Step 10 — telemetry

Extend the phase-totals line (`construct_diskann.cuh:612`) with `train` and
`populate`, and print the extra host bytes the payload arrays cost so an
oversized run fails loudly at the start instead of by OOM in the middle.

## 4. Memory analysis

### 4.1 Unit sizes

With `index_t = uint32_t`, `data_t = __half`, bf16 edge distances, and
`padded_dim = round_up(dim, 8)`:

| structure | bytes |
| --- | --- |
| vector row | `padded_dim * 2` |
| `edge_list_t(R)` | `6R` (4 B id + 2 B bf16 dist) |
| `edge_lsh_list_t(R, K, P)` | `R*K*sizeof(P) + 2R` |
| `edge_pq_list_t(R, M)` | `R*M` |
| `edge_counts` | 1 / slot |
| `d_vector_norms` | 4 / vector |
| `slot_to_id` (device only) | 4 / slot |
| `deleted_bits` (device only) | 0.125 / slot |
| `shard_ids` (host, build only) | `n_dup * 4` / vector |

`pq_m == k_ranks` by construction (`graph.cuh:78`), so one knob sets both the
LSH rank count and the PQ subquantizer count. That coupling drives the tables
below and is revisited in section 6.

`packed_t` must hold the coordinate index in its low bits (7 for uint8, 15 for
uint16, `lsh_kernels.cuh:25-31`), so dim <= 128 can use uint8 and dim = 768
must use uint16.

### 4.2 Host memory, whole-build resident

**Dataset A — 1B vectors, dim 96** (padded 96), R_int 32, R_fin 32, K = M = 4,
uint8 packing, n_dup 2:

| item | B/vector | total |
| --- | --- | --- |
| dataset (pinned) | 192 | 178.8 GiB |
| final `edges` | 192 | 178.8 GiB |
| final `counts` | 1 | 0.9 GiB |
| `shard_ids` | 8 | 7.5 GiB |
| **subtotal, build as it exists today** | **393** | **366.0 GiB** |
| final `edge_lshs` | 192 | 178.8 GiB |
| final `edge_pqs` | 128 | 119.2 GiB |
| `norms` | 4 | 3.7 GiB |
| **subtotal, with LSH + PQ** | **717** | **667.8 GiB** |

**Dataset B — 100M vectors, dim 768** (padded 768), R_int 32, R_fin 64,
K = M = 4, uint16 packing, n_dup 2:

| item | B/vector | total |
| --- | --- | --- |
| dataset (pinned) | 1536 | 143.1 GiB |
| final `edges` | 384 | 35.8 GiB |
| final `counts` | 1 | 0.1 GiB |
| `shard_ids` | 8 | 0.7 GiB |
| **subtotal, build as it exists today** | **1929** | **179.7 GiB** |
| final `edge_lshs` | 640 | 59.6 GiB |
| final `edge_pqs` | 256 | 23.8 GiB |
| `norms` | 4 | 0.4 GiB |
| **subtotal, with LSH + PQ** | **2829** | **263.5 GiB** |

The two shapes behave very differently: at dim 96 the payloads *dominate*
(+82% over the current build), at dim 768 the raw vectors dominate and the
payloads are +47%.

Sensitivity to `k_ranks` (= M), which is the single most expensive knob:

| k_ranks | A: total host | B: total host |
| --- | --- | --- |
| 4 | 667.8 GiB | 263.5 GiB |
| 8 | 906.2 GiB | 335.0 GiB |
| 16 | 1383.0 GiB | 478.1 GiB |

Two independent observations from this table:

- A 1B x 96 build with LSH + PQ at k_ranks 4 needs ~670 GiB of host RAM, and
  ~180 GiB of that is **pinned** (`load_vectors_from_file_cast` uses
  `cudaMallocHost`, `vector.cuh:260`). Pinning that much is its own scaling
  problem, independent of this change; an mmap-backed read-only dataset view
  would be the larger win at that scale.
- dim 768 wants a much finer PQ than M = 4 (dsub = 192 dims per 8-bit
  subquantizer). Raising M to 16 for PQ quality drags `k_ranks` to 16 with it and
  pushes the build to ~478 GiB.

### 4.3 Host memory, transient (per shard of *s* nodes)

- gather buffers: `2 * s * padded_dim * 2` — one shard building, one prefetched
  (`construct_diskann.cuh:574-591`).
- in-flight merge jobs: `(MERGE_QCAP + 2) * s * job_bytes_per_node`, where
  `job_bytes_per_node = 6*R_int + 1 + 4 + lsh_int + pq_int`; `MERGE_QCAP = 2`
  (`construct_diskann.cuh:539`) plus one being built plus one in the worker.

| | job B/node today | job B/node with LSH+PQ | at s = 4M: gather | jobs | total |
| --- | --- | --- | --- | --- | --- |
| A | 197 | 517 | 1.5 GB | 8.3 GB | ~9.8 GB |
| B | 197 | 645 | 12.3 GB | 10.3 GB | ~22.6 GB |

`MERGE_QCAP = 2` was chosen when a job was only edges + counts + ids. With
payloads a job is 2.6-3.3x larger, and the queue cost scales with shard size —
so pick `n_parts` for a shard of a few million nodes rather than accepting
whatever `max_shard` allows, and consider `MERGE_QCAP = 1` for directional
builds.

### 4.4 Device memory

Per-slot cost of the reused shard graph:

| | today's formula (`:388`) | actual today | actual with LSH + PQ |
| --- | --- | --- | --- |
| A | 385 B | 389.1 B | 709.1 B |
| B | 1729 B | 1733.1 B | 2181.1 B |

On an 80 GB device with `workspace_budget = 5 GiB` (so `avail` ~ 74.1 GB after
the budget + 4 GiB slack, `construct_diskann.cuh:392-395`):

| | `max_shard` as computed today | `max_shard` actually safe | overestimate |
| --- | --- | --- | --- |
| A | 192M slots | 104M slots | 1.84x |
| B | 42.9M slots | 34.0M slots | 1.26x |

This is why Step 1's sizing fix is a correctness item, not tidiness: the payload
buffers are allocated lazily on the first `populate_edge_lsh()` call, i.e. after
the shard size has already been committed.

Fixed device costs are all negligible: codebooks `1024 * padded_dim` B (98 KB /
786 KB), PQ training `d_X` `n_train * padded_dim * 2` (7.7 MB / 61 MB at
n_train 40k), rotation matrix `dim^2 * 2` (18 KB / 1.2 MB), `assign_shards`
chunk buffers capped at 512 MB.

### 4.5 On-disk size

| section | A | B |
| --- | --- | --- |
| base (`dim*2 + 1 + 6*R_fin` per node) | 358.6 GiB | 178.9 GiB |
| trailer: `edge_lshs` | 178.8 GiB | 59.6 GiB |
| trailer: `edge_pqs` + centroids | 119.2 GiB | 23.8 GiB |
| trailer: `norms` | 3.7 GiB | 0.4 GiB |
| **total** | **660.3 GiB** | **262.7 GiB** |

Note the base section stores `dim`, not `padded_dim`, so files are slightly
smaller than the in-memory footprint for unaligned dims (both example dims are
already 8-aligned, so they match here).

### 4.6 Verdict per shape

- **Dataset B (100M x 768) at k_ranks 4: use the in-shard plan.** 263.5 GiB host
  fits a 384 GB box with room for the transients, and the payload cost is only
  +47% because vectors dominate.
- **Dataset A (1B x 96): the in-shard plan needs ~670 GiB.** Viable only on a
  768 GB-1 TB host. Below that, either build one estimator instead of two
  (LSH-only: 544.9 GiB, PQ-only: 489.0 GiB) or use the streaming variant in
  section 5. `--lsh` and `--pq` must therefore be independent flags, both
  defaulting off.

## 5. Alternative: post-merge streaming population

Populate after the merge instead. Walk the final graph in node batches; for each
batch gather the needed neighbor vectors host -> device (the pageable-malloc +
OpenMP + prefetch pipeline at `construct_diskann.cuh:448-463` is reusable), run
a variant of `populate_lsh` / `pq_populate` that resolves neighbors through an
id -> staging-slot map, and stream the payloads straight into the trailer.

- Peak extra host memory becomes one batch instead of N: **366.0 GiB for A**
  (saves 302 GiB) and **179.7 GiB for B** (saves 84 GiB).
- Codebooks can be trained on the *final* graph rather than a pilot shard.
- Costs one extra pass of `N * R_fin * padded_dim * 2` bytes of random host
  reads plus the same volume over PCIe: **6.1 TB for A**, **9.8 TB for B**.
  At a realistic 2-5 GB/s for random host gathers that is tens of minutes to
  well over an hour.
- Needs a new kernel plus the id map — strictly more code than the in-shard
  plan, which reuses `populate_edge_lsh` / `populate_edge_pq` verbatim.

Recommendation: implement the in-shard plan first (nearly free in time, reuses
existing kernels, never touches a vector outside the resident shard) and add the
streaming path only if host RAM — not build time — turns out to be the binding
constraint, as it does for 1B x 96 on a normal host.

## 6. Validation

1. **`n_parts = 1, n_dup = 1`** on ~1M vectors: the merge is a no-op, so the
   build-time payloads must match a query-time `populate_edge_lsh` on the same
   index **bit-for-bit** given the same globals. Tightest test of the payload
   plumbing.
2. **`n_parts > 1, n_dup = 2`**: assert every selected merge candidate carried a
   valid source tag; count slots at or past the degree.
3. **Recall**: `run_query --lsh` / `--pq` on the trailer-bearing index vs. the
   query-time-populated same index. Should match within noise; will not be
   identical, since codebooks are pilot-shard-trained rather than
   final-graph-trained.
4. **Trailer round-trip** through `load_directional_graph_from_file`; spot-check
   payload bytes, globals and centroids.
5. **Memory**: confirm peak host RSS matches section 4.2 and that `max_shard`
   shrinks per section 4.4 (no mid-build OOM when the payload buffers allocate).

All of this builds and runs on the remote sm_120 host; there is no nvcc on the
dev laptop.

## 7. Open questions

- **Decouple `pq_m` from `k_ranks`** (`graph.cuh:78`). One knob currently sets
  both the LSH rank count and the PQ subquantizer count, which forces high-dim
  datasets to choose between a usable PQ (large M) and a compact LSH (small K) —
  section 4.2 quantifies the cost. Splitting them is a small config change with
  a large effect on dim-768-class data.
- **Entry point.** The DiskANN saver writes `medoid = 0`
  (`construct_diskann.cuh:641`) and directional search enters at the medoid. That
  is a recall cost independent of this work, but it will show up in any
  comparison against `create_lsh_index` output.
- **Pinned dataset at 1B scale.** 178.8 GiB of `cudaMallocHost` for dataset A
  (section 4.2) is a separate scaling problem; an mmap-backed view would help
  both this path and `save_to_file`, which streams vectors from the same buffer.
