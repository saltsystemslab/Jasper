# Directional quantization population inside the DiskANN-style build

Status: **all steps landed**. The two halves being joined both exist:
`include/jasper/index/construct_diskann.cuh` (shard + merge build that does not
require the dataset to fit in VRAM) and the directional estimator artifacts
driven from `include/jasper/index/graph.cuh` (`generate_lsh_globals`,
`populate_edge_lsh`, `generate_pq_codebooks`, `populate_edge_pq`,
`compute_vector_norms`). The feature is complete end to end:

```
create_diskann_index -f DATA -t float -d ip --dim 200 -n 64 --r_final 64 \
    --n_parts 64 --workspace_budget 10GB --lsh --pq -i index.bin
run_query -i index.bin --dim 200 -d ip -n 64 --lsh --prepopulated \
    -q QUERIES -g GT
```

The index records its own rotation, so no seed has to be carried between the
two commands.

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

### Step 0 — prerequisite fixes — DONE

**0a. `populate_lsh` ignored `edge_count`.** It looped over all `n_neighbors`
slots and dereferenced `g.get_vector(g.get_neighbor(...))` for slots past the
real degree. On the shard path that is fatal: the shard graph is reused via
`graph::reload()`, which does not clear `edges`, so slot `>= cnt` holds a stale
id from a possibly *larger* previous shard and indexes out of range. The loop is
now bounded by `cnt`, with an `INVALID`-neighbor guard inside it and a
zero-filled tail. `pq_populate` broke at `cnt` correctly but left the tail
stale; it got the same `INVALID` guard and tail zero-fill, so persisted codes
are deterministic rather than inherited from whatever the buffer held before.
Both the bound and the guard are block-uniform, which `pq_block_argmin`'s
`__syncthreads()` requires.

**0b. The directional trailer was unreadable.**
`load_directional_graph_from_file` computed `base_section_size` from the
*legacy* `bytes_per_node`, but `save_graph_to_file` writes an inline
`slot_to_id` column **plus** a packed deletion bitmask and a trailing `next_id`.
The loader seeked into the middle of the node records, the magic check failed,
and it silently returned a base-only bundle — so **every** save/load round-trip
lost the LSH and PQ artifacts, after which
`Graph.directional_search()` raised "requires a graph built with build_lsh=True"
(`python/jasper/__init__.py:550`).

Fixed with `base_section_bytes<graph_cfg>(dim, n_vectors, bytes_per_node)` and
the file-reading wrapper `directional_trailer_offset<graph_cfg>(fname, dim)`,
which mirror the base loader's `bpn_*` ladder and are now the single source of
the offset for both sides. The saver additionally asserts that the base section
it just wrote ends exactly where the loader will look, so a future change to the
base layout fails loudly at save time instead of silently producing an
unreadable trailer. Files written by the old saver load correctly under the new
loader, since the trailer always sat at the true end of the base section.

### Step 1 — directional plumbing in `diskann_intermediate_graph` — DONE

- `static_assert`s that INT_CFG and FIN_CFG agree on `use_lsh`, `k_ranks`,
  `packed_t`, `pq_m` and `pq_k`, alongside the existing shared-type asserts.
- The payload lives in `diskann_directional_state<fin_graph_cfg>` (`ig.dir`):
  `edge_lshs`, `edge_pqs`, the trained `globals` / `codebooks`, and
  `has_lsh` / `has_pq`. It is a **specialization on `use_lsh`** rather than a
  set of plain members, because at `k_ranks == 0` `edge_lsh_list` degenerates to
  a zero-length array — which is why `graph_segment` only ever names it through
  a pointer. A non-directional config instantiates the empty primary template.
  `diskann_shard_payload<int_graph_cfg>` does the same for the in-flight shard
  rows. Everything else is behind `if constexpr (USE_LSH)`, which is enough to
  keep those types uninstantiated because they are dependent on the class's own
  template parameters.
- `diskann_directional_opts { build_lsh, build_pq, lsh_samples, lsh_seed,
  pq_train, pq_iters, pq_seed }` is a defaulted trailing parameter of `build()`,
  so existing call sites are unaffected. Both estimators default **off**.
- **`max_shard` sizing fixed.** `per_vec` now counts `slot_to_id` (4 B/slot) and
  the deletion bit, which the old formula omitted, plus the directional payload
  buffers. The budget is also computed in whole **segments**, since
  `graph::allocate` rounds the slot count up to a multiple of `1<<20` and
  allocates every slot in each segment — sizing by exact vector count
  under-counted by up to a full segment. A budget too small for even one segment
  now throws with the numbers, instead of failing in a bare `cudaMalloc`.

### Step 2 — train globals + codebooks once, on a pilot shard — DONE

Trained inside `build_from_buf` on the first task — tasks are sorted
largest-first, so that is the biggest shard — and cached in `ig.dir` for every
later shard. A shard's Vamana edges come from the same local-neighbour residual
distribution as the final graph's, so it is a representative sample, and the
models *must* be identical across shards or the codes are not mutually
comparable at search time. `padded_dim % pq_m` is validated at the top of
`build()`, before the long shard-assignment pass.

If pilot-shard codebooks measurably cost recall, the escalation is to accumulate
training samples across the first few shards before populating any of them —
which means holding those shards' graphs or rebuilding them. Only do that if
measurements demand it.

### Step 3 — populate in-shard, before remap — DONE

`g.populate_edge_lsh()` / `g.populate_edge_pq(cb)` run in `build_from_buf`
between `construct_graph_into` and `remap_shard_edges_kernel`, and each
segment's `edge_lshs` / `edge_pqs` slice is copied back alongside `edges` /
`edge_counts`. The populate-before-remap invariant is commented at the call
site: it is silent if violated, since after the remap the stored ids are global
and index past the end of the shard graph's segments.

### Step 4 — carry payloads through `merge_row` — DONE

`merge_row` takes an optional `row_payload_t` (pointers to the node's `out`/`in`
payload rows, null when not directional). Candidates carry a `(source, slot)`
tag through the sort, and after selection the payloads are copied via
`copy_lsh_edge` / `copy_pq_edge`, which handle the `R_INT -> R_FIN` stride
change (PQ is subspace-major, so its stride really does differ).

The in-place write is **not** safe for payloads. It works for ids and distances
only because those were copied into `cid[]` / `cd[]` first; `out`'s payload row
is simultaneously source and destination, and sorted order can overwrite payload
slot *j* before the candidate that needs slot *j* is reached. The fix is to
snapshot the destination row (`const fin_edge_lsh_t prev = *pay.out_lsh`) and
copy from the snapshot — one row of stack, up to ~2.2 KB at
(R_fin 64, K 16, uint16). `tests/test_diskann_merge.cu` pins exactly this case.

Slots at or past the new degree need no clearing: the arrays start
value-initialized and `out_count` never shrinks, since every surviving `out` id
is also a candidate.

### Step 5 — per-vector norms — DONE

`graph::compute_vector_norms` wants the whole graph on device, which this build
never has. `diskann_intermediate_graph::compute_norms` streams the dataset
through the device in chunks instead (same pattern as `assign_shards`), driving
a new `vector_norms_kernel` in `construct_diskann_kernels.cuh`, and lands the
result in `ig.dir.norms`. Run only when PQ is requested, and last — after the
shard graph and workspace are freed, so the chunk buffers have the device to
themselves. Norms are of the **stored** vectors, so this stays correct once
step 7 prerotates the dataset up front.

### Step 6 — save — DONE

`save_directional_to_file(data, fname)` writes today's base section via
`save_to_file`, then appends the trailer in exactly the byte order
`save_directional_graph_to_file` uses — magic / version / flags, then
`[k_ranks, packed_size, globals, edge_lshs]`, then
`[M, K, dsub, centroids, edge_pqs]`, then `[norms]`.

The reader's per-block loops are per graph segment; the arrays here are flat in
global id order and segments are consecutive `vectors_per_segment` chunks of
that same order, so a flat write reproduces the same bytes. It is chunked at
segment size anyway, to bound the per-call write and catch short writes.
The step-0b offset helper is used for the same assert the graph saver
makes, so the writer cannot silently emit a trailer the reader won't find.
Layout is pinned byte-for-byte by `DiskannDirectionalSave.TrailerLayoutAndOffsets`
in `tests/test_graph_format.cu` (host-only — it exercises an LSH+norms trailer,
since only the PQ centroids need a device).

### Step 7 — global prerotation — DONE

Cross-polytope LSH is only well calibrated in an isotropic space;
`create_lsh_index` gets that from `params.prerotate = true`. The DiskANN path
deliberately sets `prerotate = false` per shard (shards must share one space),
and `construct_graph`'s prerotate stages the entire dataset on the device —
unusable at scale.

`prerotate_dataset(view, seed, chunk_bytes)` rotates the dataset **in place**,
streamed chunk by chunk, and `build()` runs it once before `assign_shards`
whenever an estimator was requested (`opts.rotating()`). In place matters: the
caller hands the same buffer to `save_to_file`, so the index stores the rotated
vectors and is an index *of the rotated space*.

`assign_shards` takes `already_rotated`, which skips its own hashing rotation —
a second random rotation of an already-isotropic space buys nothing and costs a
GEMM plus a chunk buffer per pass. Mean-centring still runs; it removes the DC
component, which prerotation does not touch.

No centring is applied to the stored vectors, matching `construct_graph`'s
prerotate: centring would change the vectors, and therefore inner products.
Residuals are already mean-free, which is what the estimator sees.

### Step 8 — persist prerotate in the trailer — DONE

The footgun this closes: rotation state used not to be persisted, so getting it
wrong searched unrotated queries against a rotated index — plausible nonsense,
no error — and this build path rotates by default.

`directional_rotation_info { known, prerotate, seed }` is now written as a
trailer block and handed back in `directional_graph_bundle::rotation`. `known`
distinguishes "the file says nothing" (every older file, and any writer that
didn't know) from "known and NOT rotated", which is a definitive instruction to
a reader; only the former falls back to caller-supplied values.

**A new flag bit (3), not a version bump** — deviating from what this doc
originally proposed. The reader rejects the whole trailer on an unrecognized
version, so bumping to 2 would make new indexes silently lose their LSH/PQ
artifacts under an older binary. An unknown flag bit is ignored instead, and the
block is written **last**, after the norms, so an older reader stops exactly
where it always did. That keeps the format additive in both directions, which is
what the header comment claims of it. `RotationBlockIsLastAndOptional` in
`tests/test_graph_format.cu` pins that placement, the "known but not rotated"
state, and the 64-bit seed round-trip.

The seed is stored as u64 to match `set_rotation_matrix`'s parameter, since
narrowing it anywhere would silently produce a different rotation.

Consumers:

- `run_query --prepopulated` takes the file's word over `--rotate_seed` /
  `--no_rotate`, printing a note when they disagree.
- `run_query`'s **plain, non-directional** path rotates its queries when the
  index says it is prerotated. Rotation is orthogonal, so conventional search
  over a rotated index is valid — but only with matching queries, and without
  this it silently benchmarked nonsense. That path can't call
  `load_directional_graph_from_file` (it requires `use_lsh`) and couldn't size
  the estimator blocks anyway, hence `read_rotation_info`, which finds the
  record from EOF. It is also what makes "how good is the merged graph itself?"
  measurable on a directional index instead of needing a second unrotated
  build.
- `LoadDirectionalGraph_*` does the same for its `prerotate`/`prerotate_seed`
  arguments, so `Graph.load()` no longer depends on the caller remembering.
  `GetDirectionalFlags_*` grew bit 2 (rotation in effect) so `Graph.prerotate`
  reports what the loader actually settled on rather than what was passed.
- `SaveDirectionalGraph_*` records what the handle knows, which required
  keeping `prerotate_seed` in `dir_meta` (it previously kept only the derived
  rotation matrix).

### Step 8 — persist prerotate in the trailer (recommended)

The footgun documented at `graph.cuh:1801-1813` (rotation state is not
persisted; get it wrong and queries silently search unrotated against a rotated
index — wrong results, no error) gets worse once a build path *requires*
rotation. Bump `DIRECTIONAL_TRAILER_VERSION` to 2 with `[u8 prerotate][u32
seed]`; the loader accepts both 1 and 2.

### Step 9 — CLI and consumers — DONE

- `cmd/create_diskann_index.cu`: `--lsh`, `--pq`, `--k_ranks`, `--dim`,
  `--lsh_samples`, `--pq_train`, `--no_prerotate`, `--prerotate_seed`. A
  directional build saves through `save_directional_to_file` and then releases
  the codebooks with `ig.dir.free_models()`.
  `JASPER_FOR_EACH_DIR_CONFIG` holds 8 entries: `k_ranks = 4` x packed
  {uint8 for dim <= 128, uint16 above} x {l2, ip} x (R_int, R_fin) in
  {(32,32), (32,64)}. `--dim` is required with `--lsh`/`--pq` because it selects
  the packed width, exactly as `run_query` does it. Only `k_ranks = 4` is
  compiled in — each row instantiates the whole construct + populate stack
  twice (INT and FIN) in a single TU, so adding a `k_ranks` dimension doubles
  build time; split per-config like `ffi/` before growing this much further.
- `cmd/run_query.cu`: `--prepopulated` loads through
  `load_directional_graph_from_file` and uses the trailer's codes and
  calibration instead of recomputing them. It skips `rotate_graph_vectors`
  (the stored vectors are already rotated) but still rotates the **queries**,
  so `--rotate_seed` must carry the build's `--prerotate_seed`. It errors out
  if the requested estimator's artifacts aren't in the file rather than
  silently searching uninitialized codes.
- `ffi/` and `python/` already have a directional load path
  (`LoadDirectionalGraph_*`), working since step 0b.

### Step 10 — telemetry — DONE

The phase-totals line now carries `rotate`, `train`, `populate` and `norms`
alongside the original phases, and the allocation step prints the host bytes the
payload arrays cost (and their per-node size) so an oversized run is visible at
the start instead of becoming an OOM in the middle.

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

### 4.3 Host memory, transient

- gather buffers: `2 * s * padded_dim * 2` for a shard of *s* nodes — one shard
  building, one prefetched. Scales with shard size and can't easily be chunked:
  `graph::reload` uploads a shard's vectors as one contiguous copy, which is the
  whole reason they're gathered into a buffer first.
- in-flight merge jobs: `(MERGE_QCAP + 2) * min(s, 2^20) * job_bytes_per_node`,
  with `job_bytes_per_node = 6*R_int + 1 + 4 + lsh_int + pq_int` and
  `MERGE_QCAP = 8`. **Independent of shard size** — see below.

| | job B/node | gather at s = 100M | merge jobs | total |
| --- | --- | --- | --- | --- |
| A (d96, R_int 32, LSH+PQ) | 517 | 35.8 GiB | 5.0 GiB | ~41 GiB |
| A (d96, R_int 32, PQ only) | 325 | 35.8 GiB | 3.2 GiB | ~39 GiB |

**A merge job is one graph segment, not one shard.** It didn't start that way,
and the difference decides whether large shards are affordable. With
shard-sized jobs the queue cost was `(MERGE_QCAP + 2) * s * ~500 B` — over
120 GiB at *s* = 100M — which forced `n_parts` up and shards down purely to
bound host memory, costing graph quality for nothing. Since `build_from_buf`
already copies back one `vectors_per_segment` (2²⁰) chunk at a time, and merge
rows are independent (a global id appears at most once per shard), each segment
is enqueued as its own job the moment its copy lands. The queue is then a few
GiB regardless of shard size, and `n_parts` can be chosen purely from device
capacity — which is what it should have been all along.

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

1. **`merge_row` payload semantics** — covered by `tests/test_diskann_merge.cu`
   (host-only): union/sort/dedup unchanged, payloads follow their edges through
   an in-place overwrite, dedup keeps the winning occurrence's codes, `R_FIN`
   truncation, `INVALID` slots, `out_count` monotonicity, and repeated merges.
   That file also explicitly instantiates
   `diskann_intermediate_graph<int_cfg, fin_cfg>` for a directional config,
   which is what type-checks the `if constexpr (USE_LSH)` half of the builder —
   nothing else in the tree instantiates one yet.
2. **`n_parts = 1, n_dup = 1`** on ~1M vectors: the merge is a no-op, so the
   build-time payloads must match a query-time `populate_edge_lsh` on the same
   index **bit-for-bit**. For LSH this holds for *any* `n_parts` (the codes
   depend only on the vector pair, with no shared model), and for PQ it holds
   when the reference is populated with `ig.dir.codebooks`. Needs the trailer
   writer (step 6) or an in-process reference graph to compare against;
   `dump_edge_payload()` is the stopgap.
3. **Recall**: `run_query --lsh` / `--pq` on the trailer-bearing index vs. the
   query-time-populated same index. Should match within noise; will not be
   identical, since codebooks are pilot-shard-trained rather than
   final-graph-trained.
4. **Trailer round-trip** through `load_directional_graph_from_file`; spot-check
   payload bytes, globals and centroids. The offset half of this is already
   pinned by `tests/test_graph_format.cu` (host-only: it checks
   `base_section_bytes` against every shipped layout, including the 1B x d96
   case where the offset exceeds 32 bits), and the saver asserts the same
   agreement at runtime.
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
