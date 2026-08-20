# Deletion with compaction: stable ids over a GPU hashtable

Status: **as-built + promotion proposal**. The mechanism this doc describes
already exists in the tree, gated behind two off-by-default CMake options.
This doc (a) specifies that existing design precisely enough to review it,
and (b) proposes what's needed to turn it on by default, since "deletion just
leaves holes" is the behavior every build ships today.

## 1. Problem

`consolidate()` (always available) repairs graph edges around deleted
vertices and clears their edge lists, but never reclaims their storage slot.
Long-running indexes that see steady delete+insert traffic accumulate dead
slots forever: `n_vectors` (and therefore segment count, VRAM, and the cost of
any full scan) grows monotonically even though `n_live` does not.

`compact()` — physically repacking live vectors into `[0, n_live)` — already
exists (`include/jasper/index/graph_ops.cuh:263-362`) but is compiled out by
default:

```cpp
// graph_ops.cuh:264-271
__host__ void compact(typename CONSTRUCT_GRAPH_CONFIG::graph_t& g,
                      typename CONSTRUCT_GRAPH_CONFIG::index_t extra_slack = 0) {
#if !JASPER_ENABLE_COMPACT
  throw std::runtime_error(
      "compact() is disabled in this build (deletion path is mark_deleted + "
      "consolidate only). Rebuild with -DJASPER_ENABLE_COMPACT=ON to enable.");
```

Compaction renumbers slots (a live vector at slot 900 may end up at slot 12
to fill a hole). Callers address vectors by external, caller-assigned id
(the id returned by `search`/`append`), so compaction is only safe to expose
once slot renumbering is invisible to callers — i.e. once there's a stable id
layer between "the id the caller uses" and "the slot the vector currently
occupies." That layer is `JASPER_STABLE_IDS`, also off by default.

## 2. Existing design

### 2.1 Storage layout

The index is a Vamana/DiskANN-style graph, stored as structure-of-arrays,
segmented in chunks of `vectors_per_segment = 1<<20`
(`include/jasper/index/graph.cuh:84`). Each `graph_segment` holds, per slot:

```cpp
// graph.cuh:92-235 (fields relevant to deletion, abbreviated)
edge_list_t   *edges;         // fixed R neighbors + bf16 dists
uint8_t       *edge_counts;
vector_view_t  vectors;       // padded_dim-aligned raw vector rows
uint32_t      *deleted_bits;  // 1 bit/slot soft-delete flag, packed 32/word
index_t       *slot_to_id;    // slot -> external stable id (reverse map)
```

A **slot** is the position in these arrays (`0 .. n_vectors)`, contiguous and
dense modulo tombstones. A **stable id** is the number the caller sees from
`search()`/`append()`/`mark_deleted()`. With both feature flags off, slot ==
stable id by construction and this distinction is invisible.

### 2.2 The two id maps

- **Reverse map, slot → id**: `slot_to_id[]`, one entry per slot, always
  allocated regardless of flags (`graph.cuh:125`). Written at construction
  (`assign_identity_ids_and_build`, `id_map.cuh:260-281`) and at append
  (`set_id_range_kernel`, `id_map.cuh:71-79`). Trivial array indexing, no
  hashtable needed — reverse lookups are always O(1) regardless of flags.

- **Forward map, id → slot**: only needed because ids and slots can diverge
  (after compaction, or with caller-supplied non-sequential ids). This is the
  hashtable referenced in the task: `include/jasper/index/id_map.cuh:94-198`,
  gated by `JASPER_STABLE_IDS`.

  ```cpp
  // id_map.cuh:97-98
  template <typename INDEX_T>
  using id_map_t = warpSpeed::tables::double_generic<INDEX_T, INDEX_T,
                                                     /*tile_size=*/8, /*bucket_size=*/8>;
  ```

  Backed by [WarpSpeed](https://github.com/saltsystemslab/warpSpeed)'s
  GPU double-hashing table (itself built on the
  [gallatin](https://github.com/saltsystemslab/gallatin) GPU allocator), both
  fetched via CMake `FetchContent` only when the flag is on
  (`CMakeLists.txt:47-50`) — zero cost, not even compiled, when off.

  - Key = `stable_id + 1` (`id_to_key`, `id_map.cuh:34-36`) to dodge
    WarpSpeed's reserved sentinel keys (0 and all-ones).
  - Capacity: `n*3/2 + 1024` (`id_map_capacity_for`, `id_map.cuh:39-41`);
    resized (full rebuild) once load exceeds `ID_MAP_MAX_LOAD = 0.90`
    (`id_map.cuh:189-198`).
  - Lookup is `ids_to_slots_kernel` (`id_map.cuh:137-145`): one cooperative-group
    tile per id, `table->find_with_reference`.
  - Insert is incremental (`upsert_slot_range_kernel` / `register_ids_kernel`)
    on append, and a full `build_id_map_kernel` sweep on `rebuild_id_map`
    (used after compaction, since slots moved).

  When `JASPER_STABLE_IDS=0`, `ids_to_slots_kernel` is replaced with an
  identity function (`id_map.cuh:230-237`) and every id-map host function is a
  no-op — this is what makes the flag genuinely free when off, not just
  disabled.

- **Id allocation**: `graph.next_id` is a monotonic counter
  (`graph.cuh:538`). `assign_ids()` (`id_map.cuh:83-92`) hands out
  `[next_id, next_id+count)` and never reuses an id, including ids whose
  vectors have since been deleted and compacted away. This is what lets
  compaction move a vector's slot without invalidating any id a caller might
  be holding.

### 2.3 Deletion pipeline (three stages)

**Stage 1 — soft delete** (`mark_deleted`, `graph_ops.cuh:93-134`, always on).
Translates caller ids → slots via `ids_to_slots_kernel` (identity if labels
are off), then `mark_deleted_batch_kernel`
(`include/jasper/index/deletion.cuh:22-43`) sets the slot's bit in
`deleted_bits` with `atomicOr` (race-safe against duplicate ids in the same
batch — the old word is inspected to count each id as newly-deleted exactly
once). Search filters on this bit immediately; O(batch size), no graph
structure touched. If the medoid was deleted, `update_medoid_if_deleted`
(`graph_ops.cuh:50-86`) walks the medoid's neighbor list, falling back to
`find_new_medoid_kernel`.

**Stage 2 — consolidate** (`consolidate`, `graph_ops.cuh:152-259`, always
on, auto-triggered by `maybe_consolidate` at `n_deleted/n_live >
consolidation_threshold`, default `0.1`). FreshDiskANN-style edge repair:
for every live vertex `N` with a deleted neighbor `D`, every live neighbor
`M` of `D` becomes a repair-candidate edge `N -> M`
(`count_repair_candidates_kernel` / `fill_repair_candidates_kernel`,
`deletion.cuh:49-112`), robust-pruned back into `N`'s edge list via the same
`process_reverse_edges_kernel` used at construction time. Deleted neighbors
are then dropped from live edge lists in place
(`compact_deleted_neighbors_kernel`, `deletion.cuh:118-148`), dead vertices'
own edges are zeroed (`clear_deleted_nodes_kernel`), and all tombstone bits
are cleared. **After this stage, `n_deleted == 0` and the graph is fully
correct — but dead slots are still allocated, holding stale vectors and
occupying VRAM.** This is the "holes" the task refers to. Chunked by source
range (`SAFE_CHUNK_LIMIT = 1<<30`) so per-chunk candidate counts stay under
`thrust`'s 32-bit element ceiling at scale.

**Stage 3 — compact** (`compact`, `graph_ops.cuh:263-362`, gated,
consolidates first if needed):

1. `build_is_live_kernel`: live = `edge_count > 0 || slot == medoid`. Reduce
   to get `L` = live count = final size.
2. Gather `holes` (dead slots `< L`) and `movers` (live slots `>= L`) via
   `thrust::copy_if`; they're equinumerous by construction, paired
   index-wise.
3. `copy_down_kernel` (`deletion.cuh:190-217`): bulk-synchronous, one thread
   per (mover, hole) pair, copies edges + vector + stable id from mover to
   hole. **Only the `M ≤ n_deleted` movers actually move** — everything
   already below `L` stays put, so the result is not order-preserving, but
   there is no second full-graph copy. Source (`≥ L`) and destination
   (`< L`) ranges never alias.
4. `remap_compacted_edges_kernel`: rewrite every edge among the compacted
   `[0, L)` range from old slot to new via a `mapping` array (identity for
   stay-put slots, hole-index for movers). Safe because consolidate already
   guarantees no live vertex points at a dead one.
5. Remap the medoid through the same `mapping`; truncate/free now-empty
   trailing segments; set `n_vectors = L`, `n_deleted = 0`.
6. `rebuild_id_map`: slots moved, so the forward table must be rebuilt from
   the (already-updated) `slot_to_id` reverse map. **Stable ids themselves
   never change** — only which slot they resolve to.

### 2.4 Interaction with live append

`append_batch` (`include/jasper/index/construct.cuh`, `global_offset == 0`
precondition, same as `consolidate`/`compact`) places new vectors at
`[old_n, new_n)`, wires their edges via the same construction machinery, and
extends both id maps (`set_id_range_kernel` for the reverse map,
`id_map_add_range` → `upsert_slot_range_kernel` for the forward table,
resizing at 90% load). This is the reason compaction must run **before**
resuming inserts, not concurrently with them: it assumes it owns the whole
`[0, n)` range and picks new slots for movers without coordinating with a
concurrent appender that's also claiming `[old_n, new_n)`.

### 2.5 Persistence

Save/load already carries deletion state and the reverse map to disk
(`graph.cuh` binary format: `slot_to_id` inline per node, deletion as a
packed bitmask, `next_id` as a trailing `u64`). The forward hashtable is
**not** serialized — `rebuild_id_map(g)` reconstructs it after load
(`graph.cuh:1646-1647`), which is correct but means load time for a
label-enabled graph includes an `O(n_vectors)` table rebuild.

### 2.6 Public surface (already wired end-to-end)

FFI (`ffi/jasper_ffi_common.cuh:385-425,793-795`) and Python
(`python/jasper/__init__.py:645-739`):

```python
ids = g.append(new_vectors, alpha=1.2)   # -> int32 tensor of assigned stable ids
g.mark_deleted(ids)                       # soft-delete, immediate
g.consolidate(alpha=1.2)                  # repair edges, clear tombstones
g.compact()                               # reclaim space, renumber slots
g.n_live                                  # n_vectors - n_tombstoned
g.n_tombstoned                            # pending soft-deletes
ids = g.reserve_ids(count)                # pre-allocate an id range
```

Directional (LSH/PQ proxy-distance) graph configs explicitly do not expose
`mark_deleted`/`consolidate`/`compact` (`python/jasper/__init__.py:226-231`)
— out of scope for this doc; they'd need their own review.

## 3. Why this was left off by default

From the commit that introduced it (`15afd76`, "Add deletion + stable-id +
live-append (gated off by default)"): the code is "ported/adapted from
JasperGPUANNS" (a predecessor project) and was merged in a deliberately
"stable temporary state" — validated on bigann10M/100M and GIST, with
dedicated test/bench harnesses (`cmd/test_deletion.cu`, `cmd/test_idmap.cu`,
`cmd/test_append.cu`) and cluster sbatch scripts, but never promoted to
default-on. No follow-up commit explains a specific correctness objection;
the flags read as "works, not yet trusted as default."

## 4. Known gaps to close before promoting to default-on

- **Id-map resize is a full rebuild.** `id_map_reserve` (`id_map.cuh:191-198`)
  does not grow the WarpSpeed table in place — it destroys it and calls
  `build_id_map`, which re-inserts every live id from `slot_to_id` from
  scratch (`build_id_map_kernel`, one block per slot). Fine for occasional
  `compact()` calls; on the `append_batch` path this means any append that
  crosses the 90%-load threshold pays an `O(n_vectors)` rebuild inline. Needs
  either a growth-friendly table or amortized capacity headroom sized to the
  workload's expected growth between compactions.
- **`compact()` and `append_batch()` both assume exclusive ownership of
  `[0, n)` and `global_offset == 0`.** There's no lock or epoch mechanism
  preventing a caller from interleaving `append` and `compact` calls across
  threads; today this is a documentation-only contract. Needs either an
  explicit "compaction owns the graph" mutex at the `graph<>` level or a
  documented single-writer requirement pushed to callers.
- **Medoid replacement is a worst-case O(n) linear scan.**
  `find_new_medoid_kernel` (`deletion.cuh:242-266`) only checks the old
  medoid's direct neighbors before falling back to scanning every vertex,
  single-threaded (`blockIdx.x != 0 || threadIdx.x != 0) return;`). Rare (only
  triggers when the medoid itself is deleted) but worth a second-hop fallback
  before the full scan, since the full scan runs entirely on one thread.
  Relevant at 100M+ scale where every `mark_deleted` call checks this.
  (`graph_ops.cuh:50-86`, `update_medoid_if_deleted`.)
- **Forward table isn't persisted.** Every `load()` of a stable-id-enabled
  graph pays a full `rebuild_id_map` sweep. Acceptable if load is rare
  relative to queries, but should be stated as a documented cost, not
  discovered by whoever benchmarks cold-start latency next.
- **No concurrent-with-search guarantee is documented.** `consolidate` and
  `compact` mutate edge lists and vector storage in place while presumably no
  concurrent `search` is specified as running — this should be stated
  explicitly (e.g. "callers must quiesce search traffic during
  consolidate/compact") rather than left implicit in "the driver guarantees
  global_offset == 0."
- **`extra_slack` parameter on `compact()` is unused.** Signature accepts it
  but the body does `(void)extra_slack; // in-place compaction reclaims to
  exactly n_live` (`graph_ops.cuh:283`). Either wire it up (pre-allocate
  headroom for near-term appends without re-segmenting) or drop the
  parameter — right now it's a promise the API makes and breaks silently.

## 5. Promotion plan

1. **Turn `JASPER_STABLE_IDS` and `JASPER_ENABLE_COMPACT` on in one CI
   config first**, running the existing `cmd/test_deletion.cu`,
   `cmd/test_idmap.cu`, `cmd/test_append.cu`, plus the `sbatch/test_deletion_*`
   and `sbatch/verify_stable.sbatch` cluster jobs, before touching the
   default.
2. **Close the id-map resize gap (§4)** — this is the one item with a
   plausible performance cliff at production insert rates; the rest are
   correctness/robustness hardening that can land incrementally without
   blocking the flip.
3. **Document the single-writer contract** for consolidate/compact/append
   (§4) in the README's deletion section, since it's currently only implicit
   in the `global_offset == 0` checks.
4. **Flip both flags to `ON` by default** in `CMakeLists.txt:43-44` once (1)
   passes clean and (2)/(3) are addressed. Keep the flags present (not
   removed) for one release so anyone pinned to the old on-disk id==slot
   identity behavior can opt back out.
5. **Add a compaction-scheduling knob** analogous to
   `consolidation_threshold` (e.g. auto-`compact()` when reclaimable slots
   exceed some fraction of `n_vectors`), so callers get the "no holes"
   default behavior without having to remember to call `compact()`
   themselves — mirroring how `maybe_consolidate` already works today.
