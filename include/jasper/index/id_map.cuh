#pragma once

#include <cstdint>
#include <vector>
#include <cooperative_groups.h>

#include "jasper/index/graph.cuh"   // defines JASPER_STABLE_IDS + device_view

#if JASPER_STABLE_IDS
// Header-ordering fix for WarpSpeed: double_hashing.cuh -> ht_bucket.cuh calls
// typed_atomic_write *unqualified* with no using-directive in scope. Pull in the
// gallatin headers in order (alloc_utils defines ldca etc. used by ds_utils),
// then a using-declaration so the unqualified call in ht_bucket resolves.
#include <gallatin/allocators/alloc_utils.cuh>
#include <gallatin/data_structs/ds_utils.cuh>
using gallatin::utils::typed_atomic_write;
#include <warpSpeed/tables/double_hashing.cuh>
#endif

namespace jasper {

namespace cg = cooperative_groups;

static constexpr uint32_t ID_MAP_TILE = 8;

// Reserved value written by find kernels when a key is absent.
template <typename INDEX_T>
__host__ __device__ constexpr INDEX_T id_map_miss() {
  return static_cast<INDEX_T>(~static_cast<INDEX_T>(0));
}

// key = stable_id + 1 (avoids WarpSpeed's reserved 0/all-ones sentinels).
template <typename INDEX_T>
__host__ __device__ __forceinline__ INDEX_T id_to_key(INDEX_T id) {
  return id + static_cast<INDEX_T>(1);
}

template <typename INDEX_T>
__host__ uint64_t id_map_capacity_for(INDEX_T n) {
  return static_cast<uint64_t>(n) * 3 / 2 + 1024;
}

// ── View-only kernels (no forward table; valid whether or not labels are on) ──

// Rewrite search-result pairs from internal slot to stable id. With labels off,
// slot_to_id is identity, so this is a no-op transform (slot -> slot). Padding
// slots (out of range) are left untouched. One thread per pair.
template <typename GRAPH_CFG, typename PAIR_T>
__global__ void translate_slots_to_ids_kernel(typename graph<GRAPH_CFG>::device_view g,
                                              PAIR_T* pairs, uint32_t n) {
  using index_t = typename GRAPH_CFG::index_t;
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  index_t slot = static_cast<index_t>(pairs[i].first);
  if (g.is_valid(slot)) pairs[i].first = g.get_stable_id(slot);
}

// slot_to_id[s] = base + s for s in [0, count) (identity at construction).
template <typename GRAPH_CFG>
__global__ void assign_sequential_ids_kernel(typename graph<GRAPH_CFG>::device_view g,
                                            typename GRAPH_CFG::index_t base,
                                            typename GRAPH_CFG::index_t count) {
  using index_t = typename GRAPH_CFG::index_t;
  index_t s = (index_t)(blockIdx.x * blockDim.x + threadIdx.x);
  if (s >= count) return;
  g.set_stable_id(s, base + s);
}

// slot_to_id[slot_start + i] = id_start + i for i in [0, count) (append).
template <typename GRAPH_CFG>
__global__ void set_id_range_kernel(typename graph<GRAPH_CFG>::device_view g,
                                    typename GRAPH_CFG::index_t slot_start,
                                    typename GRAPH_CFG::index_t id_start,
                                    typename GRAPH_CFG::index_t count) {
  using index_t = typename GRAPH_CFG::index_t;
  index_t i = (index_t)(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= count) return;
  g.set_stable_id(slot_start + i, id_start + i);
}

// Reserve `count` fresh stable ids: advance the monotonic counter and return
// [start, start+count). (Independent of the forward table.)
template <typename GRAPH_CFG>
__host__ std::vector<typename GRAPH_CFG::index_t>
assign_ids(graph<GRAPH_CFG>& g, typename GRAPH_CFG::index_t count) {
  using index_t = typename GRAPH_CFG::index_t;
  index_t start = g.next_id;
  g.next_id += count;
  std::vector<index_t> ids(count);
  for (index_t i = 0; i < count; i++) ids[i] = start + i;
  return ids;
}

#if JASPER_STABLE_IDS
// ═══════════════════ Forward table (stable_id -> slot), WarpSpeed ═══════════
template <typename INDEX_T>
using id_map_t = warpSpeed::tables::double_generic<INDEX_T, INDEX_T,
                                                   /*tile_size=*/8, /*bucket_size=*/8>;

template <typename INDEX_T>
__host__ id_map_t<INDEX_T>* id_map_create(uint64_t capacity, uint64_t seed = 42) {
  return id_map_t<INDEX_T>::generate_on_device(capacity, seed);
}
template <typename INDEX_T>
__host__ void id_map_destroy(id_map_t<INDEX_T>* table) {
  if (table) id_map_t<INDEX_T>::free_on_device(table);
}

template <typename GRAPH_CFG>
__global__ void build_id_map_kernel(typename graph<GRAPH_CFG>::device_view g,
                                    id_map_t<typename GRAPH_CFG::index_t>* table,
                                    typename GRAPH_CFG::index_t count) {
  using index_t = typename GRAPH_CFG::index_t;
  auto tile = cg::tiled_partition<ID_MAP_TILE>(cg::this_thread_block());
  index_t s = (index_t)blockIdx.x;
  if (s >= count) return;
  index_t id = g.get_stable_id(s);
  if (id == id_map_miss<index_t>()) return;
  table->upsert_replace(tile, id_to_key(id), s);
}

template <typename GRAPH_CFG>
__global__ void upsert_slot_range_kernel(typename graph<GRAPH_CFG>::device_view g,
                                         id_map_t<typename GRAPH_CFG::index_t>* table,
                                         typename GRAPH_CFG::index_t slot_start,
                                         typename GRAPH_CFG::index_t count) {
  using index_t = typename GRAPH_CFG::index_t;
  auto tile = cg::tiled_partition<ID_MAP_TILE>(cg::this_thread_block());
  index_t i = (index_t)blockIdx.x;
  if (i >= count) return;
  index_t s = slot_start + i, id = g.get_stable_id(s);
  if (id == id_map_miss<index_t>()) return;
  table->upsert_replace(tile, id_to_key(id), s);
}

template <typename INDEX_T>
__global__ void ids_to_slots_kernel(id_map_t<INDEX_T>* table, const INDEX_T* ids,
                                    INDEX_T* out_slots, uint32_t n) {
  auto tile = cg::tiled_partition<ID_MAP_TILE>(cg::this_thread_block());
  uint32_t i = blockIdx.x;
  if (i >= n) return;
  INDEX_T slot = 0;
  bool ok = table->find_with_reference(tile, id_to_key(ids[i]), slot);
  if (tile.thread_rank() == 0) out_slots[i] = ok ? slot : id_map_miss<INDEX_T>();
}

template <typename GRAPH_CFG>
__global__ void register_ids_kernel(typename graph<GRAPH_CFG>::device_view g,
                                    id_map_t<typename GRAPH_CFG::index_t>* table,
                                    const typename GRAPH_CFG::index_t* slots,
                                    const typename GRAPH_CFG::index_t* ids,
                                    typename GRAPH_CFG::index_t n) {
  using index_t = typename GRAPH_CFG::index_t;
  auto tile = cg::tiled_partition<ID_MAP_TILE>(cg::this_thread_block());
  index_t i = (index_t)blockIdx.x;
  if (i >= n) return;
  index_t slot = slots[i], id = ids[i];
  if (tile.thread_rank() == 0) g.set_stable_id(slot, id);
  table->upsert_replace(tile, id_to_key(id), slot);
}

template <typename GRAPH_CFG>
__host__ void build_id_map(graph<GRAPH_CFG>& g, uint64_t capacity = 0) {
  using index_t = typename GRAPH_CFG::index_t;
  if (capacity == 0) capacity = id_map_capacity_for<index_t>(g.n_vectors);
  auto* table = id_map_create<index_t>(capacity);
  g.id_map = static_cast<void*>(table);
  g.id_map_capacity = capacity;
  if (g.n_vectors > 0) {
    build_id_map_kernel<GRAPH_CFG><<<(unsigned)g.n_vectors, ID_MAP_TILE>>>(
        g.view(), table, g.n_vectors);
    cudaDeviceSynchronize();
  }
}
template <typename GRAPH_CFG>
__host__ void rebuild_id_map(graph<GRAPH_CFG>& g, uint64_t capacity = 0) {
  using index_t = typename GRAPH_CFG::index_t;
  if (g.id_map) {
    id_map_destroy<index_t>(static_cast<id_map_t<index_t>*>(g.id_map));
    g.id_map = nullptr;
  }
  build_id_map<GRAPH_CFG>(g, capacity);
}
template <typename GRAPH_CFG>
__host__ id_map_t<typename GRAPH_CFG::index_t>* id_map_of(graph<GRAPH_CFG>& g) {
  return static_cast<id_map_t<typename GRAPH_CFG::index_t>*>(g.id_map);
}

static constexpr double ID_MAP_MAX_LOAD = 0.90;  // resize before the table fills

template <typename GRAPH_CFG>
__host__ bool id_map_reserve(graph<GRAPH_CFG>& g, uint64_t min_entries) {
  uint64_t max_entries = (uint64_t)((double)g.id_map_capacity * ID_MAP_MAX_LOAD);
  if (g.id_map != nullptr && min_entries <= max_entries) return false;
  using index_t = typename GRAPH_CFG::index_t;
  rebuild_id_map<GRAPH_CFG>(g, id_map_capacity_for<index_t>((index_t)min_entries));
  return true;
}
template <typename GRAPH_CFG>
__host__ void id_map_add_range(graph<GRAPH_CFG>& g,
                               typename GRAPH_CFG::index_t slot_start,
                               typename GRAPH_CFG::index_t count) {
  if (count == 0) return;
  if (id_map_reserve<GRAPH_CFG>(g, (uint64_t)g.n_vectors)) return;  // resize reinserted all
  upsert_slot_range_kernel<GRAPH_CFG><<<(unsigned)count, ID_MAP_TILE>>>(
      g.view(), id_map_of<GRAPH_CFG>(g), slot_start, count);
  cudaDeviceSynchronize();
}
template <typename GRAPH_CFG>
__host__ void register_ids(graph<GRAPH_CFG>& g,
                           const typename GRAPH_CFG::index_t* d_slots,
                           const typename GRAPH_CFG::index_t* d_ids,
                           typename GRAPH_CFG::index_t n) {
  if (n == 0) return;
  register_ids_kernel<GRAPH_CFG><<<(unsigned)n, ID_MAP_TILE>>>(
      g.view(), id_map_of<GRAPH_CFG>(g), d_slots, d_ids, n);
  cudaDeviceSynchronize();
}

#else  // ═══════════════ Labels OFF: identity slot==id, no table ═════════════

template <typename INDEX_T> struct id_map_dummy {};
template <typename INDEX_T> using id_map_t = id_map_dummy<INDEX_T>;

template <typename INDEX_T>
__host__ id_map_t<INDEX_T>* id_map_create(uint64_t, uint64_t = 42) { return nullptr; }
template <typename INDEX_T>
__host__ void id_map_destroy(id_map_t<INDEX_T>*) {}

// Identity translate: slot == stable id, so out_slots[i] = ids[i].
template <typename INDEX_T>
__global__ void ids_to_slots_kernel(id_map_t<INDEX_T>*, const INDEX_T* ids,
                                    INDEX_T* out_slots, uint32_t n) {
  uint32_t i = blockIdx.x;
  if (i >= n) return;
  if (threadIdx.x == 0) out_slots[i] = ids[i];
}

template <typename GRAPH_CFG>
__host__ void build_id_map(graph<GRAPH_CFG>&, uint64_t = 0) {}
template <typename GRAPH_CFG>
__host__ void rebuild_id_map(graph<GRAPH_CFG>&, uint64_t = 0) {}
template <typename GRAPH_CFG>
__host__ id_map_t<typename GRAPH_CFG::index_t>* id_map_of(graph<GRAPH_CFG>&) { return nullptr; }
template <typename GRAPH_CFG>
__host__ bool id_map_reserve(graph<GRAPH_CFG>&, uint64_t) { return false; }
template <typename GRAPH_CFG>
__host__ void id_map_add_range(graph<GRAPH_CFG>&, typename GRAPH_CFG::index_t,
                               typename GRAPH_CFG::index_t) {}
template <typename GRAPH_CFG>
__host__ void register_ids(graph<GRAPH_CFG>&, const typename GRAPH_CFG::index_t*,
                           const typename GRAPH_CFG::index_t*, typename GRAPH_CFG::index_t) {}

#endif  // JASPER_STABLE_IDS

// Construction finalizer: assign identity ids (slot i -> id i), set next_id, and
// (when labels are on) build the forward table. When labels are off, build_id_map
// is a no-op and slot_to_id stays identity so translate is a slot->slot no-op.
template <typename GRAPH_CFG>
__host__ void assign_identity_ids_and_build(graph<GRAPH_CFG>& g) {
  using index_t   = typename GRAPH_CFG::index_t;
  using segment_t = graph_segment<GRAPH_CFG>;
  constexpr uint32_t VPS = graph<GRAPH_CFG>::vectors_per_segment;
  g.next_id = g.n_vectors;

  if (g.on_host) {
    std::vector<segment_t> h(g.segments.begin(), g.segments.end());
    for (uint32_t s = 0; s < g.n_segments; s++) {
      index_t base = (index_t)s * VPS;
      for (uint32_t i = 0; i < (uint32_t)h[s].n_vectors; i++)
        h[s].slot_to_id[i] = base + i;
    }
    return;
  }
  if (g.n_vectors > 0) {
    uint32_t BLK = 256, grid = (uint32_t)((g.n_vectors + BLK - 1) / BLK);
    assign_sequential_ids_kernel<GRAPH_CFG><<<grid, BLK>>>(g.view(), (index_t)0, g.n_vectors);
    cudaDeviceSynchronize();
  }
  build_id_map<GRAPH_CFG>(g);
}

}  // namespace jasper
