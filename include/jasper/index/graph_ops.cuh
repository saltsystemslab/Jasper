#pragma once

#include <cstdint>
#include <vector>
#include <algorithm>
#include <iostream>

#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>

#include "jasper/index/graph.cuh"
#include "jasper/index/construct.cuh"
#include "jasper/index/deletion.cuh"
#include "jasper/index/id_map.cuh"

namespace jasper {

// ─────────────────────────────────────────────────────────────────────────
// Host drivers for the deletion procedure, ported from JasperGPUANNS'
// bulk_gpuANN.cuh. Templated on CONSTRUCT_GRAPH_CONFIG (same config the FFI
// already instantiates as construct_cfg_##id) so consolidate() can reuse the
// reverse-edge robust-prune path from construct.cuh.
//
// These ops assume the graph is device-resident with global_offset == 0
// (the normal state for a built/loaded index).
// ─────────────────────────────────────────────────────────────────────────

template <typename CONSTRUCT_GRAPH_CONFIG>
__host__ typename CONSTRUCT_GRAPH_CONFIG::index_t
n_live(const typename CONSTRUCT_GRAPH_CONFIG::graph_t& g) {
  return g.n_vectors - g.n_deleted;
}

template <typename CONSTRUCT_GRAPH_CONFIG>
__host__ typename CONSTRUCT_GRAPH_CONFIG::index_t
n_tombstoned(const typename CONSTRUCT_GRAPH_CONFIG::graph_t& g) {
  return g.n_deleted;
}

// assign_ids() lives in id_map.cuh (it only needs the graph + index type) so
// that construct.cuh's append_batch can call it without a circular include.

// If the current medoid has been soft-deleted, replace it with a live vertex.
template <typename CONSTRUCT_GRAPH_CONFIG>
__host__ void update_medoid_if_deleted(typename CONSTRUCT_GRAPH_CONFIG::graph_t& g) {
  using graph_t = typename CONSTRUCT_GRAPH_CONFIG::graph_t;
  using index_t = typename CONSTRUCT_GRAPH_CONFIG::index_t;
  using segment_t = graph_segment<typename CONSTRUCT_GRAPH_CONFIG::graph_cfg_t>;

  if (g.n_vectors == 0) return;

  // Read medoid's deletion bit off the device.
  uint32_t seg_id = graph_t::segment_of(g.medoid);
  uint32_t loc    = graph_t::local_of(g.medoid);
  segment_t h_seg;
  cudaMemcpy(&h_seg, thrust::raw_pointer_cast(g.segments.data()) + seg_id,
             sizeof(segment_t), cudaMemcpyDeviceToHost);
  // Copy the 32-bit word holding medoid's deletion bit and test it.
  uint32_t medoid_word = 0;
  cudaMemcpy(&medoid_word, h_seg.deleted_bits + (loc >> 5), sizeof(uint32_t),
             cudaMemcpyDeviceToHost);
  bool medoid_deleted = (medoid_word >> (loc & 31u)) & 1u;
  if (!medoid_deleted) return;

  index_t* d_out;
  cudaMalloc(&d_out, sizeof(index_t));
  find_new_medoid_kernel<typename CONSTRUCT_GRAPH_CONFIG::graph_cfg_t>
      <<<1, 1>>>(g.view(), g.medoid, g.n_vectors, d_out);
  cudaDeviceSynchronize();

  index_t new_medoid;
  cudaMemcpy(&new_medoid, d_out, sizeof(index_t), cudaMemcpyDeviceToHost);
  cudaFree(d_out);

  if (new_medoid < g.n_vectors) {
    g.medoid = new_medoid;
  } else {
    std::cerr << "Warning: update_medoid_if_deleted found no live vertex\n";
  }
}

// Soft-delete a batch of vertex IDs (host array). Out-of-range IDs are
// filtered on the host before touching device memory.
// Soft-delete a batch of vectors given by STABLE ID. Ids are translated to
// internal slots through the forward table (misses are ignored), then the
// corresponding slots are tombstoned.
template <typename CONSTRUCT_GRAPH_CONFIG>
__host__ void mark_deleted(typename CONSTRUCT_GRAPH_CONFIG::graph_t& g,
                           const typename CONSTRUCT_GRAPH_CONFIG::index_t* ids_host,
                           typename CONSTRUCT_GRAPH_CONFIG::index_t n_ids) {
  using graph_cfg_t = typename CONSTRUCT_GRAPH_CONFIG::graph_cfg_t;
  using index_t     = typename CONSTRUCT_GRAPH_CONFIG::index_t;
  constexpr uint32_t BLOCK = CONSTRUCT_GRAPH_CONFIG::block_size;

  if (n_ids == 0) return;
  auto lk = g.lock_exclusive();  // exclusive: no concurrent search/mutation
#if JASPER_STABLE_IDS
  if (g.id_map == nullptr)
    throw std::runtime_error("mark_deleted: id_map not built");
#endif
  // With labels off, ids_to_slots_kernel is the identity (slot == id).

  // Upload ids, translate id -> slot (miss -> id_map_miss, skipped downstream).
  index_t* d_ids;
  cudaMalloc(&d_ids, sizeof(index_t) * n_ids);
  cudaMemcpy(d_ids, ids_host, sizeof(index_t) * n_ids, cudaMemcpyHostToDevice);

  index_t* d_slots;
  cudaMalloc(&d_slots, sizeof(index_t) * n_ids);
  ids_to_slots_kernel<index_t><<<(unsigned)n_ids, ID_MAP_TILE>>>(
      id_map_of<graph_cfg_t>(g), d_ids, d_slots, (uint32_t)n_ids);
  cudaDeviceSynchronize();

  index_t* d_newly;
  cudaMallocManaged(&d_newly, sizeof(index_t));
  *d_newly = 0;

  uint32_t grid = (n_ids + BLOCK - 1) / BLOCK;
  mark_deleted_batch_kernel<graph_cfg_t><<<grid, BLOCK>>>(
      g.view(), d_slots, n_ids, d_newly);
  cudaDeviceSynchronize();

  g.n_deleted += *d_newly;
  cudaFree(d_ids);
  cudaFree(d_slots);
  cudaFree(d_newly);

  update_medoid_if_deleted<CONSTRUCT_GRAPH_CONFIG>(g);
}

// Zero every segment's deletion bitmask (whole capacity).
template <typename CONSTRUCT_GRAPH_CONFIG>
__host__ void clear_deleted_flags(typename CONSTRUCT_GRAPH_CONFIG::graph_t& g) {
  using segment_t = graph_segment<typename CONSTRUCT_GRAPH_CONFIG::graph_cfg_t>;
  constexpr uint32_t VPS = CONSTRUCT_GRAPH_CONFIG::graph_t::vectors_per_segment;
  const size_t nbytes = segment_t::deleted_words(VPS) * sizeof(uint32_t);
  std::vector<segment_t> h_segs(g.segments.begin(), g.segments.end());
  for (auto& seg : h_segs) {
    if (seg.on_host) std::memset(seg.deleted_bits, 0, nbytes);
    else             cudaMemset(seg.deleted_bits, 0, nbytes);
  }
  cudaDeviceSynchronize();
}

// FreshDiskANN-style consolidation: repair edges that route through deleted
// vertices, drop deleted neighbors from live edge lists, then clear tombstones.
// Unlocked implementation: assumes the caller already holds g's write lock.
// compact() / maybe_consolidate() call this directly to avoid re-locking the
// non-recursive shared_mutex; external callers use consolidate() below.
template <typename CONSTRUCT_GRAPH_CONFIG>
__host__ void consolidate_impl(typename CONSTRUCT_GRAPH_CONFIG::graph_t& g,
                               float alpha = 1.2f) {
  using graph_cfg_t = typename CONSTRUCT_GRAPH_CONFIG::graph_cfg_t;
  using graph_t     = typename CONSTRUCT_GRAPH_CONFIG::graph_t;
  using index_t     = typename CONSTRUCT_GRAPH_CONFIG::index_t;
  constexpr uint32_t R     = CONSTRUCT_GRAPH_CONFIG::R;
  constexpr uint32_t BLOCK = CONSTRUCT_GRAPH_CONFIG::block_size;
  constexpr uint32_t TILE  = CONSTRUCT_GRAPH_CONFIG::tile_size;

  if (g.n_deleted == 0) return;
  if (g.global_offset != 0)
    throw std::runtime_error("consolidate requires global_offset == 0");
  if (g.on_host)
    throw std::runtime_error("consolidate requires graph on device");

  const index_t n = g.n_vectors;
  const uint32_t dim = g.dim;
  const uint32_t grid_all = (n + BLOCK - 1) / BLOCK;

  // Keep each chunk's repair-candidate count under thrust's 32-bit element
  // limit by splitting the source range.
  constexpr uint64_t SAFE_CHUNK_LIMIT = 1ULL << 30;  // ~1.07B

  uint64_t* d_total;
  cudaMallocManaged(&d_total, sizeof(uint64_t));
  *d_total = 0;
  count_repair_candidates_kernel<graph_cfg_t><<<grid_all, BLOCK>>>(
      g.view(), n, (index_t)0, n, d_total);
  cudaDeviceSynchronize();
  uint64_t n_repair_total = *d_total;
  cudaFree(d_total);

  uint32_t n_chunks = (uint32_t)((n_repair_total + SAFE_CHUNK_LIMIT - 1) / SAFE_CHUNK_LIMIT);
  if (n_chunks < 1) n_chunks = 1;
  index_t chunk_stride = (n + n_chunks - 1) / n_chunks;

  for (uint32_t ci = 0; ci < n_chunks; ci++) {
    index_t n_min = (index_t)ci * chunk_stride;
    index_t n_max = std::min<index_t>(n_min + chunk_stride, n);

    uint64_t* d_chunk;
    cudaMallocManaged(&d_chunk, sizeof(uint64_t));
    *d_chunk = 0;
    count_repair_candidates_kernel<graph_cfg_t><<<grid_all, BLOCK>>>(
        g.view(), n, n_min, n_max, d_chunk);
    cudaDeviceSynchronize();
    uint64_t n_repair = *d_chunk;
    cudaFree(d_chunk);

    if (n_repair == 0) {
      // No replacements available for this range, but still drop deleted
      // neighbors from these sources' edge lists.
      compact_deleted_neighbors_kernel<graph_cfg_t><<<grid_all, BLOCK>>>(
          g.view(), n, n_min, n_max);
      cudaDeviceSynchronize();
      continue;
    }

    thrust::device_vector<edge_pair<index_t>> repair(n_repair);
    edge_pair<index_t>* rp = thrust::raw_pointer_cast(repair.data());

    uint64_t* d_fill;
    cudaMallocManaged(&d_fill, sizeof(uint64_t));
    *d_fill = 0;
    fill_repair_candidates_kernel<graph_cfg_t><<<grid_all, BLOCK>>>(
        g.view(), n, n_min, n_max, rp, d_fill);
    cudaDeviceSynchronize();
    cudaFree(d_fill);

    // Drop deleted neighbors from this range's sources (existing edges that
    // process_reverse_edges_kernel will merge must be clean).
    compact_deleted_neighbors_kernel<graph_cfg_t><<<grid_all, BLOCK>>>(
        g.view(), n, n_min, n_max);
    cudaDeviceSynchronize();

    // Sort candidates by source and build per-source offsets, then merge +
    // robust-prune via the construction path.
    thrust::sort(repair.begin(), repair.end(),
                 semi_sort_compare_edge_pair<index_t>{});

    thrust::device_vector<index_t> offsets_flags(n_repair);
    thrust::device_vector<index_t> offsets(n_repair + 1);
    index_t num_unique = 0;
    get_unique_source_offsets<CONSTRUCT_GRAPH_CONFIG>(
        repair, (index_t)n_repair, offsets_flags, offsets, num_unique);

    if (num_unique > 0) {
      // One block per repair source (the kernel grid-strides if capped). This
      // matters at scale: with a high deletion fraction nearly every live
      // vertex is a repair source, so the construction-time grid of 1024 would
      // be far too serial.
      uint32_t pr_grid = (uint32_t)std::min<index_t>(num_unique, (index_t)2147483647u);
      process_reverse_edges_kernel<CONSTRUCT_GRAPH_CONFIG, TILE>
          <<<pr_grid, BLOCK>>>(
              rp, thrust::raw_pointer_cast(offsets.data()),
              g.view(), dim, num_unique, alpha);
      cudaDeviceSynchronize();
    }
  }

  // Zero out deleted nodes' edge lists, then clear all tombstones.
  clear_deleted_nodes_kernel<graph_cfg_t, R><<<grid_all, BLOCK>>>(g.view(), n);
  cudaDeviceSynchronize();

  clear_deleted_flags<CONSTRUCT_GRAPH_CONFIG>(g);
  g.n_deleted = 0;
}

// Repair edges around soft-deleted vertices and clear tombstones. Takes g's
// write lock (exclusive: no concurrent search or other mutation may overlap).
template <typename CONSTRUCT_GRAPH_CONFIG>
__host__ void consolidate(typename CONSTRUCT_GRAPH_CONFIG::graph_t& g,
                          float alpha = 1.2f) {
  auto lk = g.lock_exclusive();
  consolidate_impl<CONSTRUCT_GRAPH_CONFIG>(g, alpha);
}

// Physically reclaim space: reassign vertex IDs so live vertices occupy
// [0, n_live) contiguously. Consolidates first if there are pending deletions.
template <typename CONSTRUCT_GRAPH_CONFIG>
__host__ void compact(typename CONSTRUCT_GRAPH_CONFIG::graph_t& g,
                      typename CONSTRUCT_GRAPH_CONFIG::index_t extra_slack = 0) {
#if !JASPER_ENABLE_COMPACT
  (void)g; (void)extra_slack;
  // Compaction is compiled out of this build. Warn and no-op rather than
  // throwing, so callers (e.g. the Python FFI) keep running instead of
  // crashing; the deletion path stays mark_deleted + consolidate.
  std::cerr << "Warning: compact() is disabled in this build "
               "(mark_deleted + consolidate only); ignoring the call. "
               "Rebuild with -DJASPER_ENABLE_COMPACT=ON to enable.\n";
  return;
#else
  using graph_cfg_t = typename CONSTRUCT_GRAPH_CONFIG::graph_cfg_t;
  using graph_t     = typename CONSTRUCT_GRAPH_CONFIG::graph_t;
  using index_t     = typename CONSTRUCT_GRAPH_CONFIG::index_t;
  using segment_t   = graph_segment<graph_cfg_t>;
  constexpr uint32_t BLOCK = CONSTRUCT_GRAPH_CONFIG::block_size;
  constexpr uint32_t VPS   = graph_t::vectors_per_segment;

  auto lk = g.lock_exclusive();  // exclusive: no concurrent search/mutation
  if (g.on_host)
    throw std::runtime_error("compact requires graph on device");
  if (g.n_deleted > 0) consolidate_impl<CONSTRUCT_GRAPH_CONFIG>(g);

  (void)extra_slack;  // in-place compaction reclaims to exactly n_live
  const index_t n = g.n_vectors;
  if (n == 0) return;
  const uint32_t grid_all = (n + BLOCK - 1) / BLOCK;
  const uint32_t padded_dim = g.get_padded_dim();

  // 1. Mark live vertices; L = live count = final compacted size.
  thrust::device_vector<uint32_t> is_live(n);
  uint32_t* ilp = thrust::raw_pointer_cast(is_live.data());
  build_is_live_kernel<graph_cfg_t><<<grid_all, BLOCK>>>(g.view(), ilp, g.medoid, n);
  cudaDeviceSynchronize();
  index_t L = (index_t)thrust::reduce(thrust::device, is_live.begin(),
                                      is_live.end(), (uint64_t)0);
  if (L == n) return;  // dense already, nothing to reclaim

  // 2. mapping = identity; gather the holes below L (dead) and the movers
  //    above L (live). They are equinumerous (== count); pair them index-wise.
  thrust::device_vector<index_t> mapping(n);
  thrust::sequence(mapping.begin(), mapping.end());

  index_t n_dead = n - L;
  thrust::device_vector<index_t> holes(n_dead);
  thrust::device_vector<index_t> movers(n_dead);
  auto c0 = thrust::counting_iterator<index_t>(0);
  auto hend = thrust::copy_if(thrust::device, c0, c0 + L, holes.begin(),
      [ilp] __device__ (index_t i) { return ilp[i] == 0u; });
  auto cL = thrust::counting_iterator<index_t>(L);
  auto mend = thrust::copy_if(thrust::device, cL, cL + (n - L), movers.begin(),
      [ilp] __device__ (index_t i) { return ilp[i] != 0u; });
  index_t M = (index_t)(hend - holes.begin());   // == mend - movers.begin()
  (void)mend;

  // 3. Bulk-synchronous copy-down: move the M movers into the M holes, then
  //    rewrite all edges of the compacted [0, L) range through mapping.
  if (M > 0) {
    uint32_t grid_m = (uint32_t)((M + BLOCK - 1) / BLOCK);
    copy_down_kernel<graph_cfg_t><<<grid_m, BLOCK>>>(
        g.view(), thrust::raw_pointer_cast(movers.data()),
        thrust::raw_pointer_cast(holes.data()), M, padded_dim,
        thrust::raw_pointer_cast(mapping.data()));
    cudaDeviceSynchronize();
  }
  uint32_t grid_l = (uint32_t)((L + BLOCK - 1) / BLOCK);
  remap_compacted_edges_kernel<graph_cfg_t><<<grid_l, BLOCK>>>(
      g.view(), thrust::raw_pointer_cast(mapping.data()), L);
  cudaDeviceSynchronize();

  // 4. Remap the medoid.
  index_t new_medoid = 0;
  cudaMemcpy(&new_medoid, thrust::raw_pointer_cast(mapping.data()) + g.medoid,
             sizeof(index_t), cudaMemcpyDeviceToHost);
  g.medoid = new_medoid;

  // 5. Truncate: shrink per-segment counts and free now-empty trailing segments.
  index_t old_n = g.n_vectors;
  uint32_t new_n_segments = (uint32_t)((L + VPS - 1) / VPS);
  {
    std::vector<segment_t> h(g.segments.begin(), g.segments.end());
    for (uint32_t s = new_n_segments; s < g.n_segments; s++) h[s].deallocate();
    h.resize(new_n_segments);
    for (uint32_t s = 0; s < new_n_segments; s++) {
      index_t start = (index_t)s * VPS;
      h[s].n_vectors = std::min<index_t>(VPS, L - start);
    }
    g.segments = thrust::device_vector<segment_t>(h.begin(), h.end());
  }
  g.n_segments = new_n_segments;
  g.n_vectors  = L;
  g.n_deleted  = 0;

  // Slots were renumbered (slot_to_id moved with each live vertex); rebuild the
  // forward table so stable_id -> slot reflects the compacted layout. Stable
  // ids themselves are unchanged.
  rebuild_id_map<graph_cfg_t>(g);

  std::cerr << "compact: " << (uint64_t)old_n << " -> " << (uint64_t)L
            << " vertices (" << (uint64_t)(old_n - L)
            << " tombstones reclaimed, " << (uint64_t)M << " moved in place)\n";
#endif  // JASPER_ENABLE_COMPACT
}

// Consolidate automatically if the tombstone ratio exceeds the threshold.
template <typename CONSTRUCT_GRAPH_CONFIG>
__host__ bool maybe_consolidate(typename CONSTRUCT_GRAPH_CONFIG::graph_t& g,
                                float alpha = 1.2f) {
  using index_t = typename CONSTRUCT_GRAPH_CONFIG::index_t;
  auto lk = g.lock_exclusive();  // exclusive; also guards the ratio read below
  index_t live = g.n_vectors - g.n_deleted;
  if (live == 0) return false;
  double ratio = (double)g.n_deleted / (double)live;
  if (ratio > g.consolidation_threshold) {
    consolidate_impl<CONSTRUCT_GRAPH_CONFIG>(g, alpha);
    return true;
  }
  return false;
}

}  // namespace jasper
