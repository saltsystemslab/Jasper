#pragma once

#include <cstdint>
#include <limits>

#include "jasper/index/graph.cuh"
#include "jasper/index/construct.cuh"  // for edge_pair<INDEX_T>

namespace jasper {

// ─────────────────────────────────────────────────────────────────────────
// Deletion kernels — ported from JasperGPUANNS/deletion_kernels.cuh and
// rewritten to operate through graph<GRAPH_CFG>::device_view accessors instead
// of flat arrays. All indices are global; the driver guarantees global_offset
// == 0 for the graphs these kernels run on, so global index == local index.
// ─────────────────────────────────────────────────────────────────────────

// Mark a batch of vertex IDs as soft-deleted. Deletion state is a packed
// bitmask (1 bit/slot); we atomicOr the target bit into its 32-bit word and
// inspect the old word to count entries newly marked (bit previously 0). This
// makes duplicate IDs in the batch race-safe and counted once.
template <typename GRAPH_CFG>
__global__ void mark_deleted_batch_kernel(
    typename graph<GRAPH_CFG>::device_view g,
    const typename GRAPH_CFG::index_t* ids,
    typename GRAPH_CFG::index_t n_ids,
    typename GRAPH_CFG::index_t* out_new_count) {
  using index_t = typename GRAPH_CFG::index_t;
  uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n_ids) return;
  index_t id = ids[tid];
  if (id >= g.n_vectors) return;  // MISS sentinel / out-of-range slot

  index_t local_idx = g.to_local(id);
  uint32_t seg_id   = graph<GRAPH_CFG>::segment_of(local_idx);
  uint32_t loc      = graph<GRAPH_CFG>::local_of(local_idx);
  uint32_t* bits = g.segments[seg_id].deleted_bits;
  uint32_t  mask = 1u << (loc & 31u);
  uint32_t  old_word = atomicOr(&bits[loc >> 5], mask);
  if ((old_word & mask) == 0) {   // bit was unset -> this ID is newly deleted
    atomicAdd(out_new_count, (index_t)1);
  }
}

// First pass: count repair candidates. For each non-deleted vertex N with a
// deleted neighbor D, each non-deleted neighbor M of D (M != N) becomes a
// repair candidate edge N->M.  n_min/n_max restrict the source range N for
// chunked processing.
template <typename GRAPH_CFG>
__global__ void count_repair_candidates_kernel(
    typename graph<GRAPH_CFG>::device_view graph,
    typename GRAPH_CFG::index_t n_vertices,
    typename GRAPH_CFG::index_t n_min,
    typename GRAPH_CFG::index_t n_max,
    uint64_t* out_total) {
  using index_t = typename GRAPH_CFG::index_t;
  constexpr index_t INVALID = std::numeric_limits<index_t>::max();

  uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n_vertices) return;
  index_t N = (index_t)tid;
  if (N < n_min || N >= n_max) return;
  if (graph.is_deleted(N)) return;

  uint8_t deg_N = graph.get_edge_count(N);
  uint64_t local_count = 0;
  for (uint8_t j = 0; j < deg_N; j++) {
    index_t D = graph.get_neighbor(N, j);
    if (D == INVALID || D >= n_vertices || !graph.is_deleted(D)) continue;
    uint8_t deg_D = graph.get_edge_count(D);
    for (uint8_t k = 0; k < deg_D; k++) {
      index_t M = graph.get_neighbor(D, k);
      if (M == INVALID || M >= n_vertices) continue;
      if (!graph.is_deleted(M) && M != N) local_count++;
    }
  }
  atomicAdd((unsigned long long int*)out_total, (unsigned long long int)local_count);
}

// Second pass: fill the repair candidate array with edge_pair{source=N, sink=M}.
// Distances are computed later by process_reverse_edges_kernel.
template <typename GRAPH_CFG>
__global__ void fill_repair_candidates_kernel(
    typename graph<GRAPH_CFG>::device_view graph,
    typename GRAPH_CFG::index_t n_vertices,
    typename GRAPH_CFG::index_t n_min,
    typename GRAPH_CFG::index_t n_max,
    edge_pair<typename GRAPH_CFG::index_t>* out_edges,
    uint64_t* out_count) {
  using index_t = typename GRAPH_CFG::index_t;
  constexpr index_t INVALID = std::numeric_limits<index_t>::max();

  uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n_vertices) return;
  index_t N = (index_t)tid;
  if (N < n_min || N >= n_max) return;
  if (graph.is_deleted(N)) return;

  uint8_t deg_N = graph.get_edge_count(N);
  for (uint8_t j = 0; j < deg_N; j++) {
    index_t D = graph.get_neighbor(N, j);
    if (D == INVALID || D >= n_vertices || !graph.is_deleted(D)) continue;
    uint8_t deg_D = graph.get_edge_count(D);
    for (uint8_t k = 0; k < deg_D; k++) {
      index_t M = graph.get_neighbor(D, k);
      if (M == INVALID || M >= n_vertices) continue;
      if (graph.is_deleted(M) || M == N) continue;
      uint64_t pos = atomicAdd((unsigned long long int*)out_count, 1ULL);
      out_edges[pos] = edge_pair<index_t>{N, M};
    }
  }
}

// Remove deleted entries from each non-deleted vertex's edge list, compacting
// in place. Unlike the source, jasperpy stores a parallel dist[] array, so we
// move the distance alongside each surviving edge. Run AFTER
// fill_repair_candidates_kernel (which still reads deleted neighbors' lists).
template <typename GRAPH_CFG>
__global__ void compact_deleted_neighbors_kernel(
    typename graph<GRAPH_CFG>::device_view graph,
    typename GRAPH_CFG::index_t n_vertices,
    typename GRAPH_CFG::index_t n_min,
    typename GRAPH_CFG::index_t n_max) {
  using index_t = typename GRAPH_CFG::index_t;
  constexpr index_t INVALID = std::numeric_limits<index_t>::max();

  uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n_vertices) return;
  index_t N = (index_t)tid;
  if (N < n_min || N >= n_max) return;
  if (graph.is_deleted(N)) return;

  uint8_t old_count = graph.get_edge_count(N);
  uint8_t new_count = 0;
  for (uint8_t j = 0; j < old_count; j++) {
    index_t nb = graph.get_neighbor(N, j);
    if (nb == INVALID || nb >= n_vertices) continue;
    if (!graph.is_deleted(nb)) {
      // new_count <= j always, so writing slot new_count never clobbers a
      // not-yet-read slot.
      float d = graph.get_neighbor_dist(N, j);
      graph.set_neighbor(N, new_count, nb);
      graph.set_neighbor_dist(N, new_count, d);
      new_count++;
    }
  }
  graph.set_edge_count(N, new_count);
}

// Zero out edges/edge_counts for all deleted nodes, completing consolidation.
template <typename GRAPH_CFG, uint32_t R>
__global__ void clear_deleted_nodes_kernel(
    typename graph<GRAPH_CFG>::device_view graph,
    typename GRAPH_CFG::index_t n_vertices) {
  using index_t = typename GRAPH_CFG::index_t;
  constexpr index_t INVALID = std::numeric_limits<index_t>::max();

  uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n_vertices) return;
  index_t D = (index_t)tid;
  if (!graph.is_deleted(D)) return;
  for (uint32_t j = 0; j < R; j++) graph.set_neighbor(D, j, INVALID);
  graph.set_edge_count(D, 0);
}

// ─────────────────────────── compaction kernels ────────────────────────────

// Mark each vertex live (edge_count>0 or medoid) → 1, else 0.
template <typename GRAPH_CFG>
__global__ void build_is_live_kernel(
    typename graph<GRAPH_CFG>::device_view graph,
    uint32_t* out_is_live,
    typename GRAPH_CFG::index_t medoid,
    typename GRAPH_CFG::index_t n_vertices) {
  using index_t = typename GRAPH_CFG::index_t;
  uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n_vertices) return;
  index_t i = (index_t)tid;
  out_is_live[i] = (graph.get_edge_count(i) > 0 || i == medoid) ? 1u : 0u;
}

// In-place "copy-down": move the M live vertices that sit above the live-count
// boundary L into the M holes below L. Bulk-synchronous — one thread per
// (mover, hole) pair, each already assigned its destination. Only these M
// items move (M <= number deleted); everything else stays put, so the result
// is NOT order-preserving. Edge IDs are copied verbatim (still OLD ids) and
// rewritten by remap_compacted_edges_kernel afterward. mapping[mover]=hole is
// recorded; mapping was identity-initialized for all stay-put vertices.
// Source (>=L) and dest (<L) never alias, so reads and writes are independent.
template <typename GRAPH_CFG>
__global__ void copy_down_kernel(
    typename graph<GRAPH_CFG>::device_view g,
    const typename GRAPH_CFG::index_t* movers,   // live indices in [L, n)
    const typename GRAPH_CFG::index_t* holes,    // dead indices in [0, L)
    typename GRAPH_CFG::index_t n_moves,
    uint32_t padded_dim,
    typename GRAPH_CFG::index_t* mapping) {
  using index_t = typename GRAPH_CFG::index_t;
  using data_t  = typename GRAPH_CFG::data_t;
  uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n_moves) return;
  index_t src  = movers[tid];
  index_t dest = holes[tid];

  uint8_t cnt = g.get_edge_count(src);
  g.set_edge_count(dest, cnt);
  for (uint8_t e = 0; e < cnt; e++) {
    g.set_neighbor(dest, e, g.get_neighbor(src, e));          // OLD id; remapped later
    g.set_neighbor_dist(dest, e, g.get_neighbor_dist(src, e));
  }
  data_t* s = g.get_vector(src);
  data_t* d = g.get_vector(dest);
  for (uint32_t x = 0; x < padded_dim; x++) d[x] = s[x];

  g.set_stable_id(dest, g.get_stable_id(src));  // carry the stable id down
  mapping[src] = dest;
}

// Rewrite every edge of the L compacted vertices [0, L) from OLD ids to new
// ids via mapping. Dead vertices are never referenced by a live vertex (the
// preceding consolidate guarantees it), so mapping only needs to be valid for
// live old ids: identity for stay-put vertices, hole for moved ones.
template <typename GRAPH_CFG>
__global__ void remap_compacted_edges_kernel(
    typename graph<GRAPH_CFG>::device_view g,
    const typename GRAPH_CFG::index_t* mapping,
    typename GRAPH_CFG::index_t n_live) {
  using index_t = typename GRAPH_CFG::index_t;
  uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n_live) return;
  index_t j = (index_t)tid;
  uint8_t cnt = g.get_edge_count(j);
  for (uint8_t e = 0; e < cnt; e++) {
    g.set_neighbor(j, e, mapping[g.get_neighbor(j, e)]);
  }
}

// Single-threaded: find a replacement for a deleted medoid. Prefer a
// non-deleted neighbor of the old medoid; else linear scan. Writes n_vertices
// on total failure (all deleted).
template <typename GRAPH_CFG>
__global__ void find_new_medoid_kernel(
    typename graph<GRAPH_CFG>::device_view graph,
    typename GRAPH_CFG::index_t old_medoid,
    typename GRAPH_CFG::index_t n_vertices,
    typename GRAPH_CFG::index_t* out_medoid) {
  using index_t = typename GRAPH_CFG::index_t;
  constexpr index_t INVALID = std::numeric_limits<index_t>::max();
  if (blockIdx.x != 0 || threadIdx.x != 0) return;

  uint8_t deg = graph.get_edge_count(old_medoid);
  for (uint8_t j = 0; j < deg; j++) {
    index_t nb = graph.get_neighbor(old_medoid, j);
    if (nb != INVALID && nb < n_vertices && !graph.is_deleted(nb)) {
      *out_medoid = nb;
      return;
    }
  }
  for (index_t v = 0; v < n_vertices; v++) {
    if (!graph.is_deleted(v)) {
      *out_medoid = v;
      return;
    }
  }
  *out_medoid = n_vertices;
}

}  // namespace jasper
