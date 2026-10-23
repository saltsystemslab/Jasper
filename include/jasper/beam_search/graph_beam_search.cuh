// jasper/beam_search/graph_beam_search.cuh
//
// Generic directional-storage beam search. The skeleton (pop → mark visited →
// fetch exact distance → grow frontier → expand out-edges → estimate their
// distances → drop copies of nodes already listed → sort/dedup/clip) is
// identical whether out-edges are scored with cross-polytope LSH or PQ/ADC;
// the only thing that differs is *how* a node's out-edges are scored. That
// scoring scheme is factored out into an "estimator" type (see lsh_estimator
// in directional_beam_search.cuh and pq_estimator in pq_beam_search.cuh), and
// this file is templated on it.
//
// An ESTIMATOR must provide:
//   - `using globals_t = ...;`             query-invariant per-graph metadata
//                                           (calibration constants / PQ
//                                           codebooks) passed into the kernel;
//                                           the quantized/hashed edges
//                                           themselves live on the graph and
//                                           are read directly by the
//                                           estimator's methods below.
//   - `__host__ static uint32_t extra_smem_size(uint32_t padded_dim);`
//                                           dynamic shared memory the
//                                           estimator needs beyond the common
//                                           candidate/frontier/query buffers.
//   - `__device__ ESTIMATOR(unsigned char* extra_smem, DATA_T* smem_query_vec,
//                            uint32_t padded_dim, const globals_t& globals);`
//                                           one-time per-query precomputation
//                                           (e.g. building a query ADC LUT).
//   - `__device__ float visit(device_view& graph, INDEX_T u_gid,
//                              DATA_T* smem_query_vec, uint32_t padded_dim);`
//                                           called once per popped candidate
//                                           u; returns the entry distance to
//                                           record for u (smaller == closer)
//                                           and stashes whatever scalar state
//                                           estimate_neighbors() needs.
//   - `__device__ void estimate_neighbors(device_view& graph, INDEX_T u_gid,
//                              DATA_T* smem_query_vec, ENTRY_T* result_buffer,
//                              uint32_t offset, uint8_t n_edges);`
//                                           scores u's newly-appended
//                                           out-edges at result_buffer[offset,
//                                           offset+n_edges).
//
// Termination. The search stops when the beam_width best candidates have all
// been expanded (classic beam search), on `limit`, or — with
// beam_search_params::adaptive_gamma ≥ 0 — on the distance rule of Adaptive
// Beam Search (Al-Jazzazi et al. 2025, eq. 3): stop once the next candidate is
// at least (1+γ)× farther than the k-th best other discovered candidate, both
// by estimate. See adaptive_stop_slack and step 2b of the kernel.
#pragma once

#include <cub/cub.cuh>

#include "jasper/distance/distance.cuh"
#include "jasper/index/vector.cuh"
#include "jasper/index/graph.cuh"
#include "jasper/beam_search/config.cuh"
#include "jasper/beam_search/entry.cuh"
#include "jasper/beam_search/device_kernels.cuh"
#include "jasper/beam_search/beam_search_common.cuh"

namespace jasper {

// ===== Host: adaptive termination slack =====
// The paper states the rule in metric distance: stop at candidate x once
// d(q,x) ≥ (1+γ)·d_k, d_k being the k-th best distance found. Entries hold
// SQUARED L2, where the same rule reads (1+γ)²·d_k², so γ keeps the paper's
// meaning (γ = 2 is exact on a navigable graph when distances are exact).
// Inner product is no metric, and its entries hold −⟨q,v⟩; there γ is a
// relative gap instead, stop once ⟨q,x⟩ ≤ ⟨q,v_k⟩ − γ·|⟨q,v_k⟩|. Both are
//   d_x ≥ d_k + slack·|d_k|
// on stored distances, which is the test the kernel makes. Negative γ (off)
// gives a negative slack.
template <distance_func DISTANCE_FUNC>
__host__ inline float adaptive_stop_slack(float gamma) {
  if (gamma < 0.0f) return -1.0f;
  if constexpr (DISTANCE_FUNC == distance_func::L2)
    return (1.0f + gamma) * (1.0f + gamma) - 1.0f;
  return gamma;
}

// Smallest b with 2^b >= x (x >= 1).
__host__ __device__ constexpr uint32_t ceil_log2(uint32_t x) {
  uint32_t b = 0;
  while ((1u << b) < x) ++b;
  return b;
}

// ===== The kernel =====
template <typename GRAPH_CFG, uint32_t BLOCK_SIZE, distance_func DISTANCE_FUNC,
          uint32_t MAX_SEARCH_WIDTH, typename ESTIMATOR, uint32_t TILE_SIZE = 4>
__global__ void graph_beam_search_kernel(
    typename graph<GRAPH_CFG>::device_view graph,
    typename ESTIMATOR::globals_t          globals,
    thrust::pair<typename GRAPH_CFG::index_t, typename GRAPH_CFG::distance_t>* frontier_results,
    thrust::pair<typename GRAPH_CFG::index_t, typename GRAPH_CFG::distance_t>* visited_results,
    uint32_t*                              visited_counts,
    vector_view<typename GRAPH_CFG::data_t> query_vectors,
    bool                                   use_range,
    uint32_t                               query_start,
    uint32_t                               query_end,
    typename GRAPH_CFG::index_t            medoid,
    uint32_t                               k,
    uint32_t                               beam_width,
    uint32_t                               limit,
    bool                                   get_visited,
    uint32_t                               max_result_size,
    float                                  adaptive_slack)
{
  using INDEX_T    = typename GRAPH_CFG::index_t;
  using DATA_T     = typename GRAPH_CFG::data_t;
  using DISTANCE_T = typename GRAPH_CFG::distance_t;
  static_assert(GRAPH_CFG::use_lsh,
                "graph_beam_search_kernel requires graph_cfg::use_lsh (directional storage)");

  const auto      query_id   = blockIdx.x;
  const uint32_t  padded_dim = graph.get_padded_dim();
  constexpr INDEX_T INVALID_INDEX = static_cast<INDEX_T>(0x7FFFFFFFu);

  assert(beam_width + GRAPH_CFG::n_neighbors <= MAX_SEARCH_WIDTH);

  // ---- Shared memory layout ----
  //   result_buffer[beam_width+n_neighbors]  ENTRY_T
  //   result_buffer_count                    uint32_t
  //   frontier_buffer[k]                     ENTRY_T
  //   frontier_buffer_count                  uint32_t
  //   merge_scratch[n_neighbors]             ENTRY_T
  //   smem_query_vec[padded_dim]             DATA_T
  //   <estimator-specific extra smem>                 (see ESTIMATOR::extra_smem_size)
  const uint32_t result_buffer_size = beam_width + GRAPH_CFG::n_neighbors;
  extern __shared__ __align__(16) unsigned char smem_raw[];
  unsigned char* p = smem_raw;

  auto* __restrict__ result_buffer       = reinterpret_cast<ENTRY_T*>(p);
  p += result_buffer_size * sizeof(ENTRY_T);
  auto* __restrict__ result_buffer_count = reinterpret_cast<uint32_t*>(p);
  p += sizeof(uint32_t);

  p = reinterpret_cast<unsigned char*>(
        (reinterpret_cast<uintptr_t>(p) + 7) & ~uintptr_t(7));
  auto* __restrict__ frontier_buffer       = reinterpret_cast<ENTRY_T*>(p);
  p += k * sizeof(ENTRY_T);
  auto* __restrict__ frontier_buffer_count = reinterpret_cast<uint32_t*>(p);
  p += sizeof(uint32_t);

  p = reinterpret_cast<unsigned char*>(
      (reinterpret_cast<uintptr_t>(p) + 7) & ~uintptr_t(7));
  ENTRY_T* __restrict__ merge_scratch = reinterpret_cast<ENTRY_T*>(p);
  p += GRAPH_CFG::n_neighbors * sizeof(ENTRY_T);

  p = reinterpret_cast<unsigned char*>(
        (reinterpret_cast<uintptr_t>(p) + 15) & ~uintptr_t(15));
  DATA_T* __restrict__ smem_query_vec = reinterpret_cast<DATA_T*>(p);
  p += padded_dim * sizeof(DATA_T);

  p = reinterpret_cast<unsigned char*>(
        (reinterpret_cast<uintptr_t>(p) + 15) & ~uintptr_t(15));
  unsigned char* __restrict__ extra_smem = p;

  // ---- Load query vector ----
  DATA_T* query_vec_src = use_range ? graph.get_vector(query_start + query_id)
                                    : query_vectors[query_id];
  for (uint32_t i = threadIdx.x; i < padded_dim; i += BLOCK_SIZE)
    smem_query_vec[i] = query_vec_src[i];

  // ---- Hash of the popped node's new neighbor ids (step 6b) ----
  // Twice the out-degree, rounded up to a power of two, so linear probing
  // stays short.
  constexpr uint32_t NEW_ID_HASH_BITS  = ceil_log2(2u * GRAPH_CFG::n_neighbors);
  constexpr uint32_t NEW_ID_HASH_SLOTS = 1u << NEW_ID_HASH_BITS;
  __shared__ uint32_t new_id_hash[NEW_ID_HASH_SLOTS];  // node id, or EMPTY_SLOT
  __shared__ uint32_t new_id_pos[NEW_ID_HASH_SLOTS];   // its offset past `offset`

  // ---- CUB temp storage for candidate sort/dedup ----
  constexpr uint32_t TAIL_EPT =
    (GRAPH_CFG::n_neighbors + BLOCK_SIZE - 1) / BLOCK_SIZE;
  using TailSortT  = cub::BlockMergeSort<ENTRY_T, BLOCK_SIZE, TAIL_EPT>;
  using BlockScanT = cub::BlockScan<uint32_t, BLOCK_SIZE>;
  union TempStorage {
    typename TailSortT::TempStorage  tail_sort_storage;
    typename BlockScanT::TempStorage scan_storage;
  };
  __shared__ TempStorage temp_storage;

  // ---- Initialize buffers; seed candidate list with medoid ----
  for (uint32_t i = threadIdx.x; i < result_buffer_size; i += BLOCK_SIZE)
    result_buffer[i] = empty_entry();
  if (threadIdx.x == 0) {
    // Medoid: estimated distance 0 so it pops first (real dist computed on pop).
    result_buffer[0]         = set_distance(set_index(empty_entry(), medoid), 0.0f);
    result_buffer_count[0]   = 1;
    frontier_buffer_count[0] = 0;
  }
  __syncthreads();

  uint32_t visited_counter = 0;
  uint32_t loop_count      = 0;

  // One-time per-query precomputation (e.g. building a query ADC LUT). Every
  // thread executes this uniformly, so any collective work the estimator's
  // constructor does (syncthreads-based reductions, etc.) is safe.
  ESTIMATOR estimator(extra_smem, smem_query_vec, padded_dim, globals);

  // ---- Main loop ----
  while (loop_count <= limit) {
    ++loop_count;

    // 1) Pop next candidate (lowest estimated distance, not yet visited).
    __syncthreads();
    CLOCK_START(t_pop);
    auto [frontierIdx, found] =
        choose_new_frontier<INDEX_T, DISTANCE_T, BLOCK_SIZE, MAX_SEARCH_WIDTH>(
            result_buffer, result_buffer_count);
    CLOCK_ACCUM(t_pop, PHASE_POP);
    if (!found) break;

    // 2) Mark visited in candidate buffer (so we never pop it again).
    if constexpr (BLOCK_SIZE > 33) {
      if (threadIdx.x == 33)
        result_buffer[frontierIdx] = set_visited(result_buffer[frontierIdx]);
    } else {
      if (threadIdx.x == 1)
        result_buffer[frontierIdx] = set_visited(result_buffer[frontierIdx]);
    }
    __syncthreads();

    const INDEX_T u_gid =
        static_cast<INDEX_T>(get_index(result_buffer[frontierIdx]));

    // 2b) Adaptive termination (adaptive_stop_slack): stop before visiting u
    //     once its estimate is past the slack around d_k, the k-th best of the
    //     OTHER discovered candidates. That is the paper's B (the k best
    //     discovered nodes, the candidate itself excluded), with the estimates
    //     standing in for distances directional search never computes: visited
    //     entries keep the estimate they were popped with. Both sides of the
    //     test are therefore estimates, and counting the list exactly as the
    //     beam-width rule does (the medoid's seed entry included) makes γ = 0
    //     stop where beam_width = k would (up to ties between estimates).
    //
    //     Not the k-th best EXACT distance in frontier_buffer: that ranks only
    //     the expanded nodes, a few hundred per search, so at k = 100 it sits
    //     near the worst node expanded and the test never trips.
    //
    //     Every thread reads the same shared values after the barrier above,
    //     so the break is block-uniform.
    if (adaptive_slack >= 0.0f) {
      // u is at frontierIdx; step past it when it is among the first k.
      const uint32_t kth = (frontierIdx < k) ? k : k - 1;
      if (kth < result_buffer_count[0]) {
        const float d_k = get_distance(result_buffer[kth]);
        if (get_distance(result_buffer[frontierIdx]) >=
            d_k + adaptive_slack * fabsf(d_k))
          break;
      }
    }

    // 3) Fetch u's exact/base distance via the estimator (the ONE I/O per
    //    explored node): ||q-u||² for L2, or the raw dot ⟨q,u⟩ for IP.
    CLOCK_START(t_exact);
    const float exact_dist_u =
        estimator.visit(graph, u_gid, smem_query_vec, padded_dim);
    CLOCK_ACCUM(t_exact, PHASE_EXACT);

    // 4) Add u to frontier (top-k explored by EXACT dist) and visited log.
    CLOCK_START(t_frontier);
    if (threadIdx.x == 0) {
      ENTRY_T u_entry =
          set_distance(set_index(empty_entry(), u_gid), exact_dist_u);
      frontier_insert_sorted(u_entry, exact_dist_u,
                             frontier_buffer, frontier_buffer_count, k);

      if (get_visited) {
        visited_results[query_id * max_result_size + visited_counter].first  = u_gid;
        visited_results[query_id * max_result_size + visited_counter].second =
            static_cast<DISTANCE_T>(exact_dist_u);
      }
    }
    ++visited_counter;
    if (visited_counter == max_result_size) break;
    __syncthreads();
    CLOCK_ACCUM(t_frontier, PHASE_FRONTIER);

    // 5) Append u's neighbors to candidates as placeholders (no neighbor I/O).
    CLOCK_START(t_expand);
    const uint32_t offset = result_buffer_count[0];
    __syncthreads();
    add_frontier_out<GRAPH_CFG>(graph, result_buffer, result_buffer_count,
                                u_gid, k, beam_width);
    __syncthreads();
    CLOCK_ACCUM(t_expand, PHASE_EXPAND);

    // 6) ESTIMATE distances for the newly-appended neighbors.
    CLOCK_START(t_estimate);
    const uint8_t n_edges = graph.get_edge_count(u_gid);
    estimator.estimate_neighbors(graph, u_gid, smem_query_vec,
                                 result_buffer, offset, n_edges);
    CLOCK_ACCUM(t_estimate, PHASE_ESTIMATE);

    // 6b) Keep one list entry per node. A node reached from two parents carries
    //     two different estimates, so its copies do not sort next to each
    //     other and dedup_results cannot pair them. Left alone, every copy
    //     takes a beam slot, and the best nodes, reached from the most parents,
    //     take the most: a beam of width L holds far fewer than L distinct
    //     candidates. Copies of an expanded node also pop again, re-expand, and
    //     re-enter the top-k as repeats.
    //
    //     So each new neighbor is checked against the list (head) and the top-k
    //     before it is merged in. The list holds at most one copy per node
    //     going in, by induction, so there is at most one match:
    //       - in the top-k, or matched by a visited list entry: the node is
    //         expanded, and the new copy is dropped;
    //       - matched by an unvisited entry: the lower estimate stays, as it
    //         would have popped first anyway.
    //     A node expanded and since clipped from both the list and the top-k
    //     can still come back and be expanded again: a wasted pop, but its
    //     exact distance cannot win it a place in the top-k a second time.
    //
    //     The new ids go into a small hash table in shared memory, and each
    //     list and top-k entry looks itself up once. Dropped entries get
    //     DROPPED_INDEX and keep their estimate, so the head stays sorted for
    //     the merge, and dedup_results removes them after it.
    CLOCK_START(t_unique);
    {
      constexpr uint32_t EMPTY_SLOT = 0xFFFFFFFFu;
      const auto slot_of = [&](uint32_t id) {
        return (id * 0x9E3779B1u) >> (32u - NEW_ID_HASH_BITS);
      };
      const auto drop = [](ENTRY_T e) { return set_index(e, DROPPED_INDEX); };

      for (uint32_t s = threadIdx.x; s < NEW_ID_HASH_SLOTS; s += BLOCK_SIZE)
        new_id_hash[s] = EMPTY_SLOT;
      __syncthreads();

      for (uint32_t j = threadIdx.x; j < n_edges; j += BLOCK_SIZE) {
        const uint32_t id = get_index(result_buffer[offset + j]);
        if (id == INVALID_INDEX) continue;  // out-of-range neighbor
        for (uint32_t s = slot_of(id);; s = (s + 1) & (NEW_ID_HASH_SLOTS - 1)) {
          const uint32_t prev = atomicCAS(&new_id_hash[s], EMPTY_SLOT, id);
          if (prev == EMPTY_SLOT) { new_id_pos[s] = j; break; }
          if (prev == id) {  // listed twice in u's edges
            result_buffer[offset + j] = drop(result_buffer[offset + j]);
            break;
          }
        }
      }
      __syncthreads();

      // Position of id's new copy, or ~0u if u's edges do not reach id.
      const auto new_copy_of = [&](uint32_t id) -> uint32_t {
        for (uint32_t s = slot_of(id);; s = (s + 1) & (NEW_ID_HASH_SLOTS - 1)) {
          const uint32_t h = new_id_hash[s];
          if (h == id) return new_id_pos[s];
          if (h == EMPTY_SLOT) return ~0u;
        }
      };
      for (uint32_t i = threadIdx.x; i < offset; i += BLOCK_SIZE) {
        const ENTRY_T e  = result_buffer[i];
        const uint32_t id = get_index(e);
        if (id == INVALID_INDEX || id == DROPPED_INDEX) continue;
        const uint32_t j = new_copy_of(id);
        if (j == ~0u) continue;
        const ENTRY_T fresh = result_buffer[offset + j];
        // Qualified: the kernel's bool `get_visited` parameter hides the helper.
        if (jasper::get_visited(e) || get_distance(e) <= get_distance(fresh))
          result_buffer[offset + j] = drop(fresh);
        else
          result_buffer[i] = drop(e);
      }
      for (uint32_t i = threadIdx.x; i < frontier_buffer_count[0]; i += BLOCK_SIZE) {
        const uint32_t j = new_copy_of(get_index(frontier_buffer[i]));
        if (j != ~0u) result_buffer[offset + j] = drop(result_buffer[offset + j]);
      }
      __syncthreads();
    }
    CLOCK_ACCUM(t_unique, PHASE_DEDUP);

    // 7) Sort / dedup / clip the candidate buffer (estimated distances).
    CLOCK_START(t_sort);
    merge_sort_tail<INDEX_T, DISTANCE_T, BLOCK_SIZE, MAX_SEARCH_WIDTH,
                GRAPH_CFG::n_neighbors, TailSortT>(
      result_buffer, result_buffer_count, /*head_len=*/offset,
      merge_scratch, temp_storage.tail_sort_storage);
    CLOCK_ACCUM(t_sort, PHASE_SORT);

    CLOCK_START(t_dedup);
    dedup_results<INDEX_T, DISTANCE_T, BLOCK_SIZE, MAX_SEARCH_WIDTH, BlockScanT>(
        result_buffer, result_buffer_count, temp_storage.scan_storage);
    CLOCK_ACCUM(t_dedup, PHASE_DEDUP);

    CLOCK_START(t_clip);
    clip_k(result_buffer_count, beam_width);
    CLOCK_ACCUM(t_clip,  PHASE_CLIP);
  }

  // ---- Write top-k from frontier_buffer (already sorted ascending),
  //      skipping any soft-deleted vertices. ----
  __syncthreads();
  if (threadIdx.x == 0) {
    const uint32_t fc = frontier_buffer_count[0];
    uint32_t out = 0;
    for (uint32_t i = 0; i < fc && out < k; i++) {
      uint32_t idx = get_index(frontier_buffer[i]);
      if (!graph.is_valid(idx)) continue;
      if (graph.is_deleted(idx)) continue;
      frontier_results[query_id * k + out].first  = static_cast<INDEX_T>(idx);
      frontier_results[query_id * k + out].second =
          static_cast<DISTANCE_T>(get_distance(frontier_buffer[i]));
      out++;
    }
    for (uint32_t i = out; i < k; i++) {
      frontier_results[query_id * k + i].first  = INVALID_INDEX;
      frontier_results[query_id * k + i].second =
          std::numeric_limits<DISTANCE_T>::max();
    }
  }
  if (threadIdx.x == 0 && get_visited) {
    visited_counts[query_id] = visited_counter;
  }
}

// ===== Host: smem size =====
template <typename GRAPH_CFG, typename ESTIMATOR>
__host__ inline uint32_t get_graph_search_smem_size(
    uint32_t beam_width, uint32_t k, uint32_t padded_dim)
{
  using DATA_T = typename GRAPH_CFG::data_t;
  uint32_t s = 0;
  s += sizeof(ENTRY_T) * (beam_width + GRAPH_CFG::n_neighbors); // result_buffer
  s += sizeof(uint32_t);                                         // result_buffer_count
  s = (s + 7) & ~7u;
  s += sizeof(ENTRY_T) * k;                                      // frontier_buffer
  s += sizeof(uint32_t);                                         // frontier_buffer_count
  s = (s + 7) & ~7u;
  s += sizeof(ENTRY_T) * GRAPH_CFG::n_neighbors;                 // merge_scratch
  s = (s + 15) & ~15u;
  s += sizeof(DATA_T) * padded_dim;                              // smem_query_vec
  s = (s + 15) & ~15u;
  s += ESTIMATOR::extra_smem_size(padded_dim);                   // estimator's extra smem
  return s;
}

// ===== Host: launcher =====
template <typename Cfg, typename ESTIMATOR>
__host__ beam_search_result<typename Cfg::graph_cfg_t>
graph_beam_search(
    const beam_search_params<Cfg>& p,
    typename ESTIMATOR::globals_t  globals,
    cudaStream_t                   stream = 0)
{
  using graph_cfg_t = typename Cfg::graph_cfg_t;
  using entry_t     = typename Cfg::entry_t;
  static_assert(graph_cfg_t::use_lsh,
                "graph_beam_search requires graph_cfg::use_lsh (directional storage)");

  const uint32_t n_query_vectors = p.use_range
      ? (p.query_end - p.query_start)
      : p.query_vectors.n_vectors;

  dim3 threads(Cfg::block_size, 1, 1);
  dim3 blocks (n_query_vectors, 1, 1);

  const uint32_t padded_dim = p.graph.get_padded_dim();
  const uint32_t smem = get_graph_search_smem_size<graph_cfg_t, ESTIMATOR>(
      p.beam_width, p.k, padded_dim);
  const float adaptive_slack = adaptive_stop_slack<Cfg::dist_func>(p.adaptive_gamma);

  beam_search_result<graph_cfg_t> result{};
  cudaMalloc(&result.frontier, sizeof(entry_t) * n_query_vectors * p.k);
  if (p.get_visited) {
    cudaMalloc(&result.visited,
               sizeof(entry_t) * n_query_vectors * p.max_result_size);
    cudaMalloc(&result.visited_counts, sizeof(uint32_t) * n_query_vectors);
  }

  auto kernel = graph_beam_search_kernel<
      graph_cfg_t, Cfg::block_size, Cfg::dist_func,
      Cfg::max_search_width, ESTIMATOR, Cfg::tile_size>;

  if (smem > 48 * 1024) {
    cudaFuncSetAttribute(kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  }

  kernel<<<blocks, threads, smem, stream>>>(
      p.graph.view(), globals,
      result.frontier, result.visited, result.visited_counts,
      p.query_vectors, p.use_range, p.query_start, p.query_end,
      p.medoid, p.k, p.beam_width, p.limit,
      p.get_visited, p.max_result_size, adaptive_slack);

  cudaError_t err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    std::cerr << "Beam search kernel launch failed: "
              << cudaGetErrorString(err) << std::endl;
  }
  return result;
}

// Variant with pre-allocated result buffer, matching beam_search.cuh.
template <typename Cfg, typename ESTIMATOR>
__host__ void
graph_beam_search(
    const beam_search_params<Cfg>&                 p,
    typename ESTIMATOR::globals_t                  globals,
    beam_search_result<typename Cfg::graph_cfg_t>  result,
    cudaStream_t                                   stream = 0)
{
  using graph_cfg_t = typename Cfg::graph_cfg_t;
  static_assert(graph_cfg_t::use_lsh,
                "graph_beam_search requires graph_cfg::use_lsh (directional storage)");

  const uint32_t n_query_vectors = p.use_range
      ? (p.query_end - p.query_start)
      : p.query_vectors.n_vectors;

  dim3 threads(Cfg::block_size, 1, 1);
  dim3 blocks (n_query_vectors, 1, 1);

  const uint32_t padded_dim = p.graph.get_padded_dim();
  const uint32_t smem = get_graph_search_smem_size<graph_cfg_t, ESTIMATOR>(
      p.beam_width, p.k, padded_dim);
  const float adaptive_slack = adaptive_stop_slack<Cfg::dist_func>(p.adaptive_gamma);

  auto kernel = graph_beam_search_kernel<
      graph_cfg_t, Cfg::block_size, Cfg::dist_func,
      Cfg::max_search_width, ESTIMATOR, Cfg::tile_size>;

  if (smem > 48 * 1024) {
    cudaFuncSetAttribute(kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  }

  kernel<<<blocks, threads, smem, stream>>>(
      p.graph.view(), globals,
      result.frontier, result.visited, result.visited_counts,
      p.query_vectors, p.use_range, p.query_start, p.query_end,
      p.medoid, p.k, p.beam_width, p.limit,
      p.get_visited, p.max_result_size, adaptive_slack);

  cudaError_t err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    std::cerr << "Beam search kernel launch failed: "
              << cudaGetErrorString(err) << std::endl;
  }
}

}  // namespace jasper
