// jasper/beam_search/graph_beam_search.cuh
//
// Generic directional-storage beam search. The skeleton (pop → mark visited →
// fetch exact distance → grow frontier → expand out-edges → estimate their
// distances → sort/dedup/clip) is identical whether out-edges are scored with
// cross-polytope LSH or PQ/ADC; the only thing that differs is *how* a node's
// out-edges are scored. That scoring scheme is factored out into an
// "estimator" type (see lsh_estimator in directional_beam_search.cuh and
// pq_estimator in pq_beam_search.cuh), and this file is templated on it.
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

// ===== The kernel =====
template <typename GRAPH_CFG, uint32_t BLOCK_SIZE, distance_func DISTANCE_FUNC,
          uint32_t MAX_SEARCH_WIDTH, bool GET_VISITED, typename ESTIMATOR,
          uint32_t TILE_SIZE = 4, uint32_t MAX_RESULT_SIZE = 1024>
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
    uint32_t                               limit)
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

      if (GET_VISITED) {
        visited_results[query_id * MAX_RESULT_SIZE + visited_counter].first  = u_gid;
        visited_results[query_id * MAX_RESULT_SIZE + visited_counter].second =
            static_cast<DISTANCE_T>(exact_dist_u);
      }
    }
    ++visited_counter;
    if (visited_counter == MAX_RESULT_SIZE) break;
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
  if (threadIdx.x == 0 && GET_VISITED) {
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

  beam_search_result<graph_cfg_t> result{};
  cudaMalloc(&result.frontier, sizeof(entry_t) * n_query_vectors * p.k);
  if constexpr (Cfg::get_visited) {
    cudaMalloc(&result.visited,
               sizeof(entry_t) * n_query_vectors * Cfg::max_result_size);
    cudaMalloc(&result.visited_counts, sizeof(uint32_t) * n_query_vectors);
  }

  auto kernel = graph_beam_search_kernel<
      graph_cfg_t, Cfg::block_size, Cfg::dist_func,
      Cfg::max_search_width, Cfg::get_visited, ESTIMATOR,
      Cfg::tile_size, Cfg::max_result_size>;

  if (smem > 48 * 1024) {
    cudaFuncSetAttribute(kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  }

  kernel<<<blocks, threads, smem, stream>>>(
      p.graph.view(), globals,
      result.frontier, result.visited, result.visited_counts,
      p.query_vectors, p.use_range, p.query_start, p.query_end,
      p.medoid, p.k, p.beam_width, p.limit);

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

  auto kernel = graph_beam_search_kernel<
      graph_cfg_t, Cfg::block_size, Cfg::dist_func,
      Cfg::max_search_width, Cfg::get_visited, ESTIMATOR,
      Cfg::tile_size, Cfg::max_result_size>;

  if (smem > 48 * 1024) {
    cudaFuncSetAttribute(kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  }

  kernel<<<blocks, threads, smem, stream>>>(
      p.graph.view(), globals,
      result.frontier, result.visited, result.visited_counts,
      p.query_vectors, p.use_range, p.query_start, p.query_end,
      p.medoid, p.k, p.beam_width, p.limit);

  cudaError_t err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    std::cerr << "Beam search kernel launch failed: "
              << cudaGetErrorString(err) << std::endl;
  }
}

}  // namespace jasper
