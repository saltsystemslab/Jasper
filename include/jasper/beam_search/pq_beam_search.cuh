// jasper/beam_search/pq_beam_search.cuh
//
// Product-Quantization variant of the directional beam search. The beam-search
// skeleton is identical to directional_beam_search.cuh; the only difference is
// how a node's out-neighbors are scored: instead of the cross-polytope LSH
// estimator, this path uses Asymmetric Distance Computation (ADC) over learned
// PQ codebooks (see docs/lsh_beam_search_v5.md).
#pragma once

#include <cub/cub.cuh>

#include "jasper/distance/distance.cuh"
#include "jasper/index/vector.cuh"
#include "jasper/index/graph.cuh"
#include "jasper/beam_search/config.cuh"
#include "jasper/beam_search/entry.cuh"
#include "jasper/beam_search/device_kernels.cuh"
#include "jasper/beam_search/directional_beam_search.cuh"  // reuse helpers + CLOCK_*
#include "jasper/pq/pq_codebooks.cuh"

namespace jasper {

// ===== Build the per-expansion ADC lookup table =====
// LUT[j*K + c] = <δ_j, C_j[c]>, where δ = est_vec (q-u for L2, q for IP).
// Computed once per popped node and reused for all of its neighbors.
template <typename DATA_T, uint32_t M, uint32_t K>
__device__ __forceinline__ void pq_build_lut(
    const DATA_T* __restrict__ est_vec,
    const float*  __restrict__ centroids,
    uint32_t                   dsub,
    float*        __restrict__ lut)
{
  const uint32_t total = M * K;
  for (uint32_t e = threadIdx.x; e < total; e += blockDim.x) {
    const uint32_t j  = e / K;                       // subspace
    const float*  Cj = centroids + static_cast<size_t>(e) * dsub;  // (j*K+c)*dsub
    const DATA_T* dj = est_vec  + static_cast<size_t>(j) * dsub;
    float acc = 0.0f;
    for (uint32_t t = 0; t < dsub; ++t)
      acc += static_cast<float>(dj[t]) * Cj[t];
    lut[e] = acc;
  }
  __syncthreads();
}

// ===== Score each new neighbor v of u via ADC table lookups =====
// With ê the PQ reconstruction of the residual e = v - u:
//   <δ, e> ≈ Σ_j LUT[j, code_j],   ||e||² ≈ Σ_j cnorm[j, code_j].
//   L2            → ||q-v||² ≈ ||q-u||² - 2·<q-u, ê> + ||ê||²
//   INNER_PRODUCT → <q,v>    ≈ <q,u> + <q, ê>   (stored negated, smaller==closer)
template <typename GRAPH_CFG, distance_func DISTANCE_FUNC, uint32_t M, uint32_t K>
__device__ __forceinline__ void pq_populate_estimated_distances(
    float                                   exact_u,
    typename graph<GRAPH_CFG>::device_view& graph,
    typename GRAPH_CFG::index_t             u_gid,
    const float* __restrict__               cnorm,
    const float* __restrict__               lut,
    ENTRY_T*     __restrict__               result_buffer,
    uint32_t                                offset,
    uint8_t                                 n_edges)
{
  using INDEX_T = typename GRAPH_CFG::index_t;
  constexpr INDEX_T INVALID_INDEX = static_cast<INDEX_T>(0x7FFFFFFFu);

  // Hoist u-side segment lookup (same layout math as the LSH estimator).
  const INDEX_T  u_lid    = u_gid - graph.global_offset;
  const uint32_t u_segid  = static_cast<uint32_t>(u_lid / GRAPH_CFG::vectors_per_segment);
  const uint32_t u_locidx = static_cast<uint32_t>(u_lid % GRAPH_CFG::vectors_per_segment);
  const auto&    u_segment = graph.segments[u_segid];
  const uint8_t* __restrict__ code_base = u_segment.edge_pqs[u_locidx].code;

  for (uint32_t e = threadIdx.x; e < n_edges; e += blockDim.x) {
    ENTRY_T entry = result_buffer[offset + e];
    INDEX_T v     = static_cast<INDEX_T>(get_index(entry));
    if (v == INVALID_INDEX) continue;

    const uint8_t* __restrict__ code = code_base + static_cast<size_t>(e) * M;
    float dot = 0.0f, norm = 0.0f;
    #pragma unroll
    for (uint32_t j = 0; j < M; ++j) {
      const uint32_t c = code[j];
      dot  += lut[j * K + c];
      norm += cnorm[j * K + c];
    }

    float est_dist;
    if constexpr (DISTANCE_FUNC == distance_func::L2) {
      est_dist = exact_u - 2.0f * dot + norm;
    } else {  // INNER_PRODUCT
      est_dist = -(exact_u + dot);
    }
    result_buffer[offset + e] = set_distance(entry, est_dist);
  }
  __syncthreads();
}

// ===== The kernel =====
template <typename GRAPH_CFG, uint32_t BLOCK_SIZE, distance_func DISTANCE_FUNC,
          uint32_t MAX_SEARCH_WIDTH, bool GET_VISITED,
          uint32_t TILE_SIZE = 4, uint32_t MAX_RESULT_SIZE = 1024>
__global__ void pq_beam_search_kernel(
    typename graph<GRAPH_CFG>::device_view graph,
    pq_codebooks_view<GRAPH_CFG::pq_m, GRAPH_CFG::pq_k> codebooks,
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
  constexpr uint32_t M = GRAPH_CFG::pq_m;
  constexpr uint32_t K = GRAPH_CFG::pq_k;
  static_assert(GRAPH_CFG::use_lsh,
                "pq_beam_search_kernel requires graph_cfg::use_lsh (directional storage)");

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
  //   smem_qu_diff[padded_dim]               DATA_T
  //   smem_lut[M*K]                          float   (PQ ADC table)
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
  DATA_T* __restrict__ smem_qu_diff   = reinterpret_cast<DATA_T*>(p);
  p += padded_dim * sizeof(DATA_T);

  p = reinterpret_cast<unsigned char*>(
        (reinterpret_cast<uintptr_t>(p) + 15) & ~uintptr_t(15));
  float* __restrict__ smem_lut = reinterpret_cast<float*>(p);

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
    result_buffer[0]         = set_distance(set_index(empty_entry(), medoid), 0.0f);
    result_buffer_count[0]   = 1;
    frontier_buffer_count[0] = 0;
  }
  __syncthreads();

  uint32_t visited_counter = 0;
  uint32_t loop_count      = 0;

  // For INNER_PRODUCT the ADC LUT uses δ = q (the query), which is constant for
  // the whole search, so build it once here. For L2, δ = q-u depends on the
  // popped node and is rebuilt inside the loop (step 6).
  if constexpr (DISTANCE_FUNC == distance_func::INNER_PRODUCT) {
    pq_build_lut<DATA_T, M, K>(
        smem_query_vec, codebooks.centroids, codebooks.dsub, smem_lut);
  }

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

    // 2) Mark visited in candidate buffer.
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

    // 3) Fetch u's vector; compute (q-u) into smem and exact base in one pass.
    CLOCK_START(t_exact);
    DATA_T* u_vec = graph.get_vector(u_gid);
    const float exact_u =
        compute_qu_diff_and_exact<DATA_T, BLOCK_SIZE, DISTANCE_FUNC>(
            smem_query_vec, u_vec, smem_qu_diff, padded_dim);
    float exact_dist_u;
    if constexpr (DISTANCE_FUNC == distance_func::L2) {
      exact_dist_u = exact_u;
    } else {
      exact_dist_u = -exact_u;
    }
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

    // 6) ESTIMATE ||q-v||² via PQ ADC: build the LUT, then score edges.
    // L2's δ = q-u changes each iteration, so its LUT is (re)built here; the
    // IP LUT (δ = q) is constant and was built once before the loop.
    CLOCK_START(t_estimate);
    const uint8_t n_edges = graph.get_edge_count(u_gid);
    if constexpr (DISTANCE_FUNC == distance_func::L2) {
      pq_build_lut<DATA_T, M, K>(
          smem_qu_diff, codebooks.centroids, codebooks.dsub, smem_lut);
    }
    pq_populate_estimated_distances<GRAPH_CFG, DISTANCE_FUNC, M, K>(
        exact_u, graph, u_gid, codebooks.cnorm, smem_lut,
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

  // ---- Write top-k from frontier_buffer (already sorted ascending) ----
  __syncthreads();
  const uint32_t fc = frontier_buffer_count[0];
  for (uint32_t i = threadIdx.x; i < k; i += BLOCK_SIZE) {
    if (i < fc) {
      frontier_results[query_id * k + i].first  =
          static_cast<INDEX_T>(get_index(frontier_buffer[i]));
      frontier_results[query_id * k + i].second =
          static_cast<DISTANCE_T>(get_distance(frontier_buffer[i]));
    } else {
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
template <typename GRAPH_CFG>
__host__ inline uint32_t get_pq_smem_size(
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
  s += sizeof(DATA_T) * padded_dim;                              // smem_qu_diff
  s = (s + 15) & ~15u;
  s += sizeof(float) * GRAPH_CFG::pq_m * GRAPH_CFG::pq_k;        // smem_lut
  return s;
}

// ===== Host: launcher =====
template <typename Cfg>
__host__ beam_search_result<typename Cfg::graph_cfg_t>
pq_beam_search(
    const beam_search_params<Cfg>&                                            p,
    pq_codebooks_view<Cfg::graph_cfg_t::pq_m, Cfg::graph_cfg_t::pq_k>          codebooks,
    cudaStream_t                                                              stream = 0)
{
  using graph_cfg_t = typename Cfg::graph_cfg_t;
  using entry_t     = typename Cfg::entry_t;
  static_assert(graph_cfg_t::use_lsh,
                "pq_beam_search requires graph_cfg::use_lsh (directional storage)");

  const uint32_t n_query_vectors = p.use_range
      ? (p.query_end - p.query_start)
      : p.query_vectors.n_vectors;

  dim3 threads(Cfg::block_size, 1, 1);
  dim3 blocks (n_query_vectors, 1, 1);

  const uint32_t padded_dim = p.graph.get_padded_dim();
  const uint32_t smem = get_pq_smem_size<graph_cfg_t>(
      p.beam_width, p.k, padded_dim);

  beam_search_result<graph_cfg_t> result{};
  cudaMalloc(&result.frontier, sizeof(entry_t) * n_query_vectors * p.k);
  if constexpr (Cfg::get_visited) {
    cudaMalloc(&result.visited,
               sizeof(entry_t) * n_query_vectors * Cfg::max_result_size);
    cudaMalloc(&result.visited_counts, sizeof(uint32_t) * n_query_vectors);
  }

  auto kernel = pq_beam_search_kernel<
      graph_cfg_t, Cfg::block_size, Cfg::dist_func,
      Cfg::max_search_width, Cfg::get_visited,
      Cfg::tile_size, Cfg::max_result_size>;

  if (smem > 48 * 1024) {
    cudaFuncSetAttribute(kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  }

  kernel<<<blocks, threads, smem, stream>>>(
      p.graph.view(), codebooks,
      result.frontier, result.visited, result.visited_counts,
      p.query_vectors, p.use_range, p.query_start, p.query_end,
      p.medoid, p.k, p.beam_width, p.limit);

  cudaError_t err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    std::cerr << "PQ beam search kernel launch failed: "
              << cudaGetErrorString(err) << std::endl;
  }
  return result;
}

}  // namespace jasper
