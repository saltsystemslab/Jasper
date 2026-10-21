#pragma once

#include <cub/cub.cuh>
#include <algorithm>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#include <cooperative_groups.h>

#include "jasper/index/vector.cuh"
#include "jasper/index/graph.cuh"
#include "jasper/beam_search/config.cuh"
#include "jasper/beam_search/entry.cuh"
#include "jasper/beam_search/device_kernels.cuh"
#include "jasper/distance/distance.cuh"
#include "jasper/rabitq/distance.cuh"
#include "jasper/rabitq/index.cuh"
#include "jasper/rabitq/layout.cuh"

#include "assert.h"
#include "stdio.h"

namespace cg = cooperative_groups;

namespace jasper {

// RaBitQ counterpart of populate_distances: fill estimated distances for
// result_buffer[offset, count), one tile per candidate.
template <uint32_t BITS, typename GRAPH_CFG, uint32_t TILE_SIZE>
__device__ void populate_distances_rabitq_bits(
    const float* rot_query,
    const RabitqQueryFactor& query_factor,
    const rabitq_data_store<float>& codes,
    typename graph<GRAPH_CFG>::device_view& graph,
    ENTRY_T *result_buffer,
    uint32_t *result_buffer_count,
    uint32_t offset) {

  using INDEX_T = typename GRAPH_CFG::index_t;

  auto thread_block = cg::this_thread_block();
  cg::thread_block_tile<TILE_SIZE> my_tile =
      cg::tiled_partition<TILE_SIZE>(thread_block);
  uint64_t tid = my_tile.meta_group_rank();
  // 0x7FFFFFFF matches the index stored by empty_entry() (31-bit max after masking)
  constexpr INDEX_T INVALID_INDEX = static_cast<INDEX_T>(0x7FFFFFFFu);

  uint32_t count = result_buffer_count[0];
  for (unsigned i = tid + offset; i < count; i += my_tile.meta_group_size()) {
    auto dest = get_index(result_buffer[i]);
    if (dest != INVALID_INDEX) {
      float dist = rabitq_estimate_distance<BITS>(
          codes, graph.to_local(dest), rot_query, query_factor, my_tile);
      if (my_tile.thread_rank() == 0) {
        result_buffer[i] = set_distance(result_buffer[i], dist);
      }
    }
  }
}

template <typename GRAPH_CFG, uint32_t TILE_SIZE>
__device__ void populate_distances_rabitq(
    const float* rot_query,
    const RabitqQueryFactor& query_factor,
    const rabitq_data_store<float>& codes,
    typename graph<GRAPH_CFG>::device_view& graph,
    ENTRY_T *result_buffer,
    uint32_t *result_buffer_count,
    uint32_t offset) {
  // bits_per_dim is uniform across the grid, so this switch never diverges.
  // Dispatching here rather than on a template parameter keeps one search
  // kernel per (config, width) instead of one per bit width.
  switch (codes.bits_per_dim) {
    case 1:
      populate_distances_rabitq_bits<1, GRAPH_CFG, TILE_SIZE>(
          rot_query, query_factor, codes, graph, result_buffer, result_buffer_count, offset);
      break;
    case 2:
      populate_distances_rabitq_bits<2, GRAPH_CFG, TILE_SIZE>(
          rot_query, query_factor, codes, graph, result_buffer, result_buffer_count, offset);
      break;
    case 4:
      populate_distances_rabitq_bits<4, GRAPH_CFG, TILE_SIZE>(
          rot_query, query_factor, codes, graph, result_buffer, result_buffer_count, offset);
      break;
    default:
      populate_distances_rabitq_bits<8, GRAPH_CFG, TILE_SIZE>(
          rot_query, query_factor, codes, graph, result_buffer, result_buffer_count, offset);
      break;
  }
  __threadfence();
  __syncthreads();
}

// Beam search scored by RaBitQ estimates, then an exact rerank of the final
// beam. Same walk as beam_search_kernel; only the distance source differs.
template <typename GRAPH_CFG, uint32_t BLOCK_SIZE, distance_func DISTANCE_FUNC,
          uint32_t MAX_SEARCH_WIDTH, uint32_t TILE_SIZE = 8>
__global__ void rabitq_beam_search_kernel(
    typename graph<GRAPH_CFG>::device_view graph,
    rabitq_data_store<float> codes,
    const float *__restrict__ rot_queries,                // [n_queries * dim]
    const RabitqQueryFactor *__restrict__ query_factors,  // [n_queries]
    thrust::pair<typename GRAPH_CFG::index_t, typename GRAPH_CFG::distance_t> *frontier_results,
    thrust::pair<typename GRAPH_CFG::index_t, typename GRAPH_CFG::distance_t> *visited_results,
    uint32_t *visited_counts,
    vector_view<typename GRAPH_CFG::data_t> query_vectors,  // original queries, for the rerank
    typename GRAPH_CFG::index_t medoid,
    uint32_t k,
    uint32_t beam_width,
    uint32_t limit,
    bool get_visited,
    uint32_t max_result_size) {

  using INDEX_T = typename GRAPH_CFG::index_t;
  using DATA_T = typename GRAPH_CFG::data_t;
  using DISTANCE_T = typename GRAPH_CFG::distance_t;

  // get the query vector for this block
  const auto query_id = blockIdx.x;
  uint32_t visited_counter = 0;

  assert(beam_width + 64 <= MAX_SEARCH_WIDTH);

  // allocate shared memory
  const uint32_t result_buffer_size = beam_width + 64;
  extern __shared__ __align__(128) uint32_t smem[];
  auto *__restrict__ result_buffer = reinterpret_cast<ENTRY_T *>(smem);
  auto *__restrict__ result_buffer_count = reinterpret_cast<uint32_t *>(result_buffer + result_buffer_size);

  // One 16-byte aligned query region: it holds the rotated float query during
  // the walk, then the original query for the exact rerank. The two pointers
  // alias, so neither is __restrict__.
  uintptr_t query_vec_offset = reinterpret_cast<uintptr_t>(result_buffer_count + 1);
  query_vec_offset = (query_vec_offset + 15) & ~uintptr_t(15);
  auto *smem_rot_query = reinterpret_cast<float *>(query_vec_offset);
  auto *smem_query_vec = reinterpret_cast<DATA_T *>(query_vec_offset);

  // Load the rotated query, zero-padded to the estimator's read width.
  const uint32_t dim = codes.dim;
  const uint32_t rot_padded_dim = codes.query_padded_dim();
  const float *rot_query = rot_queries + static_cast<size_t>(query_id) * dim;
  for (uint32_t i = threadIdx.x; i < rot_padded_dim; i += blockDim.x) {
    smem_rot_query[i] = (i < dim) ? rot_query[i] : 0.0f;
  }
  const RabitqQueryFactor query_factor = query_factors[query_id];

  // cub temporary workspace
  constexpr uint32_t ELEMENTS_PER_THREAD = (MAX_SEARCH_WIDTH - 1) / BLOCK_SIZE + 1;
  using BlockMergeSortT = cub::BlockMergeSort<ENTRY_T, BLOCK_SIZE, ELEMENTS_PER_THREAD>;
  using BlockScanT = cub::BlockScan<uint32_t, BLOCK_SIZE>;
  union TempStorage {
    typename cub::BlockMergeSort<ENTRY_T, BLOCK_SIZE, ELEMENTS_PER_THREAD>::TempStorage sort_storage;
    typename cub::BlockScan<uint32_t, BLOCK_SIZE>::TempStorage scan_storage;
  };
  __shared__ TempStorage temp_storage;

  // initialize shared memory to default (invalid) key
  for (unsigned i = threadIdx.x; i < result_buffer_size; i += blockDim.x) {
    result_buffer[i] = empty_entry();
  }

  // populate frontier in shared memory
  if (threadIdx.x == 0) {
    result_buffer[0] = set_index(empty_entry(), medoid);
    result_buffer_count[0] = 1;
  }
  __syncthreads();

  populate_distances_rabitq<GRAPH_CFG, TILE_SIZE>(
      smem_rot_query, query_factor, codes, graph,
      result_buffer, result_buffer_count, 0);

  // loop
  uint32_t offset;
  uint32_t loop_count = 0;
  while (loop_count <= limit) {

    loop_count += 1;
    // choose a new frontier to explore
    auto [frontierIdx, found] =
        choose_new_frontier<INDEX_T, DISTANCE_T, BLOCK_SIZE, MAX_SEARCH_WIDTH>(
          result_buffer, result_buffer_count);
    if (!found) {
      break;  // we converged
    };

    if constexpr (BLOCK_SIZE > 33) {
      if (threadIdx.x == 33) {
        result_buffer[frontierIdx] = set_visited(result_buffer[frontierIdx]);
      }
    } else {
      if (threadIdx.x == 1) {
        result_buffer[frontierIdx] = set_visited(result_buffer[frontierIdx]);
      }
    }
    __syncthreads();

    // Add frontier to visited list (distances are the RaBitQ estimates)
    if (threadIdx.x == 0 && get_visited) {
      visited_results[query_id * max_result_size + visited_counter].first =
          get_index(result_buffer[frontierIdx]);
      visited_results[query_id * max_result_size + visited_counter].second =
          get_distance(result_buffer[frontierIdx]);
    }
    visited_counter += 1;
    // break if we exceeded the result size for visited list.
    if (visited_counter == max_result_size) break;
    __syncthreads();

    // record offset so that we know where to start calculating distances
    // (all the previous ones are calculated)
    offset = result_buffer_count[0];
    __syncthreads();

    // Add frontier's neighbor to results
    add_frontier_out<GRAPH_CFG>(graph, result_buffer, result_buffer_count,
                     get_index(result_buffer[frontierIdx]), k, beam_width);
    __syncthreads();

    // populate estimated distances
    populate_distances_rabitq<GRAPH_CFG, TILE_SIZE>(
        smem_rot_query, query_factor, codes, graph,
        result_buffer, result_buffer_count, offset);

    // sort the result buffer
    merge_sort<INDEX_T, DISTANCE_T, BLOCK_SIZE, MAX_SEARCH_WIDTH, BlockMergeSortT>(
        result_buffer, result_buffer_count, temp_storage.sort_storage);

    dedup_results<INDEX_T, DISTANCE_T, BLOCK_SIZE, MAX_SEARCH_WIDTH, BlockScanT>(
        result_buffer, result_buffer_count, temp_storage.scan_storage);

    clip_k(result_buffer_count, beam_width);
  }

  // ── Rerank the final beam with exact distances ──
  // Swap the rotated query for the original one, drop candidates that can't be
  // returned (empty slots, soft-deleted vertices) so they skip the exact
  // distance and sort last, then re-sort by exact distance.
  __syncthreads();
  const DATA_T *query_vec = query_vectors[query_id];
  for (uint i = threadIdx.x; i < graph.get_padded_dim(); i += blockDim.x) {
    smem_query_vec[i] = query_vec[i];
  }
  const uint32_t beam_count = result_buffer_count[0];
  for (uint32_t i = threadIdx.x; i < beam_count; i += blockDim.x) {
    const uint32_t idx = get_index(result_buffer[i]);
    if (!graph.is_valid(idx) || graph.is_deleted(idx)) {
      result_buffer[i] = empty_entry();
    }
  }
  __syncthreads();

  populate_distances<GRAPH_CFG, TILE_SIZE, DISTANCE_FUNC>(
      smem_query_vec, graph, result_buffer, result_buffer_count, 0);
  merge_sort<INDEX_T, DISTANCE_T, BLOCK_SIZE, MAX_SEARCH_WIDTH, BlockMergeSortT>(
      result_buffer, result_buffer_count, temp_storage.sort_storage);

  // copy to the external result, skipping any soft-deleted vertices so they
  // never surface in search results.
  if (threadIdx.x == 0) {
    const uint32_t buf_count = result_buffer_count[0];
    uint32_t out = 0;
    for (uint32_t i = 0; i < buf_count && out < k; i++) {
      uint32_t idx = get_index(result_buffer[i]);
      if (!graph.is_valid(idx)) continue;       // empty / out-of-range slot
      if (graph.is_deleted(idx)) continue;       // soft-deleted
      frontier_results[query_id * k + out].first  = idx;
      frontier_results[query_id * k + out].second = get_distance(result_buffer[i]);
      out++;
    }
    for (uint32_t i = out; i < k; i++) {
      frontier_results[query_id * k + i].first  = std::numeric_limits<INDEX_T>::max();
      frontier_results[query_id * k + i].second = static_cast<DISTANCE_T>(1e30f);
    }
  }
  if (threadIdx.x == 0 && get_visited) {
    visited_counts[query_id] = visited_counter;
  }
}

// Shared memory for rabitq_beam_search_kernel, in bytes. The query region is
// shared by the rotated float query (walk) and the original query (rerank).
template <typename DATA_T>
__host__ uint32_t get_rabitq_smem_size(const uint32_t beam_width,
                                       const uint32_t padded_dim,
                                       const uint32_t rot_padded_dim) {
  uint32_t smem_size = 0;
  smem_size += sizeof(ENTRY_T) * (beam_width + 64);  // result_buffer
  smem_size += sizeof(uint32_t);                      // result_buffer_count
  smem_size = (smem_size + 15) & ~15u;                // align to 16 bytes
  smem_size += std::max<uint32_t>(sizeof(float) * rot_padded_dim,
                                  sizeof(DATA_T) * padded_dim);  // query region
  return smem_size;
}

// Throws if `rq` can't serve searches on `g`: missing, built for a different
// dimension, or stale (slot count changed by append_batch / compact).
template <typename GRAPH_CFG>
__host__ void validate_rabitq_index(const graph<GRAPH_CFG>& g,
                                    const rabitq_index& rq) {
  if (rq.empty()) {
    throw std::invalid_argument(
        "rabitq_search: RaBitQ index is empty; build it with build_rabitq_index()");
  }
  if (rq.dim() != g.dim) {
    throw std::invalid_argument(
        "rabitq_search: RaBitQ index dim " + std::to_string(rq.dim()) +
        " != graph dim " + std::to_string(g.dim));
  }
  if (rq.n_vectors() != static_cast<uint64_t>(g.n_vectors)) {
    throw std::runtime_error(
        "rabitq_search: RaBitQ index is stale (built for " +
        std::to_string(rq.n_vectors()) + " vectors, graph now has " +
        std::to_string(static_cast<uint64_t>(g.n_vectors)) +
        "); rebuild it with build_rabitq_index() after append/compact");
  }
}

template <typename Cfg>
__host__ beam_search_result<typename Cfg::graph_cfg_t>
rabitq_beam_search(const beam_search_params<Cfg>& p,
                   const rabitq_index& rq,
                   const rabitq_queries& queries,
                   cudaStream_t stream = 0) {

  using entry_t = typename Cfg::entry_t;

  if (p.use_range) {
    throw std::invalid_argument(
        "rabitq_beam_search: use_range is not supported; pass query vectors");
  }
  validate_rabitq_index(p.graph, rq);
  const uint32_t n_query_vectors = p.query_vectors.n_vectors;
  if (queries.n_queries != n_query_vectors || queries.dim != p.graph.dim) {
    throw std::invalid_argument(
        "rabitq_beam_search: prepared queries don't match query_vectors");
  }

  beam_search_result<typename Cfg::graph_cfg_t> result{};

  // Allocate frontier
  cudaMalloc(&result.frontier, sizeof(entry_t) * n_query_vectors * p.k);
  // Allocate visited
  if (p.get_visited) {
    cudaMalloc(&result.visited,
               sizeof(entry_t) * n_query_vectors * p.max_result_size);
    cudaMalloc(&result.visited_counts,
               sizeof(uint32_t) * n_query_vectors);
  }
  if (n_query_vectors == 0) return result;

  dim3 threads(Cfg::block_size, 1, 1);
  dim3 blocks(n_query_vectors, 1, 1);
  uint32_t smem = get_rabitq_smem_size<typename Cfg::data_t>(
      p.beam_width, p.graph.get_padded_dim(), rq.codes.query_padded_dim());

  auto kernel = rabitq_beam_search_kernel<
      typename Cfg::graph_cfg_t,
      Cfg::block_size, Cfg::dist_func,
      Cfg::max_search_width,
      Cfg::tile_size>;
  if (smem > 48 * 1024) {
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         static_cast<int>(smem));
  }

  // Launch
  kernel<<<blocks, threads, smem, stream>>>(
      p.graph.view(),
      rq.codes,
      queries.rot_queries,
      queries.factors,
      result.frontier,
      result.visited,
      result.visited_counts,
      p.query_vectors,
      p.medoid,
      p.k,
      p.beam_width,
      p.limit,
      p.get_visited,
      p.max_result_size);

  cudaError_t err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    std::cerr << "RaBitQ beam search kernel launch failed: " << cudaGetErrorString(err) << std::endl;
  }

  return result;
}

}  // namespace jasper
