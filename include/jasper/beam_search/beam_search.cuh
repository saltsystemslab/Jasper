#pragma once

#include <cub/cub.cuh>
#include <fstream>
#include <iostream>
#include <vector>

#include <cooperative_groups.h>

#include "jasper/index/vector.cuh"
#include "jasper/index/graph.cuh"
#include "jasper/beam_search/config.cuh"
#include "jasper/beam_search/entry.cuh"
#include "jasper/beam_search/device_kernels.cuh"
#include "jasper/distance/distance.cuh"

#include "assert.h"
#include "stdio.h"

namespace cg = cooperative_groups;

namespace jasper {

// main search kernel
template <typename GRAPH_CFG, uint32_t BLOCK_SIZE, distance_func DISTANCE_FUNC,
          uint32_t MAX_SEARCH_WIDTH, uint32_t TILE_SIZE = 4>
__global__ void beam_search_kernel(
    typename graph<GRAPH_CFG>::device_view graph,
    thrust::pair<typename GRAPH_CFG::index_t, typename GRAPH_CFG::distance_t> *frontier_results,
    thrust::pair<typename GRAPH_CFG::index_t, typename GRAPH_CFG::distance_t> *visited_results,
    uint32_t *visited_counts,
    vector_view<typename GRAPH_CFG::data_t> query_vectors,
    bool use_range,
    uint32_t query_start,
    uint32_t query_end,
    typename GRAPH_CFG::index_t medoid,
    uint32_t k,
    uint32_t beam_width,
    uint32_t limit,
    bool get_visited,
    uint32_t max_result_size) {

  #ifdef _CLK_BREAKDOWN
    std::uint64_t clk_init = 0;
    std::uint64_t clk_1st_opulate_distance = 0;
    std::uint64_t clk_choose_new_frontier = 0;
    std::uint64_t clk_insert_hash = 0;
    std::uint64_t clk_add_frontier_out = 0;
    std::uint64_t clk_populate_distances = 0;
    std::uint64_t clk_merge_sort = 0;
    std::uint64_t clk_clip_k = 0;
    std::uint64_t clk_dedup = 0;
    std::uint64_t clk_filter_frontier = 0;
    std::uint64_t clk_start;
    #define _CLK_START() clk_start = clock64();
    #define _CLK_REC(V) V += clock64() - clk_start;
  #else
    #define _CLK_START();
    #define _CLK_REC(V);
  #endif

  using INDEX_T = typename GRAPH_CFG::index_t;
  using DATA_T = typename GRAPH_CFG::data_t;
  using DISTANCE_T = typename GRAPH_CFG::distance_t;
  using EDGE_LIST_T = typename GRAPH_CFG::edge_list_t;

  // get the query vector for this block
  const auto query_id = blockIdx.x;
  uint32_t visited_counter = 0;

  assert(beam_width + 64 <= MAX_SEARCH_WIDTH);

  // allocate shared memory
  _CLK_START();
  const uint32_t result_buffer_size = beam_width + 64;
  extern __shared__ __align__(128) uint32_t smem[];
  auto *__restrict__ result_buffer = reinterpret_cast<ENTRY_T *>(smem);
  auto *__restrict__ result_buffer_count = reinterpret_cast<uint32_t *>(result_buffer + result_buffer_size);

  // Align smem_query_vec to 16 bytes
  uintptr_t query_vec_offset = reinterpret_cast<uintptr_t>(result_buffer_count + 1);
  query_vec_offset = (query_vec_offset + 15) & ~uintptr_t(15);
  auto *__restrict__ smem_query_vec = reinterpret_cast<DATA_T *>(query_vec_offset);

  // Load query vector to shared memory (including zero padding for aligned loads)
  DATA_T *query_vec;
  if (use_range) {
    query_vec = graph.get_vector(query_start + query_id);
  } else {
    query_vec = query_vectors[query_id];
  }
  for (uint i = threadIdx.x; i < graph.get_padded_dim(); i += blockDim.x) {
    smem_query_vec[i] = query_vec[i];
  }
  
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
  // TODO: we only put the mediod in there right now,
  //       maybe it is more efficient to start with multiple points.
  if (threadIdx.x == 0) {
    result_buffer[0] = set_index(empty_entry(), medoid);
    result_buffer_count[0] = 1;
  }
  __syncthreads();
  _CLK_REC(clk_init);

  _CLK_START();
  populate_distances<GRAPH_CFG, TILE_SIZE, DISTANCE_FUNC>(smem_query_vec, graph,
                                       result_buffer, result_buffer_count, 0);
  _CLK_REC(clk_1st_opulate_distance);

  // loop
  uint32_t offset;
  uint32_t loop_count = 0;
  while (loop_count <= limit) {

    loop_count += 1;
    // choose a new frontier to explore
    _CLK_START();
    auto [frontierIdx, found] =
        choose_new_frontier<INDEX_T, DISTANCE_T, BLOCK_SIZE, MAX_SEARCH_WIDTH>(
          result_buffer, result_buffer_count);
    if (!found) {
      break;  // we converged
    };
    _CLK_REC(clk_choose_new_frontier);

    _CLK_START();
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
    _CLK_REC(clk_insert_hash);
    
    // Add frontier to visited list
    // we do not add the first point
    if (threadIdx.x == 0 && get_visited) {
      visited_results[query_id * max_result_size + visited_counter].first =
          get_index(result_buffer[frontierIdx]);
      visited_results[query_id * max_result_size + visited_counter].second =
          get_distance(result_buffer[frontierIdx]);
      // assert(visited_counter < max_result_size);
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
    _CLK_START();
    add_frontier_out<GRAPH_CFG>(graph, result_buffer, result_buffer_count,
                     get_index(result_buffer[frontierIdx]), k, beam_width);
    __syncthreads();
    _CLK_REC(clk_add_frontier_out);

    // populate distance
    _CLK_START();
    populate_distances<GRAPH_CFG, TILE_SIZE, DISTANCE_FUNC>(smem_query_vec, graph,
                                         result_buffer, result_buffer_count,
                                         offset);
    __syncthreads();
    _CLK_REC(clk_populate_distances);

    // sort the result buffer
    _CLK_START();
    merge_sort<INDEX_T, DISTANCE_T, BLOCK_SIZE, MAX_SEARCH_WIDTH, BlockMergeSortT>(
        result_buffer, result_buffer_count, temp_storage.sort_storage);
    _CLK_REC(clk_merge_sort);

    _CLK_START();
    dedup_results<INDEX_T, DISTANCE_T, BLOCK_SIZE, MAX_SEARCH_WIDTH, BlockScanT>(
        result_buffer, result_buffer_count, temp_storage.scan_storage);
    _CLK_REC(clk_dedup);

    _CLK_START();
    clip_k(result_buffer_count, beam_width);
    _CLK_REC(clk_clip_k);
  }

  // copy to the external result, skipping any soft-deleted vertices so they
  // never surface in search results (matches JasperGPUANNS beam search).
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

  #ifdef _CLK_BREAKDOWN
  if (threadIdx.x == 0 && blockIdx.x == 20) {
    printf(
      "%s:%d "
      "query %d thread %d visited=%u\n"
      " - init,                %lu\n"
      " - 1st_distance,        %lu\n"
      " - choose_new_frontier, %lu\n"
      " - mark_visited,        %lu\n"
      " - add_frontier_out,    %lu\n"
      " - populate_distances,  %lu\n"
      " - merge_sort,          %lu\n"
      " - deduplication,       %lu\n"
      " - clip_k,              %lu\n"
      "\n",
      __FILE__,
      __LINE__,
      blockIdx.x,
      threadIdx.x,
      visited_counter,
      clk_init,
      clk_1st_opulate_distance,
      clk_choose_new_frontier,
      clk_insert_hash,
      clk_add_frontier_out,
      clk_populate_distances,
      clk_merge_sort,
      clk_dedup,
      clk_clip_k
    );
  }
  #endif
}

// Get how big the shared memory needs to be
// in bytes.
template <typename INDEX_T, typename DISTANCE_T, typename DATA_T>
__host__ uint32_t get_smem_size(const uint32_t beam_width,
                                const uint32_t block_size,
                                const uint32_t k,
                                const uint32_t padded_dim) {
  uint32_t smem_size = 0;
  smem_size += sizeof(ENTRY_T) * (beam_width + 64);  // result_buffer
  smem_size += sizeof(uint32_t);                      // result_buffer_count
  smem_size = (smem_size + 15) & ~15u;                // align to 16 bytes
  smem_size += sizeof(DATA_T) * padded_dim;           // smem_query_vec
  return smem_size;
}

template <typename Cfg>
__host__ beam_search_result<typename Cfg::graph_cfg_t>
beam_search(const beam_search_params<Cfg>& p, cudaStream_t stream = 0) {

  using entry_t = typename Cfg::entry_t;

  uint32_t n_query_vectors;
  if (p.use_range) {
    n_query_vectors = p.query_end - p.query_start;
  } else {
    n_query_vectors = p.query_vectors.n_vectors;
  }

  dim3 threads(Cfg::block_size, 1, 1);
  dim3 blocks(n_query_vectors, 1, 1);
  uint32_t smem = get_smem_size<typename Cfg::index_t, typename Cfg::distance_t, typename Cfg::data_t>(
      p.beam_width, Cfg::block_size, p.k, p.graph.get_padded_dim());

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

  auto graph_device_view = p.graph.view();

  // Launch
  beam_search_kernel<
      typename Cfg::graph_cfg_t,
      Cfg::block_size, Cfg::dist_func,
      Cfg::max_search_width,
      Cfg::tile_size>
    <<<blocks, threads, smem, stream>>>(
        graph_device_view,
        result.frontier,
        result.visited,
        result.visited_counts,
        p.query_vectors,
        p.use_range,
        p.query_start,
        p.query_end,
        p.medoid,
        p.k,
        p.beam_width,
        p.limit,
        p.get_visited,
        p.max_result_size);

  cudaError_t err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    std::cerr << "Beam search kernel launch failed: " << cudaGetErrorString(err) << std::endl;
  }

  return result;
}

// beam search with pre allocated buffer for result
template <typename Cfg>
__host__ void beam_search(
  const beam_search_params<Cfg>& p,
  beam_search_result<typename Cfg::graph_cfg_t> result, // pre-allocated.
  cudaStream_t stream = 0
) {

  using entry_t = typename Cfg::entry_t;

  uint32_t n_query_vectors;
  if (p.use_range) {
    n_query_vectors = p.query_end - p.query_start;
  } else {
    n_query_vectors = p.query_vectors.n_vectors;
  }

  dim3 threads(Cfg::block_size, 1, 1);
  dim3 blocks(n_query_vectors, 1, 1);
  uint32_t smem = get_smem_size<typename Cfg::index_t, typename Cfg::distance_t, typename Cfg::data_t>(
      p.beam_width, Cfg::block_size, p.k, p.graph.dim);

  auto graph_device_view = p.graph.view();

  // Launch
  beam_search_kernel<
      typename Cfg::graph_cfg_t,
      Cfg::block_size, Cfg::dist_func,
      Cfg::max_search_width,
      Cfg::tile_size>
    <<<blocks, threads, smem, stream>>>(
        graph_device_view,
        result.frontier,
        result.visited,
        result.visited_counts,
        p.query_vectors,
        p.use_range,
        p.query_start,
        p.query_end,
        p.medoid,
        p.k,
        p.beam_width,
        p.limit,
        p.get_visited,
        p.max_result_size);

  cudaError_t err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    std::cerr << "Beam search kernel launch failed: " << cudaGetErrorString(err) << std::endl;
  }
}

}  // namespace jasper