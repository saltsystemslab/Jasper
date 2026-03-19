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
#include "jasper/distance/distance.cuh"

#include "assert.h"
#include "stdio.h"

namespace cg = cooperative_groups;

namespace jasper {

template <typename DATA_T, typename INDEX_T,
          typename DISTANCE_T, uint32_t TILE_SIZE,
          distance_func DIST_FUNC>
__device__ void populate_distances(
    DATA_T *query_vec,
    vector_view<DATA_T> data_vectors,
    ENTRY_T *result_buffer,
    uint32_t *result_buffer_count, 
    uint32_t offset) {
  auto thread_block = cg::this_thread_block();
  cg::thread_block_tile<TILE_SIZE> my_tile =
      cg::tiled_partition<TILE_SIZE>(thread_block);
  uint64_t tid = my_tile.meta_group_rank();
  constexpr INDEX_T INVALID_INDEX = std::numeric_limits<INDEX_T>::max();
  uint32_t padded_dim = data_vectors.padded_dim;

  uint32_t count = result_buffer_count[0];
  for (unsigned i = tid + offset; i < count; i += my_tile.meta_group_size()) {
    auto dest = get_index(result_buffer[i]);
    if (dest != INVALID_INDEX) {
      float dist = compute_distance<DIST_FUNC, DATA_T, DISTANCE_T, TILE_SIZE>(
        query_vec, data_vectors[dest], padded_dim, my_tile);
      if (my_tile.thread_rank() == 0) {
        result_buffer[i] = set_distance(result_buffer[i], dist);
      }
    }
  }
  __threadfence();
  __syncthreads();
}

// select and return the new frontier
template <typename INDEX_T, typename DISTANCE_T, uint32_t BLOCK_SIZE, uint32_t MAX_SEARCH_WIDTH>
__device__ thrust::pair<INDEX_T, bool> choose_new_frontier(
    ENTRY_T *result_buffer,
    uint32_t *result_buffer_count) {

  constexpr uint32_t ELEMENTS_PER_THREAD = (MAX_SEARCH_WIDTH - 1) / BLOCK_SIZE + 1;

  // first_index is the index we want to find that we haven't traversed yet.
  __shared__ INDEX_T first_index;
  if (threadIdx.x == 0) first_index = ~0u;
  __syncthreads();

  for (uint i=0; i<ELEMENTS_PER_THREAD; i++) {
    INDEX_T visited;
    INDEX_T index_to_visit = i * BLOCK_SIZE + threadIdx.x;
    if (index_to_visit < *result_buffer_count) {
      visited = static_cast<uint32_t>(get_visited(result_buffer[index_to_visit]));
    } else {
      visited = 1;
    }
    unsigned mask = __ballot_sync(0xffffffff, !visited);
    if (mask != 0) {
      int lane_id = threadIdx.x % warpSize;
      int first_lane = __ffs(mask) - 1;
      if (lane_id == first_lane) {
        atomicMin(&first_index, index_to_visit);
      }
    }
  }
  __syncthreads();
  
  return {first_index, first_index != ~0u};
}

// adding the newly selected frontier's neighbors to the frontier list
template <typename INDEX_T, typename DISTANCE_T, typename EDGE_LIST_T>
__device__ void add_frontier_out(
    const EDGE_LIST_T * __restrict__ graph, const uint8_t * __restrict__ edge_count,
    ENTRY_T *result_buffer,
    uint32_t *result_buffer_count, const INDEX_T & frontier, const uint32_t & k,
    const uint32_t & beam_width) {

  uint8_t n_edges = edge_count[frontier];
  uint32_t offset = result_buffer_count[0];
  __syncthreads();

  const uint4* l_ptr = reinterpret_cast<const uint4*>(&graph[frontier].edges);

  for (uint i = threadIdx.x*4; i < n_edges; i += blockDim.x*4){
    uint4 loaded_edges = l_ptr[i/4];
    const uint32_t * loaded_edges_ptr = (uint32_t *) &loaded_edges;
    for (uint j = 0; j < 4; j++){
      result_buffer[offset+i+j] = set_index(empty_entry(), loaded_edges_ptr[j]);
    }
  }

  __syncthreads();
  if (threadIdx.x == 0) {
    result_buffer_count[0] += n_edges;
  }
  __syncthreads();
}

// Custom comparison for {index, distance} pairs
struct CustomPairLess {
  __device__ __forceinline__ bool operator()(
      const ENTRY_T &a,
      const ENTRY_T &b) {
    float da = __uint_as_float(a & 0xFFFFFFFFu);
    float db = __uint_as_float(b & 0xFFFFFFFFu);

    // distance comparison
    uint32_t dist_lt = static_cast<uint32_t>(da < db);
    uint32_t dist_eq = static_cast<uint32_t>(da == db);

    // index comparison. (31 bits)
    uint32_t ia = static_cast<uint32_t>((a >> 32) & 0x7FFFFFFFul);
    uint32_t ib = static_cast<uint32_t>((b >> 32) & 0x7FFFFFFFul);
    uint32_t idx_lt = static_cast<uint32_t>(ia < ib);

    // final result: (dist_lt) OR (dist_eq AND idx_lt)
    return (dist_lt | (dist_eq & idx_lt)) != 0;
  }
};

// sort the result
template <typename INDEX_T, typename DISTANCE_T, uint32_t BLOCK_SIZE,
          uint32_t MAX_SEARCH_WIDTH, typename BlockMergeSortT>
__device__ void merge_sort(ENTRY_T *result_buffer,
                           uint32_t *result_buffer_count,
                           typename BlockMergeSortT::TempStorage &temp_storage) {
  uint32_t count = result_buffer_count[0];
  constexpr uint32_t ELEMENTS_PER_THREAD = (MAX_SEARCH_WIDTH - 1) / BLOCK_SIZE + 1;

  ENTRY_T thread_item[ELEMENTS_PER_THREAD];
#pragma unroll
  for (unsigned i = 0; i < ELEMENTS_PER_THREAD; i++) {
    uint32_t element_id = threadIdx.x * ELEMENTS_PER_THREAD + i;
    if (element_id < count) {
      thread_item[i] = result_buffer[element_id];
    } else {
      thread_item[i] = empty_entry();
    }
  }

  // sort by distance
  BlockMergeSortT(temp_storage).Sort(thread_item, CustomPairLess());

#pragma unroll
  for (unsigned i = 0; i < ELEMENTS_PER_THREAD; i++) {
    uint32_t element_id = threadIdx.x * ELEMENTS_PER_THREAD + i;
    if (element_id < count) {
      result_buffer[element_id] = thread_item[i];
    }
  }

  __syncthreads();
}

// clip the search results to beam_width
__device__ void clip_k(uint32_t* result_buffer_count, const uint32_t & beam_width) {
  if (threadIdx.x == 0) {
    result_buffer_count[0] = min(result_buffer_count[0], beam_width);
  }
  __syncthreads();
}

// deduplicate the frontier list
// assume the list is already sorted
template <typename INDEX_T, typename DISTANCE_T, uint32_t BLOCK_SIZE,
          uint32_t MAX_SEARCH_WIDTH, typename BlockScanT>
__device__ void dedup_results(ENTRY_T *result_buffer,
                              uint32_t *result_buffer_count,
                              typename BlockScanT::TempStorage &temp_storage) {
  constexpr uint32_t ELEMENTS_PER_THREAD =
      (MAX_SEARCH_WIDTH - 1) / BLOCK_SIZE + 1;

  uint32_t count = result_buffer_count[0];

  ENTRY_T this_entry[ELEMENTS_PER_THREAD];
  bool is_unique[ELEMENTS_PER_THREAD];

#pragma unroll
  for (unsigned i = 0; i < ELEMENTS_PER_THREAD; i++) {
    is_unique[i] = true;
    uint32_t element_id = threadIdx.x * ELEMENTS_PER_THREAD + i;
    if (element_id < count) {
      this_entry[i] = result_buffer[element_id];
      INDEX_T this_index = get_index(this_entry[i]);
      INDEX_T last_index = get_index(result_buffer[element_id - 1]);
      if (element_id > 0 && (this_index == last_index)) {
        is_unique[i] = false;

        #if MEASURE_WASTED_CALCS
        atomicAdd(&wasted_distance_calcs, 1);
        #endif

      }
    } else {
      is_unique[i] = false;
    }
  }
  __syncthreads();

  uint32_t thread_data[ELEMENTS_PER_THREAD];
#pragma unroll
  for (unsigned i = 0; i < ELEMENTS_PER_THREAD; i++) {
    thread_data[i] = is_unique[i] ? 1 : 0;
  }

  BlockScanT(temp_storage).ExclusiveSum(thread_data, thread_data);

#pragma unroll
  for (unsigned i = 0; i < ELEMENTS_PER_THREAD; i++) {
    if (is_unique[i]) {
      result_buffer[thread_data[i]] = this_entry[i];
    }
  }
  if (threadIdx.x == blockDim.x - 1) {
    // result_buffer_count[0] = thread_data[ELEMENTS_PER_THREAD - 1];
    result_buffer_count[0] = thread_data[ELEMENTS_PER_THREAD - 1]
                           + (is_unique[ELEMENTS_PER_THREAD - 1] ? 1 : 0);
  }
  __syncthreads();
}

// main search kernel
template <typename INDEX_T, typename DATA_T,
          typename DISTANCE_T, typename EDGE_LIST_T, uint32_t BLOCK_SIZE,
          distance_func DISTANCE_FUNC,
          uint32_t MAX_SEARCH_WIDTH, bool GET_VISITED, uint32_t TILE_SIZE = 4, uint32_t MAX_RESULT_SIZE = 1024>
__global__ void beam_search_single_kernel(
    EDGE_LIST_T *graph, 
    uint8_t *edge_count,
    thrust::pair<INDEX_T, DISTANCE_T> *frontier_results,
    thrust::pair<INDEX_T, DISTANCE_T> *visited_results,
    uint32_t *visited_counts, 
    vector_view<DATA_T> data_vectors,
    vector_view<DATA_T> query_vectors,
    INDEX_T medoid, 
    uint32_t k, 
    uint32_t beam_width,
    uint32_t limit) {

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
  uint32_t padded_dim = query_vectors.padded_dim;
  DATA_T *query_vec = query_vectors[query_id];
  for (uint i = threadIdx.x; i < padded_dim; i += blockDim.x) {
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
  populate_distances<DATA_T, INDEX_T, DISTANCE_T, TILE_SIZE,
                     DISTANCE_FUNC>(smem_query_vec, data_vectors,
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
    if (threadIdx.x == 0 && GET_VISITED) {
      visited_results[query_id * MAX_RESULT_SIZE + visited_counter].first =
          get_index(result_buffer[frontierIdx]);
      visited_results[query_id * MAX_RESULT_SIZE+ visited_counter].second =
          get_distance(result_buffer[frontierIdx]);
      // assert(visited_counter < MAX_RESULT_SIZE);
    }
    visited_counter += 1;
    // break if we exceeded the result size for visited list.
    if (visited_counter == MAX_RESULT_SIZE) break;
    __syncthreads();

    // record offset so that we know where to start calculating distances
    // (all the previous ones are calculated)
    offset = result_buffer_count[0];
    __syncthreads();

    // Add frontier's neighbor to results
    _CLK_START();
    add_frontier_out<INDEX_T, DISTANCE_T, EDGE_LIST_T>(graph, edge_count, result_buffer, result_buffer_count,
                     get_index(result_buffer[frontierIdx]), k, beam_width);
    __syncthreads();
    _CLK_REC(clk_add_frontier_out);

    // populate distance
    _CLK_START();
    populate_distances<DATA_T, INDEX_T, DISTANCE_T, TILE_SIZE,
                       DISTANCE_FUNC>(smem_query_vec, data_vectors,
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

  // copy to the external result
  for (uint i=threadIdx.x; i<k; i += blockDim.x) {
    frontier_results[query_id * k + i].first =
        get_index(result_buffer[i]);
    frontier_results[query_id * k + i].second =
        get_distance(result_buffer[i]);
  }
  if (threadIdx.x == 0 && GET_VISITED) {
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
beam_search(const beam_search_params<Cfg>& p) {

  using entry_t = typename Cfg::entry_t;

  dim3 threads(Cfg::block_size, 1, 1);
  dim3 blocks(p.query_vectors.n_vectors, 1, 1);
  uint32_t smem = get_smem_size<typename Cfg::index_t, typename Cfg::distance_t, typename Cfg::data_t>(
      p.beam_width, Cfg::block_size, p.k, p.data_vectors.padded_dim);

  beam_search_result<typename Cfg::graph_cfg_t> result{};

  // Allocate frontier
  cudaMalloc(&result.frontier, sizeof(entry_t) * p.query_vectors.n_vectors * p.k);
  // Allocate visited
  if constexpr (Cfg::get_visited) {
    cudaMalloc(&result.visited,
               sizeof(entry_t) * p.query_vectors.n_vectors * Cfg::max_result_size);
    cudaMalloc(&result.visited_counts,
               sizeof(uint32_t) * p.query_vectors.n_vectors);
  }

  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    std::cerr << "Beam search malloc memory failed: " << cudaGetErrorString(err) << std::endl;
  }

  // Launch
  beam_search_single_kernel<
      typename Cfg::index_t, typename Cfg::data_t, 
      typename Cfg::distance_t, typename Cfg::graph_cfg_t::edge_list_t,
      Cfg::block_size, Cfg::dist_func,
      Cfg::max_search_width, Cfg::get_visited,
      Cfg::tile_size, Cfg::max_result_size>
    <<<blocks, threads, smem>>>(
        p.graph.edges, p.graph.edge_counts,
        result.frontier, result.visited, result.visited_counts,
        p.data_vectors,
        p.query_vectors,
        p.medoid, p.k, p.beam_width, p.limit);

  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    std::cerr << "Beam search kernel launch failed: " << cudaGetErrorString(err) << std::endl;
  }

  return result;
}

// beam search with pre allocated buffer for result
template <typename Cfg>
__host__ void beam_search(
  const beam_search_params<Cfg>& p,
  beam_search_result<typename Cfg::graph_cfg_t> result // pre-allocated.
) {

  using entry_t = typename Cfg::entry_t;

  dim3 threads(Cfg::block_size, 1, 1);
  dim3 blocks(p.query_vectors.n_vectors, 1, 1);
  uint32_t smem = get_smem_size<typename Cfg::index_t, typename Cfg::distance_t, typename Cfg::data_t>(
      p.beam_width, Cfg::block_size, p.k, p.data_vectors.dim);

  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    std::cerr << "Beam search malloc memory failed: " << cudaGetErrorString(err) << std::endl;
  }

  // Launch
  beam_search_single_kernel<
      typename Cfg::index_t, typename Cfg::data_t, 
      typename Cfg::distance_t, typename Cfg::graph_cfg_t::edge_list_t,
      Cfg::block_size, Cfg::dist_func,
      Cfg::max_search_width, Cfg::get_visited,
      Cfg::tile_size, Cfg::max_result_size>
    <<<blocks, threads, smem>>>(
        p.graph.edges, p.graph.edge_counts,
        result.frontier, result.visited, result.visited_counts,
        p.data_vectors,
        p.query_vectors,
        p.medoid, p.k, p.beam_width, p.limit);

  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    std::cerr << "Beam search kernel launch failed: " << cudaGetErrorString(err) << std::endl;
  }
}

}  // namespace jasper