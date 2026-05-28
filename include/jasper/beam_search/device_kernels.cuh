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

template <typename GRAPH_CFG, uint32_t TILE_SIZE, distance_func DIST_FUNC>
__device__ void populate_distances(
    typename GRAPH_CFG::data_t *query_vec,
    typename graph<GRAPH_CFG>::device_view& graph,
    ENTRY_T *result_buffer,
    uint32_t *result_buffer_count, 
    uint32_t offset) {

  using INDEX_T = typename GRAPH_CFG::index_t;
  using DATA_T = typename GRAPH_CFG::data_t;
  using DISTANCE_T = typename GRAPH_CFG::distance_t;
  using EDGE_LIST_T = typename GRAPH_CFG::edge_list_t;
  
  auto thread_block = cg::this_thread_block();
  cg::thread_block_tile<TILE_SIZE> my_tile =
      cg::tiled_partition<TILE_SIZE>(thread_block);
  uint64_t tid = my_tile.meta_group_rank();
  // 0x7FFFFFFF matches the index stored by empty_entry() (31-bit max after masking)
  constexpr INDEX_T INVALID_INDEX = static_cast<INDEX_T>(0x7FFFFFFFu);
  uint32_t padded_dim = graph.get_padded_dim();

  uint32_t count = result_buffer_count[0];
  for (unsigned i = tid + offset; i < count; i += my_tile.meta_group_size()) {
    auto dest = get_index(result_buffer[i]);
    if (dest != INVALID_INDEX) {
      float dist = compute_distance<DIST_FUNC, DATA_T, DISTANCE_T, TILE_SIZE>(
        query_vec, graph.get_vector(dest), padded_dim, my_tile);
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
template <typename GRAPH_CFG>
__device__ void add_frontier_out(
    typename graph<GRAPH_CFG>::device_view& graph,
    ENTRY_T *result_buffer,
    uint32_t *result_buffer_count,
    const typename GRAPH_CFG::index_t & frontier,
    const uint32_t & k,
    const uint32_t & beam_width) {

  using INDEX_T = typename GRAPH_CFG::index_t;

  const uint8_t  n_edges = graph.get_edge_count(frontier);
  const uint32_t offset  = result_buffer_count[0];
  __syncthreads();

  const INDEX_T* __restrict__ edge_ptr =
      &graph.get_neighbor_list(frontier).edges[0];
  const INDEX_T range_lo = graph.global_offset;
  const INDEX_T range_hi = graph.global_offset + graph.n_vectors;

  for (uint32_t i = threadIdx.x; i < n_edges; i += blockDim.x) {
    const INDEX_T nb = edge_ptr[i];
    if (nb >= range_lo && nb < range_hi) {
      result_buffer[offset + i] = set_index(empty_entry(), nb);
    } else {
      result_buffer[offset + i] = set_visited(empty_entry());
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
  // constexpr INDEX_T  INVALID = static_cast<INDEX_T>(get_index(empty_entry()));
  constexpr INDEX_T INVALID = static_cast<INDEX_T>(0x7FFFFFFFu);

  const uint32_t count = result_buffer_count[0];

  ENTRY_T this_entry[ELEMENTS_PER_THREAD];
  bool    is_unique [ELEMENTS_PER_THREAD];

  // ── 1. One pass over shared memory: load my items into registers ──
  #pragma unroll
  for (unsigned i = 0; i < ELEMENTS_PER_THREAD; ++i) {
    uint32_t element_id = threadIdx.x * ELEMENTS_PER_THREAD + i;
    this_entry[i] = (element_id < count) ? result_buffer[element_id]
                                         : empty_entry();
  }

  // ── 2. Find previous thread's last index without re-reading smem ──
  // My last item's index (or INVALID if out of range).
  const uint32_t my_last_id = threadIdx.x * ELEMENTS_PER_THREAD
                            + ELEMENTS_PER_THREAD - 1;
  const INDEX_T  my_last_index = (my_last_id < count)
      ? static_cast<INDEX_T>(get_index(this_entry[ELEMENTS_PER_THREAD - 1]))
      : INVALID;

  // Intra-warp: previous lane's last index, in one shuffle.
  const INDEX_T prev_in_warp =
      __shfl_up_sync(0xFFFFFFFFu, my_last_index, 1);

  // Inter-warp: one slot per warp.
  constexpr uint32_t N_WARPS = BLOCK_SIZE / 32;
  __shared__ INDEX_T warp_last[N_WARPS > 0 ? N_WARPS : 1];
  const uint32_t lane = threadIdx.x & 31u;
  const uint32_t wid  = threadIdx.x >> 5;
  if (lane == 31) warp_last[wid] = my_last_index;
  __syncthreads();

  INDEX_T prev_index;
  if (threadIdx.x == 0) {
    prev_index = INVALID;                  // no predecessor at all
  } else if (lane == 0) {
    prev_index = warp_last[wid - 1];       // first lane of warp w
  } else {
    prev_index = prev_in_warp;             // lane > 0 within warp
  }

  // ── 3. Mark unique using the register chain ─────────────────────
  #pragma unroll
  for (unsigned i = 0; i < ELEMENTS_PER_THREAD; ++i) {
    uint32_t element_id = threadIdx.x * ELEMENTS_PER_THREAD + i;
    if (element_id < count) {
      INDEX_T this_index = static_cast<INDEX_T>(get_index(this_entry[i]));
      is_unique[i] = (this_index != prev_index);
      prev_index   = this_index;

#if MEASURE_WASTED_CALCS
      if (!is_unique[i]) atomicAdd(&wasted_distance_calcs, 1);
#endif
    } else {
      is_unique[i] = false;
    }
  }

  // ── 4. Exclusive scan over the unique flags ─────────────────────
  uint32_t thread_data[ELEMENTS_PER_THREAD];
  #pragma unroll
  for (unsigned i = 0; i < ELEMENTS_PER_THREAD; ++i) {
    thread_data[i] = is_unique[i] ? 1u : 0u;
  }
  BlockScanT(temp_storage).ExclusiveSum(thread_data, thread_data);

  // ── 5. Scatter unique entries to their compacted slots ──────────
  __syncthreads();   // make sure all reads of result_buffer are done
  #pragma unroll
  for (unsigned i = 0; i < ELEMENTS_PER_THREAD; ++i) {
    if (is_unique[i]) {
      result_buffer[thread_data[i]] = this_entry[i];
    }
  }

  if (threadIdx.x == blockDim.x - 1) {
    result_buffer_count[0] = thread_data[ELEMENTS_PER_THREAD - 1]
                           + (is_unique[ELEMENTS_PER_THREAD - 1] ? 1u : 0u);
  }
  __syncthreads();
}

}