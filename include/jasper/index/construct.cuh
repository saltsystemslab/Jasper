#pragma once

#include <cstdlib>
#include <cstdint>
#include <iostream>

#include <cooperative_groups.h>

#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/binary_search.h>
#include <thrust/iterator/counting_iterator.h>

#include "jasper/index/graph.cuh"
#include "jasper/index/vector.cuh"
#include "jasper/beam_search/beam_search.cuh"
#include "jasper/index/utils.cuh"

namespace cg = cooperative_groups;

namespace jasper {

template <typename GRAPH_CONFIG,
          uint32_t BLOCK_SIZE,
          uint32_t TILE_SIZE,
          uint32_t R_SIZE,
          uint32_t L_SIZE>
struct graph_construct_config {
  using graph_cfg_t = GRAPH_CONFIG;
  using graph_t = graph<GRAPH_CONFIG>;
  using index_t       = typename GRAPH_CONFIG::index_t;
  using data_t        = typename GRAPH_CONFIG::data_t;
  using distance_t    = typename GRAPH_CONFIG::distance_t;
  using edge_list_t   = typename GRAPH_CONFIG::edge_list_t;
  using vector_view_t = typename graph_t::vector_view_t;
  using entry_t       = thrust::pair<index_t, distance_t>;

  static constexpr uint32_t block_size = BLOCK_SIZE;
  static constexpr uint32_t tile_size  = TILE_SIZE;

  static constexpr uint32_t R = R_SIZE;
  static constexpr uint32_t L = L_SIZE;

  static_assert(R <= 255, "R must be smaller or equal to 255");
  static_assert(L <= 255, "L must be smaller or equal to 255");

  // this affects how many new results we are anticipating in the
  // robust prune stage.
  static constexpr uint32_t BEAM_SEARCH_LIMIT = 256;

  static constexpr uint32_t beam_search_max_search_width = L + 64;
  // we can hardcap this to a smaller number because most of the result 
  // will be removed during robust pruning step.
  static constexpr uint32_t beam_search_max_result_size  = 256; 
  static_assert(beam_search_max_search_width <= beam_search_max_result_size, 
    "Beam search width must be smaller or equal to max result size.");
};

template <typename CONSTRUCT_GRAPH_CONFIG>
struct graph_construct_params {
  // Vectors (on host initially)
  typename CONSTRUCT_GRAPH_CONFIG::vector_view_t data_vectors;

  // Pruning factor
  float alpha = 1.2;

  // Maximum batch size decides how much vectors do we consider at once.
  // Normally this number should be
  // 1. less or equal to 10% of the total vector.
  // 2. batch size needs to be fit inside device memory.
  uint32_t max_batch_size = 10000;

  // Is the construction on host
  bool on_host = false;
};

template <typename INDEX_T>
struct edge_pair {
  INDEX_T source;
  INDEX_T sink;

  static __host__ __device__ edge_pair sentinel() {
    return {cuda::std::numeric_limits<INDEX_T>::max(),
            cuda::std::numeric_limits<INDEX_T>::max()};
  }

  __host__ __device__ bool is_sentinel() const {
    return source == cuda::std::numeric_limits<INDEX_T>::max();
  }
};

template <typename INDEX_T>
struct semi_sort_compare_edge_pair {
  __device__ bool operator()(const edge_pair<INDEX_T>& a,
                             const edge_pair<INDEX_T>& b) const {
    return a.source < b.source;
  }
};

template <typename INDEX_T>
struct extract_source_from_edge_pair {
  __device__ INDEX_T operator()(const edge_pair<INDEX_T>& e) const {
    return e.source;
  }
};

template <typename GRAPH_CONSTRUCT_CONFIG>
struct graph_construct_workspace {
  using index_t = typename GRAPH_CONSTRUCT_CONFIG::graph_cfg_t::index_t;
  using distance_t = typename GRAPH_CONSTRUCT_CONFIG::graph_cfg_t::distance_t;
  using entry_t = thrust::pair<index_t, distance_t>;

  uint32_t max_batch_size;

  // beam search workspace
  entry_t*  frontier;        // [max_batch_size * L]
  entry_t*  visited;         // [max_batch_size * max_result_size]
  uint32_t* visited_counts;  // [max_batch_size]

  // robust pruning workspace
  entry_t*  prune_candidates;  // [max_batch_size * (max_result_size + R)]
  uint32_t* prune_counts;      // [max_batch_size]

  // reverse edge workspace
  thrust::device_vector<edge_pair<index_t>> reverse_edges; // [batch_size * R]
  thrust::device_vector<index_t> reverse_offsets;          // [batch_size * R + 1]
  edge_pair<index_t>* reverse_edges_ptr;
  index_t* reverse_offsets_ptr;

  static graph_construct_workspace allocate(uint32_t max_batch_size) {
    graph_construct_workspace ws;
    ws.max_batch_size = max_batch_size;
    constexpr uint32_t L = GRAPH_CONSTRUCT_CONFIG::L;
    constexpr uint32_t R = GRAPH_CONSTRUCT_CONFIG::R;
    constexpr uint32_t max_result_size = GRAPH_CONSTRUCT_CONFIG::beam_search_max_result_size;
    constexpr uint32_t max_candidates = max_result_size + R; // default = 256 + 64

    cudaMalloc(&ws.frontier,            sizeof(entry_t)  * max_batch_size * L);
    cudaMalloc(&ws.visited,             sizeof(entry_t)  * max_batch_size * max_result_size);
    cudaMalloc(&ws.visited_counts,      sizeof(uint32_t) * max_batch_size);
    cudaMalloc(&ws.prune_candidates,    sizeof(entry_t)  * max_batch_size * max_candidates);
    cudaMalloc(&ws.prune_counts,        sizeof(uint32_t) * max_batch_size);
    
    ws.reverse_edges.resize(max_batch_size * R);
    ws.reverse_offsets.resize(max_batch_size * R + 1);

    ws.reverse_edges_ptr = thrust::raw_pointer_cast(ws.reverse_edges.data());
    ws.reverse_offsets_ptr = thrust::raw_pointer_cast(ws.reverse_offsets.data());

    return ws;
  }

  void print_space_usage() {
    constexpr uint32_t L = GRAPH_CONSTRUCT_CONFIG::L;
    constexpr uint32_t R = GRAPH_CONSTRUCT_CONFIG::R;
    constexpr uint32_t max_result_size = GRAPH_CONSTRUCT_CONFIG::beam_search_max_result_size;
    constexpr uint32_t max_candidates = max_result_size + R;

    size_t frontier_bytes     = sizeof(entry_t)           * max_batch_size * L;
    size_t visited_bytes      = sizeof(entry_t)           * max_batch_size * max_result_size;
    size_t visited_cnt_bytes  = sizeof(uint32_t)          * max_batch_size;
    size_t prune_cand_bytes   = sizeof(entry_t)           * max_batch_size * max_candidates;
    size_t prune_cnt_bytes    = sizeof(uint32_t)          * max_batch_size;
    size_t rev_edges_bytes    = sizeof(edge_pair<index_t>) * reverse_edges.size();
    size_t rev_offsets_bytes  = sizeof(index_t)            * reverse_offsets.size();

    size_t total = frontier_bytes + visited_bytes + visited_cnt_bytes
                 + prune_cand_bytes + prune_cnt_bytes
                 + rev_edges_bytes + rev_offsets_bytes;

    auto mb = [](size_t b) { return b / (1024.0 * 1024.0); };

    std::printf("[workspace] max_batch_size=%u, R=%u, L=%u\n", max_batch_size, R, L);
    std::printf("  frontier         : %8.2f MB\n", mb(frontier_bytes));
    std::printf("  visited          : %8.2f MB\n", mb(visited_bytes));
    std::printf("  visited_counts   : %8.2f MB\n", mb(visited_cnt_bytes));
    std::printf("  prune_candidates : %8.2f MB\n", mb(prune_cand_bytes));
    std::printf("  prune_counts     : %8.2f MB\n", mb(prune_cnt_bytes));
    std::printf("  reverse_edges    : %8.2f MB\n", mb(rev_edges_bytes));
    std::printf("  reverse_offsets  : %8.2f MB\n", mb(rev_offsets_bytes));
    std::printf("  total            : %8.2f MB\n", mb(total)); 
  }

  void free() {
    cudaFree(frontier);
    cudaFree(visited);
    cudaFree(visited_counts);
    cudaFree(prune_candidates);
    cudaFree(prune_counts);
    
    reverse_edges.clear();
    reverse_edges.shrink_to_fit();
    reverse_offsets.clear();
    reverse_offsets.shrink_to_fit();
  }
};

template <typename CONSTRUCT_GRAPH_CONFIG, uint32_t TILE_SIZE, uint32_t BLOCK_SIZE>
__global__ void merge_candidates_kernel(
  const typename CONSTRUCT_GRAPH_CONFIG::entry_t* __restrict__ visited,
  const uint32_t* __restrict__ visited_counts,
  const typename CONSTRUCT_GRAPH_CONFIG::edge_list_t* __restrict__ edges,
  const uint8_t* __restrict__ edge_counts,
  const typename CONSTRUCT_GRAPH_CONFIG::data_t* __restrict__ all_vectors,
  const typename CONSTRUCT_GRAPH_CONFIG::data_t* __restrict__ query_vectors,
  uint32_t dim,
  uint32_t batch_offset,
  uint32_t batch_size,
  uint32_t max_visited,
  uint32_t max_candidates,
  typename CONSTRUCT_GRAPH_CONFIG::entry_t* __restrict__ out_candidates,
  uint32_t* __restrict__ out_counts
) {
  using index_t    = typename CONSTRUCT_GRAPH_CONFIG::index_t;
  using distance_t = typename CONSTRUCT_GRAPH_CONFIG::distance_t;
  using entry_t    = typename CONSTRUCT_GRAPH_CONFIG::entry_t;
  using data_t     = typename CONSTRUCT_GRAPH_CONFIG::data_t;

  uint32_t bid = blockIdx.x;
  if (bid >= batch_size) return;

  auto thread_block = cg::this_thread_block();
  cg::thread_block_tile<TILE_SIZE> my_tile = cg::tiled_partition<TILE_SIZE>(thread_block);
  uint64_t tid = my_tile.meta_group_rank();
  uint64_t n_tiles = my_tile.meta_group_size();

  uint32_t global_id = batch_offset + bid;
  entry_t* my_out = out_candidates + bid * max_candidates;
  __shared__ uint32_t s_count;
  if (threadIdx.x == 0) s_count = 0;
  __syncthreads();
  
  // invalid entry
  constexpr index_t INVALID_INDEX = std::numeric_limits<index_t>::max();
  constexpr entry_t SENTINEL = {INVALID_INDEX, std::numeric_limits<distance_t>::max()};

  // add existing neighbors (compute distance to query)
  uint8_t n_edges = edge_counts[global_id];
  const data_t* query_vec = query_vectors + static_cast<uint64_t>(bid) * dim;
  for (uint32_t i = tid; i < n_edges; i+=n_tiles) {
    index_t neighbor = edges[global_id].edges[i];
    if (neighbor == INVALID_INDEX) continue;

    const data_t* neighbor_vec = all_vectors + static_cast<uint64_t>(neighbor) * dim;

    distance_t d = compute_distance<CONSTRUCT_GRAPH_CONFIG::graph_cfg_t::dist_func, data_t, distance_t, TILE_SIZE>(
      query_vec, neighbor_vec, dim, my_tile);

    if (my_tile.thread_rank() == 0) {
      uint32_t slot = atomicAdd(&s_count, 1u);
      if (slot < max_candidates)  {
        my_out[slot] = {neighbor, d};
      }
    }
  }
  __syncthreads();

  // add nodes from beam search
  uint32_t n_visited = visited_counts[bid];
  const entry_t* my_visited = visited + bid * max_visited;

  for (uint32_t i = threadIdx.x; i < n_visited; i += blockDim.x) {
    index_t vid = my_visited[i].first;
    if (vid != global_id && vid != INVALID_INDEX) {
      uint32_t slot = atomicAdd(&s_count, 1u);
      if (slot < max_candidates) {
        my_out[slot] = my_visited[i];
      }
    }
  }
  __syncthreads();

  uint32_t count = min(s_count, max_candidates);

  // sort with cub
  constexpr uint32_t MAX_CANDS = CONSTRUCT_GRAPH_CONFIG::beam_search_max_result_size + CONSTRUCT_GRAPH_CONFIG::R;
  constexpr uint32_t ITEMS_PER_THREAD = (MAX_CANDS + BLOCK_SIZE - 1) / BLOCK_SIZE;
  using BlockMergeSortT = cub::BlockMergeSort<entry_t, BLOCK_SIZE, ITEMS_PER_THREAD>;
  __shared__ typename BlockMergeSortT::TempStorage sort_storage;

  entry_t items[ITEMS_PER_THREAD];
  #pragma unroll
  for (uint32_t i = 0; i < ITEMS_PER_THREAD; i++) {
    uint32_t idx = threadIdx.x * ITEMS_PER_THREAD + i;
    items[i] = (idx < count) ? my_out[idx] : SENTINEL;
  }

  struct EntryLess {
    __device__ __forceinline__ bool operator()(
        const entry_t& a, const entry_t& b) const {
      if (a.second != b.second) return a.second < b.second;
      return a.first < b.first;
    }
  };
  BlockMergeSortT(sort_storage).Sort(items, EntryLess());

  #pragma unroll
  for (uint32_t i = 0; i < ITEMS_PER_THREAD; i++) {
    uint32_t idx = threadIdx.x * ITEMS_PER_THREAD + i;
    if (idx < count) {
      my_out[idx] = items[i];
    }
  }

  if (threadIdx.x == 0) {
    out_counts[bid] = count;
  }
}

template <typename CONSTRUCT_GRAPH_CONFIG, uint32_t TILE_SIZE>
__global__ void robust_prune_kernel(
  const typename CONSTRUCT_GRAPH_CONFIG::entry_t* __restrict__ candidates, // [batch_size * max_candidates]
  const uint32_t* __restrict__ candidate_counts,                  // [batch_size]
  typename CONSTRUCT_GRAPH_CONFIG::edge_list_t* __restrict__ edges,        // graph edges
  uint8_t* __restrict__ edge_counts,                              // graph edge counts
  const typename CONSTRUCT_GRAPH_CONFIG::data_t* __restrict__ all_vectors, // flat [n_vectors * dim]
  uint32_t dim,
  uint32_t batch_offset,    // global index of first vector in this batch
  uint32_t batch_size,
  uint32_t max_candidates,
  float alpha
){
  using index_t     = typename CONSTRUCT_GRAPH_CONFIG::index_t;
  using distance_t  = typename CONSTRUCT_GRAPH_CONFIG::distance_t;
  using entry_t     = typename CONSTRUCT_GRAPH_CONFIG::entry_t;
  using data_t      = typename CONSTRUCT_GRAPH_CONFIG::data_t;
  using edge_list_t = typename CONSTRUCT_GRAPH_CONFIG::edge_list_t;
  constexpr uint32_t R = CONSTRUCT_GRAPH_CONFIG::R;
  constexpr index_t INVALID_INDEX = std::numeric_limits<index_t>::max();

  uint32_t bid = blockIdx.x;
  if (bid >= batch_size) return;

  auto thread_block = cg::this_thread_block();
  auto my_tile = cg::tiled_partition<TILE_SIZE>(thread_block);
  uint64_t tile_id = my_tile.meta_group_rank();
  uint64_t n_tiles = my_tile.meta_group_size();

  uint32_t global_id = batch_offset + bid;
  uint32_t n_cands = candidate_counts[bid];
  const entry_t* my_cands = candidates + bid * max_candidates;

  // Shared state: all tiles in this block cooperate on one vector
  __shared__ index_t    s_selected[R];
  __shared__ uint32_t   s_n_selected;
  __shared__ bool       s_pruned;

  if (threadIdx.x == 0) s_n_selected = 0;
  __syncthreads();

  for (uint32_t i = 0; i < n_cands; i++) {
    // Early exit if we have enough neighbors
    if (s_n_selected >= R) break;

    index_t cand_id = my_cands[i].first;
    distance_t cand_dist = my_cands[i].second;

    if (cand_id == global_id || cand_id == INVALID_INDEX) continue;

    const data_t* cand_vec = all_vectors + static_cast<uint64_t>(cand_id) * dim;

    // Reset prune flag
    if (threadIdx.x == 0) s_pruned = false;
    __syncthreads();

    // Each tile checks one already-selected neighbor in parallel.
    // Tiles round-robin over selected neighbors.
    uint32_t n_sel = s_n_selected;
    for (uint32_t j = tile_id; j < n_sel; j += n_tiles) {
      const data_t* sel_vec = all_vectors + static_cast<uint64_t>(s_selected[j]) * dim;

      // Tile-parallel distance: each thread in tile handles a chunk of dims
      distance_t d = compute_distance<CONSTRUCT_GRAPH_CONFIG::graph_cfg_t::dist_func, data_t, distance_t, TILE_SIZE>(
        cand_vec, sel_vec, dim, my_tile);

      // Tile leader checks the prune condition
      if (my_tile.thread_rank() == 0) {
        if (d * alpha < cand_dist) {
          s_pruned = true;
        }
      }
    }
    __syncthreads();

    // If not pruned, add to selected set
    if (!s_pruned) {
      if (threadIdx.x == 0) {
        uint32_t slot = s_n_selected;
        if (slot < R) {
          s_selected[slot] = cand_id;
          s_n_selected = slot + 1;
        }
      }
    }
    __syncthreads();
  }

  // Write to graph — one thread writes, all threads read shared state
  uint32_t final_count = s_n_selected;
  for (uint32_t i = threadIdx.x; i < final_count; i += blockDim.x) {
    edges[global_id].edges[i] = s_selected[i];
  }
  if (threadIdx.x == 0) {
    edge_counts[global_id] = static_cast<uint8_t>(final_count);
  }
}

template <typename CONSTRUCT_GRAPH_CONFIG>
__global__ void fill_reverse_edges(
  const typename CONSTRUCT_GRAPH_CONFIG::edge_list_t* __restrict__ graph_edges,
  const uint8_t* __restrict__ edge_counts,
  edge_pair<typename CONSTRUCT_GRAPH_CONFIG::index_t>* reverse_edges,
  uint32_t batch_size,
  uint32_t batch_offset,
  uint32_t max_edges_per_node // R
) {
  using index_t = typename CONSTRUCT_GRAPH_CONFIG::index_t;

  // one thread per slot in [batch_size * max_edges_per_node]
  uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= (uint64_t)batch_size * max_edges_per_node) return;

  uint32_t local_id = tid / max_edges_per_node;
  uint32_t edge_idx = tid % max_edges_per_node;
  index_t  global_id = local_id + batch_offset;

  if (edge_idx < edge_counts[global_id]) {
    index_t neighbor = graph_edges[global_id].edges[edge_idx];
    // edge in graph:   global_id -> neighbor
    // reversed:        source=neighbor, sink=global_id
    reverse_edges[tid] = {neighbor, global_id};
  } else {
    reverse_edges[tid] = edge_pair<index_t>::sentinel();
  }
}

template <typename CONSTRUCT_GRAPH_CONFIG, uint32_t TILE_SIZE>
__global__ void process_reverse_edges_kernel(
  const edge_pair<typename CONSTRUCT_GRAPH_CONFIG::index_t>* __restrict__ reverse_edges,
  const typename CONSTRUCT_GRAPH_CONFIG::index_t* __restrict__ reverse_offsets,
  typename CONSTRUCT_GRAPH_CONFIG::edge_list_t* __restrict__ graph_edges,
  uint8_t* __restrict__ edge_counts,
  const typename CONSTRUCT_GRAPH_CONFIG::data_t* __restrict__ all_vectors,
  uint32_t dim,
  typename CONSTRUCT_GRAPH_CONFIG::index_t num_vertices,
  float alpha
) {
  using index_t    = typename CONSTRUCT_GRAPH_CONFIG::index_t;
  using distance_t = typename CONSTRUCT_GRAPH_CONFIG::distance_t;
  using entry_t    = typename CONSTRUCT_GRAPH_CONFIG::entry_t;
  using data_t     = typename CONSTRUCT_GRAPH_CONFIG::data_t;

  constexpr uint32_t R = CONSTRUCT_GRAPH_CONFIG::R;
  constexpr uint32_t BLOCK_SIZE = CONSTRUCT_GRAPH_CONFIG::block_size;
  constexpr uint32_t MAX_CANDS = CONSTRUCT_GRAPH_CONFIG::beam_search_max_result_size + R;
  constexpr uint32_t ITEMS_PER_THREAD = (MAX_CANDS + BLOCK_SIZE - 1) / BLOCK_SIZE;
  constexpr index_t INVALID_INDEX = std::numeric_limits<index_t>::max();
  constexpr entry_t SENTINEL = {INVALID_INDEX, std::numeric_limits<distance_t>::max()};

  // merge sort compare
  struct EntryLess {
    __device__ __forceinline__ bool operator()(
        const entry_t& a, const entry_t& b) const {
      if (a.second != b.second) return a.second < b.second;
      return a.first < b.first;
    }
  };

  // merge sort state shares with s_candidate since they don't overlap.
  using BlockMergeSortT = cub::BlockMergeSort<entry_t, BLOCK_SIZE, ITEMS_PER_THREAD>;
  constexpr size_t cand_bytes = sizeof(entry_t) * MAX_CANDS;
  constexpr size_t sort_bytes = sizeof(typename BlockMergeSortT::TempStorage);
  constexpr size_t shared_bytes = (cand_bytes > sort_bytes) ? cand_bytes : sort_bytes;
  __shared__ __align__(alignof(entry_t)) char shared_buf[shared_bytes];
  entry_t* s_candidates = reinterpret_cast<entry_t*>(shared_buf);
  auto& sort_storage = reinterpret_cast<typename BlockMergeSortT::TempStorage&>(shared_buf);

  __shared__ uint32_t s_count;
  __shared__ index_t  s_selected[R];
  __shared__ uint32_t s_n_selected;
  __shared__ bool     s_pruned;

  // block merge sort state
  entry_t items[ITEMS_PER_THREAD];

  // cg tile
  auto thread_block = cg::this_thread_block();
  auto my_tile = cg::tiled_partition<TILE_SIZE>(thread_block);
  uint64_t tile_id = my_tile.meta_group_rank();
  uint64_t n_tiles = my_tile.meta_group_size();

  for (index_t vid = blockIdx.x; vid < num_vertices; vid += gridDim.x) {
    index_t rev_start = reverse_offsets[vid];
    index_t rev_end   = reverse_offsets[vid + 1];
    uint32_t n_reverse = rev_end - rev_start;
    if (n_reverse == 0) continue;

    const data_t* src_vec = all_vectors + static_cast<uint64_t>(vid) * dim;

    // merge existing edges + reverse edges to candidates
    if (threadIdx.x == 0) s_count = 0;
    __syncthreads();

    // grab existing edges
    uint8_t n_edges = edge_counts[vid];
    for (uint32_t i=tile_id; i<n_edges; i+=n_tiles) {
      index_t neighbor = graph_edges[vid].edges[i];
      if (neighbor == INVALID_INDEX) continue;

      const data_t* neighbor_vec = all_vectors + static_cast<uint64_t>(neighbor) * dim;
      distance_t d = compute_distance<CONSTRUCT_GRAPH_CONFIG::graph_cfg_t::dist_func,
        data_t, distance_t, TILE_SIZE>(src_vec, neighbor_vec, dim, my_tile);

      if (my_tile.thread_rank() == 0) {
        uint32_t slot = atomicAdd(&s_count, 1u);
        if (slot < MAX_CANDS) s_candidates[slot] = {neighbor, d};
      }
    }

    // grab reversed edge sinks
    for (uint32_t i=tile_id; i<n_reverse; i+=n_tiles) {
      index_t sink = reverse_edges[rev_start + i].sink;
      if (sink == INVALID_INDEX || sink == vid) continue;

      const data_t* sink_vec = all_vectors + static_cast<uint64_t>(sink) * dim;
      distance_t d = compute_distance<CONSTRUCT_GRAPH_CONFIG::graph_cfg_t::dist_func,
          data_t, distance_t, TILE_SIZE>(src_vec, sink_vec, dim, my_tile);

      if (my_tile.thread_rank() == 0) {
        uint32_t slot = atomicAdd(&s_count, 1u);
        if (slot < MAX_CANDS) s_candidates[slot] = {sink, d};
      }
    }
    __syncthreads();

    uint32_t count = min(s_count, (uint32_t)MAX_CANDS);

    // block merge sort
    #pragma unroll
    for (uint32_t i=0; i<ITEMS_PER_THREAD; i++) {
      uint32_t idx = threadIdx.x * ITEMS_PER_THREAD + i;
      items[i] = (idx < count) ? s_candidates[idx] : SENTINEL;
    }

    __syncthreads();
    BlockMergeSortT(sort_storage).Sort(items, EntryLess());
    __syncthreads();

    #pragma unroll
    for (uint32_t i=0; i<ITEMS_PER_THREAD; i++) {
      uint32_t idx = threadIdx.x * ITEMS_PER_THREAD + i;
      if (idx < count) s_candidates[idx] = items[i];
    }
    __syncthreads();

    // robust prune
    if (threadIdx.x == 0) s_n_selected = 0;
    __syncthreads();

    for (uint32_t i=0; i<count; i++) {
      if (s_n_selected >= R) break;

      index_t cand_id = s_candidates[i].first;
      distance_t cand_dist = s_candidates[i].second;

      if (cand_id == vid || cand_id == INVALID_INDEX) continue;

      const data_t* cand_vec = all_vectors + static_cast<uint64_t>(cand_id) * dim;

      if (threadIdx.x == 0) s_pruned = false;
      __syncthreads();

      uint32_t n_sel = s_n_selected;
      for (uint32_t j=tile_id; j<n_sel; j+=n_tiles) {
        const data_t* selected_vec = all_vectors + static_cast<uint64_t>(s_selected[j]) * dim;
        distance_t d = compute_distance<CONSTRUCT_GRAPH_CONFIG::graph_cfg_t::dist_func, data_t, distance_t, TILE_SIZE>(
          cand_vec, selected_vec, dim, my_tile
        );

        if (my_tile.thread_rank() == 0) {
          if (d * alpha < cand_dist) {
            s_pruned = true;
          }
        }
      }
      __syncthreads();

      if (!s_pruned) {
        if (threadIdx.x == 0) {
          uint32_t slot = s_n_selected;
          if (slot < R) {
            s_selected[slot] = cand_id;
            s_n_selected = slot + 1;
          }
        }
      }
      
      __syncthreads();
    }

    // write back to graph
    uint32_t final_count = s_n_selected;
    for (uint32_t i=threadIdx.x; i<final_count; i+=blockDim.x) {
      graph_edges[vid].edges[i] = s_selected[i];
    }
    if (threadIdx.x == 0) {
      edge_counts[vid] = static_cast<uint8_t>(final_count);
    }

    __syncthreads();
    
  } // end for
}

template <typename CONSTRUCT_GRAPH_CONFIG,
          typename BEAM_SEARCH_CONFIG>
__host__ void process_batch(
  typename CONSTRUCT_GRAPH_CONFIG::graph_t& graph, 
  graph_construct_workspace<CONSTRUCT_GRAPH_CONFIG>& ws,
  uint32_t batch_offset,
  uint32_t batch_size, 
  float alpha,
  construct_timer& timer
) {
  using data_t      = typename CONSTRUCT_GRAPH_CONFIG::data_t;
  using entry_t     = typename CONSTRUCT_GRAPH_CONFIG::entry_t;
  using index_t     = typename CONSTRUCT_GRAPH_CONFIG::index_t;
  using edge_list_t = typename CONSTRUCT_GRAPH_CONFIG::edge_list_t;
  constexpr uint32_t L = CONSTRUCT_GRAPH_CONFIG::L;
  constexpr uint32_t R = CONSTRUCT_GRAPH_CONFIG::R;
  constexpr uint32_t max_result = CONSTRUCT_GRAPH_CONFIG::beam_search_max_result_size;
  uint32_t max_candidates = max_result + R;
  uint32_t dim = graph.vectors.dim;

  constexpr uint32_t BLOCK_SIZE = CONSTRUCT_GRAPH_CONFIG::block_size;
  constexpr uint32_t TILE_SIZE  = CONSTRUCT_GRAPH_CONFIG::tile_size;

  cudaEvent_t e0, e1, e2, e3, e4, e5, e6;
  cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventCreate(&e2);
  cudaEventCreate(&e3); cudaEventCreate(&e4); cudaEventCreate(&e5);
  cudaEventCreate(&e6);

  // beam search
  cudaEventRecord(e0);
  vector_view<data_t> d_query_vectors = graph.vectors.subview(batch_offset, batch_size);
  beam_search_params<BEAM_SEARCH_CONFIG> bp {
    .graph          = graph,
    .data_vectors   = graph.vectors,
    .query_vectors  = d_query_vectors,
    .medoid         = graph.medoid,
    .k              = 1,
    .beam_width     = L,
    .limit          = CONSTRUCT_GRAPH_CONFIG::BEAM_SEARCH_LIMIT,
  };
  beam_search_result<typename CONSTRUCT_GRAPH_CONFIG::graph_cfg_t> bs_result;
  bs_result.frontier       = ws.frontier;
  bs_result.visited        = ws.visited;
  bs_result.visited_counts = ws.visited_counts;
  beam_search<BEAM_SEARCH_CONFIG>(bp, bs_result);
  cudaEventRecord(e1);

  // add the edges to the points in batch
  merge_candidates_kernel<CONSTRUCT_GRAPH_CONFIG, TILE_SIZE, BLOCK_SIZE><<<batch_size, BLOCK_SIZE>>>(
      ws.visited, ws.visited_counts,
      graph.edges, graph.edge_counts,
      graph.vectors.data, d_query_vectors.data,
      dim, batch_offset, batch_size,
      max_result, max_candidates,
      ws.prune_candidates, ws.prune_counts);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    std::cerr << "merge_candidates_kernel failed: " << cudaGetErrorString(err) << std::endl;
  }
  cudaEventRecord(e2);
  
  // robust prune
  robust_prune_kernel<CONSTRUCT_GRAPH_CONFIG, TILE_SIZE><<<batch_size, BLOCK_SIZE>>>(
      ws.prune_candidates, ws.prune_counts,
      graph.edges, graph.edge_counts,
      graph.vectors.data, dim,
      batch_offset, batch_size,
      max_candidates, alpha);
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    std::cerr << "robust_prune_kernel failed: " << cudaGetErrorString(err) << std::endl;
  }
  cudaEventRecord(e3);
  
  // reverse edges
  uint32_t total_slots = batch_size * R;
  dim3 grid((total_slots + BLOCK_SIZE - 1) / BLOCK_SIZE);
  fill_reverse_edges<CONSTRUCT_GRAPH_CONFIG><<<grid, BLOCK_SIZE>>>(
    graph.edges, graph.edge_counts,
    ws.reverse_edges_ptr,
    batch_size, 
    batch_offset, 
    R);
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    std::cerr << "reverse edges failed: " << cudaGetErrorString(err) << std::endl;
  }
  cudaEventRecord(e4);

  // semisort using thrust
  // after this step, the reversed edges will be stored 
  // in `ws.reverse_edges` and `ws.reverse_offsets`
  edge_pair<index_t> sentinel = edge_pair<index_t>::sentinel();
  thrust::sort(
    ws.reverse_edges.begin(), 
    ws.reverse_edges.begin() + total_slots, 
    semi_sort_compare_edge_pair<index_t>{});

  auto it = thrust::lower_bound(
    ws.reverse_edges.begin(), 
    ws.reverse_edges.begin() + total_slots,
    sentinel, 
    semi_sort_compare_edge_pair<index_t>{});
  index_t n_edges = static_cast<index_t>(it - ws.reverse_edges.begin());

  if (n_edges == 0) return;

  edge_pair<index_t> last_edge = ws.reverse_edges[n_edges - 1]; 
  index_t num_vertices = last_edge.source + 1;

  std::cout << "batch_size=" << static_cast<uint32_t>(batch_size) 
    << " n_reverse_edges="<< static_cast<uint32_t>(n_edges) << " num_vertices=" << static_cast<uint32_t>(num_vertices) << std::endl;

  // thrust::host_vector<edge_pair<index_t>> h_reverse_edges = reverse_edges;
  // std::cout << "reverse_edges=[";
  // for(auto i = h_reverse_edges.begin(); i != h_reverse_edges.end(); ++i) std::cout << *i << " ";
  // std::cout << "]" << std::endl;

  // thrust::host_vector<index_t> h_reverse_edges = reverse_edges;
  // std::cout << "reverse_edges=[";
  // for(auto i = h_reverse_edges.begin(); i != h_reverse_edges.end(); ++i) std::cout << *i << " ";
  // std::cout << "]" << std::endl;

  // if (num_vertices + 1 > ws.reverse_offsets.size()) {
  //   ws.reverse_offsets.resize(num_vertices + 1);
  //   ws.reverse_offsets_ptr = thrust::raw_pointer_cast(ws.reverse_offsets.data());
  // }

  auto src_begin = thrust::make_transform_iterator(
            ws.reverse_edges.begin(), extract_source_from_edge_pair<index_t>{});
  thrust::counting_iterator<index_t> count(0);
  thrust::lower_bound(src_begin, src_begin + n_edges,
                      count, count + num_vertices + 1,
                      ws.reverse_offsets.begin());

  // get unique sources
  

  cudaEventRecord(e5);

  // add edges + robust prune
  constexpr uint64_t process_reverse_edges_kernel_grid_size = 1024; // fix size
  process_reverse_edges_kernel<CONSTRUCT_GRAPH_CONFIG, TILE_SIZE>
       <<<process_reverse_edges_kernel_grid_size, BLOCK_SIZE>>>( //<<<num_vertices, BLOCK_SIZE>>>(
          ws.reverse_edges_ptr,
          ws.reverse_offsets_ptr,
          graph.edges,
          graph.edge_counts,
          graph.vectors.data,
          dim,
          num_vertices,
          alpha);
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    std::cerr << "process_reverse_edges_kernel failed: " << cudaGetErrorString(err) << std::endl;
  }
  cudaEventRecord(e6);

  cudaEventSynchronize(e6);
  timer.beam_search_ms   += elapsed_ms(e0, e1);
  timer.merge_cands_ms   += elapsed_ms(e1, e2);
  timer.robust_prune_ms  += elapsed_ms(e2, e3);
  timer.fill_reverse_ms  += elapsed_ms(e3, e4);
  timer.sort_offsets_ms  += elapsed_ms(e4, e5);
  timer.reverse_prune_ms += elapsed_ms(e5, e6);
  timer.n_batches++;

  cudaEventDestroy(e0); cudaEventDestroy(e1); cudaEventDestroy(e2);
  cudaEventDestroy(e3); cudaEventDestroy(e4); cudaEventDestroy(e5);
  cudaEventDestroy(e6);
}

template <typename CONSTRUCT_GRAPH_CONFIG,
          typename BEAM_SEARCH_CONFIG>
__host__ void construction_round(
  typename CONSTRUCT_GRAPH_CONFIG::graph_t &graph, 
  graph_construct_workspace<CONSTRUCT_GRAPH_CONFIG> &ws,
  float alpha, 
  uint32_t max_batch_size,
  construct_timer& timer
){
  uint32_t count = 0;
  uint32_t batch_size = 1;

  std::printf("[construct] round alpha=%.2f, n_vectors=%u\n", alpha, graph.n_vectors);

  while (count < graph.n_vectors) {
    batch_size = std::min(batch_size, graph.n_vectors - count);
    batch_size = std::min(batch_size, max_batch_size);

    process_batch<CONSTRUCT_GRAPH_CONFIG, BEAM_SEARCH_CONFIG>(
        graph, ws, count, batch_size, alpha, timer);

    count += batch_size;

    // std::printf("\r[construct] %u / %u (%.1f%%)", count, graph.n_vectors,
    //             100.0f * count / graph.n_vectors);
    // std::fflush(stdout);

    if (batch_size < max_batch_size) {
      batch_size = std::min(max_batch_size, batch_size * 2);
    }
  }

  std::printf("\n[construct] round complete\n");
}

template <typename CONSTRUCT_GRAPH_CONFIG>
__host__ graph<typename CONSTRUCT_GRAPH_CONFIG::graph_cfg_t> construct_graph(
  graph_construct_params<CONSTRUCT_GRAPH_CONFIG> params
){
  using graph_cfg_t = typename CONSTRUCT_GRAPH_CONFIG::graph_cfg_t;
  using graph_t = typename CONSTRUCT_GRAPH_CONFIG::graph_t;
  using index_t = typename graph_cfg_t::index_t;
  using distance_t = typename graph_cfg_t::distance_t;
  using edge_list_t = typename graph_cfg_t::edge_list_t;
  using vector_view_t = typename graph_cfg_t::vector_view_t;
  using entry_t       = thrust::pair<index_t, distance_t>;

  // constexpr uint32_t L = CONSTRUCT_GRAPH_CONFIG::L;
  float alpha = params.alpha;
  // Hard code medoid to be the first vector.
  // From empirical data it does not affect the performance.
  index_t medoid = 0;

  uint32_t vector_dim = params.data_vectors.dim;
  uint32_t n_vectors = params.data_vectors.n_vectors;

  if (params.on_host) {
    std::cerr << "on_host index construction is not supported." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // allocate graph on device memory
  graph_t g;
  g.dim = vector_dim;
  g.n_vectors = n_vectors;
  g.medoid = medoid;
  cudaMalloc(&g.edge_counts, sizeof(uint8_t)      * n_vectors);
  cudaMalloc(&g.edges,       sizeof(edge_list_t)  * n_vectors);
  g.vectors = params.data_vectors.to_device();

  // workspace
  auto ws = graph_construct_workspace<CONSTRUCT_GRAPH_CONFIG>::allocate(params.max_batch_size);
  ws.print_space_usage();

  constexpr uint32_t beam_search_tile_size = 4;
  constexpr uint32_t beam_search_block_size = 64;
  using beam_search_cfg = beam_search_config<
    graph_cfg_t, graph_cfg_t::dist_func,
    beam_search_block_size,
    true,  // get visited
    CONSTRUCT_GRAPH_CONFIG::beam_search_max_search_width,
    beam_search_tile_size,
    CONSTRUCT_GRAPH_CONFIG::beam_search_max_result_size>;

  // first round establish edges
  // construct_timer first_round_timer;
  // construction_round<CONSTRUCT_GRAPH_CONFIG, beam_search_cfg>(
  //   g, ws, 1.0, params.max_batch_size, first_round_timer
  // );  
  // first_round_timer.print();

  // second round add long hop edges
  construct_timer second_round_timer;
  construction_round<CONSTRUCT_GRAPH_CONFIG, beam_search_cfg>(
    g, ws, alpha, params.max_batch_size, second_round_timer
  );
  second_round_timer.print();

  ws.free();
  return g;
}

}