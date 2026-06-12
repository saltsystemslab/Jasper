#pragma once

#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <type_traits>
#include <vector>

#include <cooperative_groups.h>

#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/binary_search.h>
#include <thrust/adjacent_difference.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/execution_policy.h>

#include "jasper/index/graph.cuh"
#include "jasper/index/vector.cuh"
#include "jasper/beam_search/beam_search.cuh"
#include "jasper/index/utils.cuh"
#include "jasper/rotation/rotation.cuh"

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
  // 1. less or equal to 2% of the total vector.
  // 2. batch size needs to be fit inside device memory.
  uint32_t max_batch_size = 10000;

  // Is the final graph on host
  bool on_host = false;

  // Prerotate the dataset
  bool prerotate = false;

  // Prerotate seed
  uint32_t prerotate_seed = 42;
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
  thrust::device_vector<index_t> reverse_offsets_flags;    // [batch_size * R]
  thrust::device_vector<index_t> reverse_offsets;          // [batch_size * R + 1]
  edge_pair<index_t>* reverse_edges_ptr;
  index_t* reverse_offsets_flags_ptr;
  index_t* reverse_offsets_ptr;

  static graph_construct_workspace allocate(uint32_t max_batch_size) {
    graph_construct_workspace ws;
    ws.max_batch_size = max_batch_size;
    constexpr uint32_t L = GRAPH_CONSTRUCT_CONFIG::L;
    constexpr uint32_t R = GRAPH_CONSTRUCT_CONFIG::R;
    constexpr uint32_t max_result_size = GRAPH_CONSTRUCT_CONFIG::beam_search_max_result_size;
    constexpr uint32_t max_candidates = max_result_size + R;

    size_t total_bytes =
        sizeof(entry_t)  * max_batch_size * L +
        sizeof(entry_t)  * max_batch_size * max_result_size +
        sizeof(uint32_t) * max_batch_size +
        sizeof(entry_t)  * max_batch_size * max_candidates +
        sizeof(uint32_t) * max_batch_size;

    auto check = [&](cudaError_t err, const char* name) {
      if (err != cudaSuccess) {
        const char* err_str = cudaGetErrorString(err);
        // Clear the sticky error so subsequent CUDA calls don't fail
        cudaGetLastError();
        // Free any buffers that were already allocated
        ws.free();

        size_t free_mem = 0, total_mem = 0;
        cudaMemGetInfo(&free_mem, &total_mem);

        throw std::runtime_error(
          std::string("cudaMalloc failed for ") + name + ": " + err_str +
          " (requested workspace: " + std::to_string(total_bytes / (1 << 20)) + " MB, "
          "GPU free: " + std::to_string(free_mem / (1 << 20)) + " MB). "
          "Try increasing --workspace-budget or reducing batch size.");
      }
    };

    try {
      check(cudaMalloc(&ws.frontier,         sizeof(entry_t)  * max_batch_size * L),              "frontier");
      check(cudaMalloc(&ws.visited,          sizeof(entry_t)  * max_batch_size * max_result_size), "visited");
      check(cudaMalloc(&ws.visited_counts,   sizeof(uint32_t) * max_batch_size),                  "visited_counts");
      check(cudaMalloc(&ws.prune_candidates, sizeof(entry_t)  * max_batch_size * max_candidates), "prune_candidates");
      check(cudaMalloc(&ws.prune_counts,     sizeof(uint32_t) * max_batch_size),                  "prune_counts");

      ws.reverse_edges.resize(max_batch_size * R);
      ws.reverse_offsets_flags.resize(max_batch_size * R);
      ws.reverse_offsets.resize(max_batch_size * R + 1);
    } catch (...) {
      cudaGetLastError();  // Clear any sticky error from thrust
      ws.free();
      throw;
    }

    ws.reverse_edges_ptr = thrust::raw_pointer_cast(ws.reverse_edges.data());
    ws.reverse_offsets_flags_ptr = thrust::raw_pointer_cast(ws.reverse_offsets_flags.data());
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
    size_t rev_offsets_flags_bytes = sizeof(index_t)       * reverse_offsets_flags.size();
    size_t rev_offsets_bytes  = sizeof(index_t)            * reverse_offsets.size();

    size_t total = frontier_bytes + visited_bytes + visited_cnt_bytes
                 + prune_cand_bytes + prune_cnt_bytes
                 + rev_edges_bytes + rev_offsets_bytes + rev_offsets_flags_bytes;

    auto mb = [](size_t b) { return b / (1024.0 * 1024.0); };

    std::printf("[workspace] max_batch_size=%u, R=%u, L=%u\n", max_batch_size, R, L);
    std::printf("  frontier         : %8.2f MB\n", mb(frontier_bytes));
    std::printf("  visited          : %8.2f MB\n", mb(visited_bytes));
    std::printf("  visited_counts   : %8.2f MB\n", mb(visited_cnt_bytes));
    std::printf("  prune_candidates : %8.2f MB\n", mb(prune_cand_bytes));
    std::printf("  prune_counts     : %8.2f MB\n", mb(prune_cnt_bytes));
    std::printf("  reverse_edges    : %8.2f MB\n", mb(rev_edges_bytes));
    std::printf("  reverse_flags    : %8.2f MB\n", mb(rev_offsets_flags_bytes));
    std::printf("  reverse_offsets  : %8.2f MB\n", mb(rev_offsets_bytes));
    std::printf("  total            : %8.2f MB\n", mb(total)); 
  }

  // Returns the largest max_batch_size whose total allocation fits in `budget_bytes`.
  // Returns 0 if even a batch size of 1 does not fit.
  static uint32_t max_batch_size_for_budget(size_t budget_bytes) {
    constexpr uint32_t L              = GRAPH_CONSTRUCT_CONFIG::L;
    constexpr uint32_t R              = GRAPH_CONSTRUCT_CONFIG::R;
    constexpr uint32_t max_result_size = GRAPH_CONSTRUCT_CONFIG::beam_search_max_result_size;
    constexpr uint32_t max_candidates = max_result_size + R;

    // Per-batch-element cost (bytes)
    constexpr size_t per_item =
        sizeof(entry_t)            * L                // frontier
      + sizeof(entry_t)            * max_result_size  // visited
      + sizeof(uint32_t)                              // visited_counts
      + sizeof(entry_t)            * max_candidates   // prune_candidates
      + sizeof(uint32_t)                              // prune_counts
      + sizeof(edge_pair<index_t>) * R                // reverse_edges
      + sizeof(index_t)            * R                // rev_offsets_flags
      + sizeof(index_t)            * R;               // reverse_offsets

    // Fixed overhead: reverse_offsets has one extra element (+1)
    constexpr size_t fixed = sizeof(index_t);  // the +1 slot in reverse_offsets

    if (budget_bytes <= fixed) return 0;

    uint32_t batch = static_cast<uint32_t>((budget_bytes - fixed) / per_item);
    return batch;
  }

  void free() {
    cudaFree(frontier);
    cudaFree(visited);
    cudaFree(visited_counts);
    cudaFree(prune_candidates);
    cudaFree(prune_counts);
    
    reverse_edges.clear();
    reverse_edges.shrink_to_fit();
    reverse_offsets_flags.clear();
    reverse_offsets_flags.shrink_to_fit();
    reverse_offsets.clear();
    reverse_offsets.shrink_to_fit();
  }
};

template <typename CONSTRUCT_GRAPH_CONFIG, uint32_t TILE_SIZE, uint32_t BLOCK_SIZE>
__global__ void merge_candidates_kernel(
  const typename CONSTRUCT_GRAPH_CONFIG::entry_t* __restrict__ visited,
  const uint32_t* __restrict__ visited_counts,
  typename CONSTRUCT_GRAPH_CONFIG::graph_t::device_view graph,
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

  // add existing neighbors using stored distances (avoids recomputing d(node, neighbor))
  uint8_t n_edges = graph.get_edge_count(global_id);
  for (uint32_t i = threadIdx.x; i < n_edges; i += blockDim.x) {
    index_t neighbor = graph.get_neighbor(global_id, i);
    if (neighbor == INVALID_INDEX) continue;
    distance_t d = static_cast<distance_t>(graph.get_neighbor_dist(global_id, i));
    uint32_t slot = atomicAdd(&s_count, 1u);
    if (slot < max_candidates) my_out[slot] = {neighbor, d};
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
  typename CONSTRUCT_GRAPH_CONFIG::graph_t::device_view graph,
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
  __shared__ distance_t s_selected_dist[R];
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

    data_t* cand_vec = graph.get_vector(cand_id);

    // Reset prune flag
    if (threadIdx.x == 0) s_pruned = false;
    __syncthreads();

    // Each tile checks one already-selected neighbor in parallel.
    // Tiles round-robin over selected neighbors.
    uint32_t n_sel = s_n_selected;
    for (uint32_t j = tile_id; j < n_sel; j += n_tiles) {
      data_t* sel_vec = graph.get_vector(s_selected[j]);

      distance_t d = compute_distance<CONSTRUCT_GRAPH_CONFIG::graph_cfg_t::dist_func, data_t, distance_t, TILE_SIZE>(
        cand_vec, sel_vec, dim, my_tile);

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
          s_selected_dist[slot] = cand_dist;
          s_n_selected = slot + 1;
        }
      }
    }
    __syncthreads();
  }

  uint32_t final_count = s_n_selected;
  for (uint32_t i = threadIdx.x; i < final_count; i += blockDim.x) {
    graph.set_neighbor(global_id, i, s_selected[i]);
    graph.set_neighbor_dist(global_id, i, s_selected_dist[i]);
  }
  if (threadIdx.x == 0) {
    graph.set_edge_count(global_id, final_count);
  }
}

template <typename CONSTRUCT_GRAPH_CONFIG>
__global__ void fill_reverse_edges(
  typename CONSTRUCT_GRAPH_CONFIG::graph_t::device_view graph,
  edge_pair<typename CONSTRUCT_GRAPH_CONFIG::index_t>* reverse_edges,
  uint32_t batch_size,
  uint32_t batch_offset,
  uint32_t max_edges_per_node // R
) {
  using index_t = typename CONSTRUCT_GRAPH_CONFIG::index_t;

  // one thread per slot in [batch_size * max_edges_per_node]
  uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= (uint64_t)batch_size * max_edges_per_node) return;

  uint32_t local_id = tid / max_edges_per_node;
  uint32_t edge_idx = tid % max_edges_per_node;
  index_t  global_id = local_id + batch_offset;

  if (edge_idx < graph.get_edge_count(global_id)) {
    index_t neighbor = graph.get_neighbor(global_id, edge_idx);
    // edge in graph:   global_id -> neighbor
    // reversed:        source=neighbor, sink=global_id
    reverse_edges[tid] = {neighbor, global_id};
  } else {
    reverse_edges[tid] = edge_pair<index_t>::sentinel();
  }
}

// Given a sorted-by-source vector of (source, sink) pairs,
// returns a vector of starting indices for each unique source,
// and (via reference) the count of unique sources.
template <typename CONSTRUCT_GRAPH_CONFIG>
void get_unique_source_offsets(
  thrust::device_vector<edge_pair<typename CONSTRUCT_GRAPH_CONFIG::index_t>>& reversed_edges,
  typename CONSTRUCT_GRAPH_CONFIG::index_t n_reversed_edges,
  thrust::device_vector<typename CONSTRUCT_GRAPH_CONFIG::index_t>& reverse_offsets_flags,
  thrust::device_vector<typename CONSTRUCT_GRAPH_CONFIG::index_t>& reverse_offsets,
  typename CONSTRUCT_GRAPH_CONFIG::index_t& num_unique_sources,
  cudaStream_t stream = 0
){
  using index_t = typename CONSTRUCT_GRAPH_CONFIG::index_t;

  if (n_reversed_edges == 0) {
    num_unique_sources = 0;
    return;
  }

  auto policy = thrust::cuda::par.on(stream);

  auto src_begin = thrust::make_transform_iterator(
    reversed_edges.begin(), extract_source_from_edge_pair<index_t>());
  auto src_end   = thrust::make_transform_iterator(
    reversed_edges.begin()+n_reversed_edges, extract_source_from_edge_pair<index_t>());

  // mark where a new source group begins:
  // flags[i] = 1 if i == 0 or source[i] != source[i-1]
  thrust::adjacent_difference(policy, src_begin, src_end, reverse_offsets_flags.begin());

  // adjacent_difference copies the first element as-is; force it to 1
  reverse_offsets_flags[0] = 1;

  // For the remaining elements, we just need != 0 → 1
  thrust::transform(policy,
                    reverse_offsets_flags.begin() + 1, reverse_offsets_flags.end(),
                    reverse_offsets_flags.begin() + 1,
                    [] __device__ (int v) { return v != 0 ? 1 : 0; });

  // count unique sources
  num_unique_sources = thrust::reduce(
    policy,
    reverse_offsets_flags.begin(),
    reverse_offsets_flags.begin()+n_reversed_edges);

  // collect the indices where flags == 1
  thrust::copy_if(
    policy,
    thrust::counting_iterator<int>(0),
    thrust::counting_iterator<int>(n_reversed_edges),
    reverse_offsets_flags.begin(),
    reverse_offsets.begin(),
    [] __device__ (int f) { return f == 1; });

  reverse_offsets[num_unique_sources] = n_reversed_edges;
}

template <typename CONSTRUCT_GRAPH_CONFIG, uint32_t TILE_SIZE>
__global__ void process_reverse_edges_kernel(
  const edge_pair<typename CONSTRUCT_GRAPH_CONFIG::index_t>* __restrict__ reverse_edges,
  const typename CONSTRUCT_GRAPH_CONFIG::index_t* __restrict__ reverse_offsets,
  typename CONSTRUCT_GRAPH_CONFIG::graph_t::device_view graph,
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

  __shared__ uint32_t   s_count;
  __shared__ index_t    s_selected[R];
  __shared__ distance_t s_selected_dist[R];
  __shared__ uint32_t   s_n_selected;
  __shared__ bool       s_pruned;

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

    index_t actual_vertex = reverse_edges[rev_start].source;
    data_t* src_vec = graph.get_vector(actual_vertex);

    // merge existing edges + reverse edges to candidates
    if (threadIdx.x == 0) s_count = 0;
    __syncthreads();

    // grab existing edges using stored distances (avoids recomputing d(vertex, neighbor))
    uint8_t n_edges = graph.get_edge_count(actual_vertex);
    for (uint32_t i=threadIdx.x; i<n_edges; i+=blockDim.x) {
      index_t neighbor = graph.get_neighbor(actual_vertex, i);
      if (neighbor == INVALID_INDEX) continue;

      // data_t* neighbor_vec = graph.get_vector(neighbor);
      // distance_t d = compute_distance<CONSTRUCT_GRAPH_CONFIG::graph_cfg_t::dist_func,
      //   data_t, distance_t, TILE_SIZE>(src_vec, neighbor_vec, dim, my_tile);

      // if (my_tile.thread_rank() == 0) {
      //   uint32_t slot = atomicAdd(&s_count, 1u);
      //   if (slot < MAX_CANDS) s_candidates[slot] = {neighbor, d};
      // }
      distance_t d = static_cast<distance_t>(graph.get_neighbor_dist(actual_vertex, i));
      uint32_t slot = atomicAdd(&s_count, 1u);
      if (slot < MAX_CANDS) s_candidates[slot] = {neighbor, d};
    }

    // grab reversed edge sinks
    for (uint32_t i=tile_id; i<n_reverse; i+=n_tiles) {
      index_t sink = reverse_edges[rev_start + i].sink;
      if (sink == INVALID_INDEX || sink == actual_vertex) continue;

      data_t* sink_vec = graph.get_vector(sink);
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

      if (cand_id == actual_vertex || cand_id == INVALID_INDEX) continue;

      data_t* cand_vec = graph.get_vector(cand_id);

      if (threadIdx.x == 0) s_pruned = false;
      __syncthreads();

      uint32_t n_sel = s_n_selected;
      for (uint32_t j=tile_id; j<n_sel; j+=n_tiles) {
        data_t* selected_vec = graph.get_vector(s_selected[j]);
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
            s_selected_dist[slot] = cand_dist;
            s_n_selected = slot + 1;
          }
        }
      }

      __syncthreads();
    }

    // write back to graph
    uint32_t final_count = s_n_selected;
    for (uint32_t i=threadIdx.x; i<final_count; i+=blockDim.x) {
      graph.set_neighbor(actual_vertex, i, s_selected[i]);
      graph.set_neighbor_dist(actual_vertex, i, s_selected_dist[i]);
    }
    if (threadIdx.x == 0) {
      graph.set_edge_count(actual_vertex, static_cast<uint8_t>(final_count));
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
  construct_timer& timer,
  cudaStream_t stream = 0
) {
  using data_t      = typename CONSTRUCT_GRAPH_CONFIG::data_t;
  using entry_t     = typename CONSTRUCT_GRAPH_CONFIG::entry_t;
  using index_t     = typename CONSTRUCT_GRAPH_CONFIG::index_t;
  using edge_list_t = typename CONSTRUCT_GRAPH_CONFIG::edge_list_t;
  constexpr uint32_t L = CONSTRUCT_GRAPH_CONFIG::L;
  constexpr uint32_t R = CONSTRUCT_GRAPH_CONFIG::R;
  constexpr uint32_t max_result = CONSTRUCT_GRAPH_CONFIG::beam_search_max_result_size;
  uint32_t max_candidates = max_result + R;
  uint32_t dim = graph.dim;

  constexpr uint32_t BLOCK_SIZE = CONSTRUCT_GRAPH_CONFIG::block_size;
  constexpr uint32_t TILE_SIZE  = CONSTRUCT_GRAPH_CONFIG::tile_size;

  typename CONSTRUCT_GRAPH_CONFIG::graph_t::device_view device_graph_view = graph.view();

  cudaEvent_t e0, e1, e2, e3, e4, e5, e6;
  cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventCreate(&e2);
  cudaEventCreate(&e3); cudaEventCreate(&e4); cudaEventCreate(&e5);
  cudaEventCreate(&e6);

  auto thrust_policy = thrust::cuda::par.on(stream);

  // beam search
  cudaEventRecord(e0, stream);
  beam_search_params<BEAM_SEARCH_CONFIG> bp {
    .graph          = graph,
    .use_range      = true,
    .query_start    = batch_offset,
    .query_end      = batch_offset + batch_size,
    .medoid         = graph.medoid,
    .k              = L,
    .beam_width     = L,
    .limit          = CONSTRUCT_GRAPH_CONFIG::BEAM_SEARCH_LIMIT,
  };
  beam_search_result<typename CONSTRUCT_GRAPH_CONFIG::graph_cfg_t> bs_result;
  bs_result.frontier       = ws.frontier;
  bs_result.visited        = ws.visited;
  bs_result.visited_counts = ws.visited_counts;
  beam_search<BEAM_SEARCH_CONFIG>(bp, bs_result, stream);
  cudaEventRecord(e1, stream);

  // add the edges to the points in batch
  merge_candidates_kernel<CONSTRUCT_GRAPH_CONFIG, TILE_SIZE, BLOCK_SIZE><<<batch_size, BLOCK_SIZE, 0, stream>>>(
      ws.visited,
      ws.visited_counts,
      device_graph_view,
      dim,
      batch_offset,
      batch_size,
      max_result,
      max_candidates,
      ws.prune_candidates,
      ws.prune_counts);
  cudaError_t err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    std::cerr << "merge_candidates_kernel failed: " << cudaGetErrorString(err) << std::endl;
  }
  cudaEventRecord(e2, stream);

  // robust prune
  robust_prune_kernel<CONSTRUCT_GRAPH_CONFIG, TILE_SIZE><<<batch_size, BLOCK_SIZE, 0, stream>>>(
      ws.prune_candidates,
      ws.prune_counts,
      device_graph_view,
      dim,
      batch_offset,
      batch_size,
      max_candidates,
      alpha);
  err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    std::cerr << "robust_prune_kernel failed: " << cudaGetErrorString(err) << std::endl;
  }
  cudaEventRecord(e3, stream);

  // reverse edges
  uint32_t total_slots = batch_size * R;
  dim3 grid((total_slots + BLOCK_SIZE - 1) / BLOCK_SIZE);
  fill_reverse_edges<CONSTRUCT_GRAPH_CONFIG><<<grid, BLOCK_SIZE, 0, stream>>>(
    device_graph_view,
    ws.reverse_edges_ptr,
    batch_size,
    batch_offset,
    R);
  err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    std::cerr << "reverse edges failed: " << cudaGetErrorString(err) << std::endl;
  }
  cudaEventRecord(e4, stream);

  // semisort using thrust
  // after this step, the reversed edges will be stored
  // in `ws.reverse_edges` and `ws.reverse_offsets`
  edge_pair<index_t> sentinel = edge_pair<index_t>::sentinel();
  thrust::sort(
    thrust_policy,
    ws.reverse_edges.begin(),
    ws.reverse_edges.begin() + total_slots,
    semi_sort_compare_edge_pair<index_t>{});

  auto it = thrust::lower_bound(
    thrust_policy,
    ws.reverse_edges.begin(),
    ws.reverse_edges.begin() + total_slots,
    sentinel,
    semi_sort_compare_edge_pair<index_t>{});
  index_t n_edges = static_cast<index_t>(it - ws.reverse_edges.begin());

  if (n_edges == 0) {
    cudaEventDestroy(e0); cudaEventDestroy(e1); cudaEventDestroy(e2);
    cudaEventDestroy(e3); cudaEventDestroy(e4); cudaEventDestroy(e5);
    cudaEventDestroy(e6);
    return;
  }

  index_t num_unique_sources = 0;
  get_unique_source_offsets<CONSTRUCT_GRAPH_CONFIG>(
    ws.reverse_edges,
    n_edges,
    ws.reverse_offsets_flags,
    ws.reverse_offsets,
    num_unique_sources,
    stream
  );

  cudaEventRecord(e5, stream);

  // add edges + robust prune
  constexpr uint64_t process_reverse_edges_kernel_grid_size = 1024; // fix size
  process_reverse_edges_kernel<CONSTRUCT_GRAPH_CONFIG, TILE_SIZE>
       <<<process_reverse_edges_kernel_grid_size, BLOCK_SIZE, 0, stream>>>(
          ws.reverse_edges_ptr,
          ws.reverse_offsets_ptr,
          device_graph_view,
          dim,
          num_unique_sources,
          alpha);
  err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    std::cerr << "process_reverse_edges_kernel failed: " << cudaGetErrorString(err) << std::endl;
  }
  cudaEventRecord(e6, stream);

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
  construct_timer& timer,
  cudaStream_t stream = 0
){
  uint32_t count = 0;
  uint32_t batch_size = 1;

  std::printf("[construct] round alpha=%.2f, n_vectors=%u, max_batch_size=%u\n", alpha, graph.n_vectors, max_batch_size);

  while (count < graph.n_vectors) {
    batch_size = std::min(batch_size, graph.n_vectors - count);
    batch_size = std::min(batch_size, max_batch_size);

    process_batch<CONSTRUCT_GRAPH_CONFIG, BEAM_SEARCH_CONFIG>(
        graph, ws, count, batch_size, alpha, timer, stream);

    count += batch_size;

    std::printf("\r[construct] %u / %u (%.1f%%)", count, graph.n_vectors,
                100.0f * count / graph.n_vectors);
    std::fflush(stdout);

    if (batch_size < max_batch_size) {
      batch_size = std::min(max_batch_size, batch_size * 2);
    }
  }

  std::printf("\n");
  // std::printf("\n[construct] round complete\n");
}

template <typename CONSTRUCT_GRAPH_CONFIG>
__host__ graph<typename CONSTRUCT_GRAPH_CONFIG::graph_cfg_t> construct_graph(
  graph_construct_params<CONSTRUCT_GRAPH_CONFIG> params,
  cudaStream_t stream = 0
){
  using graph_cfg_t = typename CONSTRUCT_GRAPH_CONFIG::graph_cfg_t;
  using graph_t = typename CONSTRUCT_GRAPH_CONFIG::graph_t;
  using index_t = typename graph_cfg_t::index_t;
  using distance_t = typename graph_cfg_t::distance_t;
  using edge_list_t = typename graph_cfg_t::edge_list_t;
  using vector_view_t = typename graph_cfg_t::vector_view_t;
  using entry_t       = thrust::pair<index_t, distance_t>;

  float alpha = params.alpha;

  uint32_t vector_dim = params.data_vectors.dim;
  uint32_t n_vectors = params.data_vectors.n_vectors;

  auto check_cuda = [](cudaError_t err, const char* name) {
    if (err != cudaSuccess) {
      const char* err_str = cudaGetErrorString(err);
      cudaGetLastError();  // clear sticky error
      throw std::runtime_error(
        std::string("CUDA error in construct_graph at ") + name + ": " + err_str);
    }
  };

  // Replace each row v of params.data_vectors with v * P, where P is a
  // random orthogonal matrix seeded by params.prerotate_seed.
  if (params.prerotate) {
    std::cout << "[construct] params.prerotate is enabled, rotating the vector dataset." << std::endl;
    using data_t = typename graph_cfg_t::data_t;
    static_assert(std::is_same<data_t, __half>::value,
                  "prerotate requires __half (f16) data_t");

    uint32_t padded_dim = params.data_vectors.padded_dim;

    // Build rotation matrix as float (QR needs float), then cast to __half.
    std::vector<float> h_P_f(static_cast<size_t>(vector_dim) * vector_dim);
    std::vector<float> h_Pt_f(static_cast<size_t>(vector_dim) * vector_dim);
    set_rotation_matrix(vector_dim, h_P_f.data(), h_Pt_f.data(),
                        params.prerotate_seed);

    std::vector<__half> h_P(h_P_f.size());
    for (size_t i = 0; i < h_P.size(); ++i) {
      h_P[i] = static_cast<__half>(h_P_f[i]);
    }

    size_t P_bytes = sizeof(__half) * vector_dim * vector_dim;
    __half* d_P = nullptr;
    check_cuda(cudaMalloc(&d_P, P_bytes), "cudaMalloc d_P (prerotate)");
    check_cuda(cudaMemcpy(d_P, h_P.data(), P_bytes, cudaMemcpyHostToDevice),
               "cudaMemcpy d_P (prerotate)");

    size_t data_bytes =
        sizeof(__half) * static_cast<size_t>(n_vectors) * padded_dim;
    const bool input_on_host = params.data_vectors.on_host;
    __half* d_in = nullptr;
    if (input_on_host) {
      check_cuda(cudaMalloc(&d_in, data_bytes),
                 "cudaMalloc d_in (prerotate)");
      check_cuda(cudaMemcpy(d_in, params.data_vectors.data, data_bytes,
                            cudaMemcpyHostToDevice),
                 "cudaMemcpy data to device (prerotate)");
    } else {
      d_in = params.data_vectors.data;
    }

    // Zero d_out so the row pad lanes stay zero: cublas only writes the
    // first vector_dim rows of each column when ldc = padded_dim.
    __half* d_out = nullptr;
    check_cuda(cudaMalloc(&d_out, data_bytes),
               "cudaMalloc d_out (prerotate)");
    check_cuda(cudaMemsetAsync(d_out, 0, data_bytes, stream),
               "cudaMemset d_out (prerotate)");

    // Same orientation trick as rotate_data_vec in rotation.cuh, but with
    // stride = padded_dim to skip row pad lanes. Uses __half I/O with FP32
    // accumulation for accuracy.
    cudaDeviceSynchronize();
    auto r0 = std::chrono::steady_clock::now();
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSetStream(handle, stream);
    const float one = 1.0f, zero = 0.0f;
    cublasStatus_t stat = cublasGemmEx(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        vector_dim,
        n_vectors,
        vector_dim,
        &one,
        d_P,  CUDA_R_16F, vector_dim,
        d_in, CUDA_R_16F, padded_dim,
        &zero,
        d_out, CUDA_R_16F, padded_dim,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT);
    cublasDestroy(handle);
    if (stat != CUBLAS_STATUS_SUCCESS) {
      cudaFree(d_out);
      cudaFree(d_P);
      if (input_on_host) cudaFree(d_in);
      throw std::runtime_error(
          "cublasGemmEx failed during prerotate: status=" +
          std::to_string(stat));
    }
    cudaDeviceSynchronize();
    auto r1 = std::chrono::steady_clock::now();
    auto rs1 = std::chrono::duration_cast<std::chrono::milliseconds>(r1 - r0).count();
    std::cout << "  rotate_data_vectors: "      << rs1 << " ms\n";

    cudaMemcpyKind kind =
        input_on_host ? cudaMemcpyDeviceToHost : cudaMemcpyDeviceToDevice;
    check_cuda(cudaMemcpy(params.data_vectors.data, d_out, data_bytes, kind),
               "cudaMemcpy rotated data back (prerotate)");

    cudaFree(d_out);
    cudaFree(d_P);
    if (input_on_host) cudaFree(d_in);
  }

  cudaError_t err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    std::cerr << "Construct launch failed: "
              << cudaGetErrorString(err) << std::endl;
  }

  size_t free_bytes, total_bytes;
  cudaMemGetInfo(&free_bytes, &total_bytes);
  std::cout << "[graph malloc] free=" << (free_bytes >> 20)
            << " MiB / total=" << (total_bytes >> 20) << " MiB\n";

  // allocate a graph with only vector populated.
  graph_t g = graph_t::allocate_and_load(params.data_vectors, params.on_host);

  // workspace
  auto ws = graph_construct_workspace<CONSTRUCT_GRAPH_CONFIG>::allocate(params.max_batch_size);
  // ws.print_space_usage();

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
    g, ws, alpha, params.max_batch_size, second_round_timer, stream
  );
  second_round_timer.print();

  ws.free();
  return g;
}

}