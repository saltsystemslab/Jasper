#pragma once

#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <vector>
#include <thread>
#include <exception>
#include <numeric>

#include "jasper/index/construct.cuh"

namespace jasper {

// Variant of robust_prune_kernel for the scale-merge pass.
//
// Pruning is only applied when the selected neighbor's vector is accessible
// from either b1_graph or b2_graph.  Candidates whose vector cannot be
// resolved from either partition are skipped entirely.  Results are written
// back into b1_graph, which owns the [batch_offset, batch_offset+batch_size)
// global range.
template <typename CONSTRUCT_GRAPH_CONFIG, uint32_t TILE_SIZE>
__global__ void robust_prune_kernel_scale(
  const typename CONSTRUCT_GRAPH_CONFIG::entry_t* __restrict__ candidates,  // [batch_size * max_candidates]
  const uint32_t* __restrict__ candidate_counts,                             // [batch_size]
  typename CONSTRUCT_GRAPH_CONFIG::graph_t::device_view b1_graph,
  typename CONSTRUCT_GRAPH_CONFIG::graph_t::device_view b2_graph,
  uint32_t dim,
  uint32_t batch_offset,   // global index of the first vector in this batch (within B1)
  uint32_t batch_size,
  uint32_t max_candidates,
  float alpha
) {
  using index_t     = typename CONSTRUCT_GRAPH_CONFIG::index_t;
  using distance_t  = typename CONSTRUCT_GRAPH_CONFIG::distance_t;
  using entry_t     = typename CONSTRUCT_GRAPH_CONFIG::entry_t;
  using data_t      = typename CONSTRUCT_GRAPH_CONFIG::data_t;
  constexpr uint32_t R = CONSTRUCT_GRAPH_CONFIG::R;
  constexpr index_t INVALID_INDEX = std::numeric_limits<index_t>::max();

  uint32_t bid = blockIdx.x;
  if (bid >= batch_size) return;

  auto thread_block = cg::this_thread_block();
  auto my_tile = cg::tiled_partition<TILE_SIZE>(thread_block);
  uint64_t tile_id = my_tile.meta_group_rank();
  uint64_t n_tiles  = my_tile.meta_group_size();

  index_t  global_id = static_cast<index_t>(batch_offset + bid);
  uint32_t n_cands   = candidate_counts[bid];
  const entry_t* my_cands = candidates + bid * max_candidates;

  __shared__ index_t    s_selected[R];
  __shared__ distance_t s_selected_dist[R];
  __shared__ uint32_t   s_n_selected;
  __shared__ bool       s_pruned;

  if (threadIdx.x == 0) s_n_selected = 0;
  __syncthreads();

  for (uint32_t i = 0; i < n_cands; i++) {
    if (s_n_selected >= R) break;

    index_t    cand_id   = my_cands[i].first;
    distance_t cand_dist = my_cands[i].second;

    if (cand_id == global_id || cand_id == INVALID_INDEX) continue;

    // Resolve the candidate vector from whichever partition owns it.
    data_t* cand_vec = nullptr;
    if      (b1_graph.is_valid(cand_id)) cand_vec = b1_graph.get_vector(cand_id);
    else if (b2_graph.is_valid(cand_id)) cand_vec = b2_graph.get_vector(cand_id);
    else {
      // we cant prune this, but we need to keep it
      if (threadIdx.x == 0) {
        uint32_t slot = s_n_selected;
        if (slot < R) {
          s_selected[slot]      = cand_id;
          s_selected_dist[slot] = cand_dist;
          s_n_selected          = slot + 1;
        }
      }
      __syncthreads();
      continue;
    };

    if (threadIdx.x == 0) s_pruned = false;
    __syncthreads();

    uint32_t n_sel = s_n_selected;
    for (uint32_t j = tile_id; j < n_sel; j += n_tiles) {
      index_t sel_id = s_selected[j];

      // Only prune against a selected neighbor whose vector is available in B1 or B2.
      data_t* sel_vec = nullptr;
      if      (b1_graph.is_valid(sel_id)) sel_vec = b1_graph.get_vector(sel_id);
      else if (b2_graph.is_valid(sel_id)) sel_vec = b2_graph.get_vector(sel_id);

      if (sel_vec == nullptr) continue;

      distance_t d = compute_distance<
          CONSTRUCT_GRAPH_CONFIG::graph_cfg_t::dist_func,
          data_t, distance_t, TILE_SIZE>(cand_vec, sel_vec, dim, my_tile);

      if (my_tile.thread_rank() == 0) {
        if (d * alpha < cand_dist) s_pruned = true;
      }
    }
    __syncthreads();

    if (!s_pruned) {
      if (threadIdx.x == 0) {
        uint32_t slot = s_n_selected;
        if (slot < R) {
          s_selected[slot]      = cand_id;
          s_selected_dist[slot] = cand_dist;
          s_n_selected          = slot + 1;
        }
      }
    }
    __syncthreads();
  }

  // Write results back to B1.
  uint32_t final_count = s_n_selected;
  for (uint32_t i = threadIdx.x; i < final_count; i += blockDim.x) {
    b1_graph.set_neighbor(global_id, i, s_selected[i]);
    b1_graph.set_neighbor_dist(global_id, i, s_selected_dist[i]);
  }
  if (threadIdx.x == 0) {
    b1_graph.set_edge_count(global_id, static_cast<uint8_t>(final_count));
  }
}

// Variant of process_reverse_edges_kernel for the scale-merge pass.
//
// Processes reverse edges whose source is a B2 node (identified by
// b2_graph.is_valid).  Sink nodes come from B1, so vector lookups are
// dispatched to whichever of b1_graph / b2_graph owns the global id.
// Results are written back into b2_graph.
template <typename CONSTRUCT_GRAPH_CONFIG, uint32_t TILE_SIZE>
__global__ void process_reverse_edges_kernel_scale(
  const edge_pair<typename CONSTRUCT_GRAPH_CONFIG::index_t>* __restrict__ reverse_edges,
  const typename CONSTRUCT_GRAPH_CONFIG::index_t* __restrict__ reverse_offsets,
  typename CONSTRUCT_GRAPH_CONFIG::graph_t::device_view b1_graph,
  typename CONSTRUCT_GRAPH_CONFIG::graph_t::device_view b2_graph,
  uint32_t dim,
  typename CONSTRUCT_GRAPH_CONFIG::index_t num_vertices,
  float alpha
) {
  using index_t    = typename CONSTRUCT_GRAPH_CONFIG::index_t;
  using distance_t = typename CONSTRUCT_GRAPH_CONFIG::distance_t;
  using entry_t    = typename CONSTRUCT_GRAPH_CONFIG::entry_t;
  using data_t     = typename CONSTRUCT_GRAPH_CONFIG::data_t;

  constexpr uint32_t R          = CONSTRUCT_GRAPH_CONFIG::R;
  constexpr uint32_t BLOCK_SIZE = CONSTRUCT_GRAPH_CONFIG::block_size;
  constexpr uint32_t MAX_CANDS  = CONSTRUCT_GRAPH_CONFIG::beam_search_max_result_size + R;
  constexpr uint32_t ITEMS_PER_THREAD = (MAX_CANDS + BLOCK_SIZE - 1) / BLOCK_SIZE;
  constexpr index_t  INVALID_INDEX    = std::numeric_limits<index_t>::max();
  constexpr entry_t  SENTINEL         = {INVALID_INDEX, std::numeric_limits<distance_t>::max()};

  struct EntryLess {
    __device__ __forceinline__ bool operator()(const entry_t& a, const entry_t& b) const {
      if (a.second != b.second) return a.second < b.second;
      return a.first < b.first;
    }
  };

  using BlockMergeSortT = cub::BlockMergeSort<entry_t, BLOCK_SIZE, ITEMS_PER_THREAD>;
  constexpr size_t cand_bytes   = sizeof(entry_t) * MAX_CANDS;
  constexpr size_t sort_bytes   = sizeof(typename BlockMergeSortT::TempStorage);
  constexpr size_t shared_bytes = (cand_bytes > sort_bytes) ? cand_bytes : sort_bytes;
  __shared__ __align__(alignof(entry_t)) char shared_buf[shared_bytes];
  entry_t* s_candidates = reinterpret_cast<entry_t*>(shared_buf);
  auto& sort_storage    = reinterpret_cast<typename BlockMergeSortT::TempStorage&>(shared_buf);

  __shared__ uint32_t   s_count;
  __shared__ index_t    s_selected[R];
  __shared__ distance_t s_selected_dist[R];
  __shared__ uint32_t   s_n_selected;
  __shared__ bool       s_pruned;

  entry_t items[ITEMS_PER_THREAD];

  auto thread_block = cg::this_thread_block();
  auto my_tile      = cg::tiled_partition<TILE_SIZE>(thread_block);
  uint64_t tile_id  = my_tile.meta_group_rank();
  uint64_t n_tiles  = my_tile.meta_group_size();

  for (index_t vid = blockIdx.x; vid < num_vertices; vid += gridDim.x) {
    index_t  rev_start  = reverse_offsets[vid];
    index_t  rev_end    = reverse_offsets[vid + 1];
    uint32_t n_reverse  = rev_end - rev_start;
    if (n_reverse == 0) continue;

    index_t actual_vertex = reverse_edges[rev_start].source;

    // Only update B2 nodes; skip reverse edges whose source is in B1.
    if (!b2_graph.is_valid(actual_vertex)) continue;

    data_t* src_vec = b2_graph.get_vector(actual_vertex);

    if (threadIdx.x == 0) s_count = 0;
    __syncthreads();

    // Collect existing B2 edges using stored distances.
    uint8_t n_edges = b2_graph.get_edge_count(actual_vertex);
    for (uint32_t i = threadIdx.x; i < n_edges; i += blockDim.x) {
      index_t neighbor = b2_graph.get_neighbor(actual_vertex, i);
      if (neighbor == INVALID_INDEX) continue;
      distance_t d = static_cast<distance_t>(b2_graph.get_neighbor_dist(actual_vertex, i));
      uint32_t slot = atomicAdd(&s_count, 1u);
      if (slot < MAX_CANDS) s_candidates[slot] = {neighbor, d};
    }

    // Collect reverse-edge sinks (B1 nodes) and compute distances.
    for (uint32_t i = tile_id; i < n_reverse; i += n_tiles) {
      index_t sink = reverse_edges[rev_start + i].sink;
      if (sink == INVALID_INDEX || sink == actual_vertex) continue;

      data_t* sink_vec = nullptr;
      if      (b1_graph.is_valid(sink)) sink_vec = b1_graph.get_vector(sink);
      else if (b2_graph.is_valid(sink)) sink_vec = b2_graph.get_vector(sink);
      if (sink_vec == nullptr) continue;

      distance_t d = compute_distance<
          CONSTRUCT_GRAPH_CONFIG::graph_cfg_t::dist_func,
          data_t, distance_t, TILE_SIZE>(src_vec, sink_vec, dim, my_tile);

      if (my_tile.thread_rank() == 0) {
        uint32_t slot = atomicAdd(&s_count, 1u);
        if (slot < MAX_CANDS) s_candidates[slot] = {sink, d};
      }
    }
    __syncthreads();

    uint32_t count = min(s_count, (uint32_t)MAX_CANDS);

    // Block merge sort.
    #pragma unroll
    for (uint32_t i = 0; i < ITEMS_PER_THREAD; i++) {
      uint32_t idx = threadIdx.x * ITEMS_PER_THREAD + i;
      items[i] = (idx < count) ? s_candidates[idx] : SENTINEL;
    }
    __syncthreads();
    BlockMergeSortT(sort_storage).Sort(items, EntryLess());
    __syncthreads();
    #pragma unroll
    for (uint32_t i = 0; i < ITEMS_PER_THREAD; i++) {
      uint32_t idx = threadIdx.x * ITEMS_PER_THREAD + i;
      if (idx < count) s_candidates[idx] = items[i];
    }
    __syncthreads();

    // Robust prune with cross-partition vector lookups.
    if (threadIdx.x == 0) s_n_selected = 0;
    __syncthreads();

    for (uint32_t i = 0; i < count; i++) {
      if (s_n_selected >= R) break;

      index_t    cand_id   = s_candidates[i].first;
      distance_t cand_dist = s_candidates[i].second;
      if (cand_id == actual_vertex || cand_id == INVALID_INDEX) continue;

      data_t* cand_vec = nullptr;
      if      (b1_graph.is_valid(cand_id)) cand_vec = b1_graph.get_vector(cand_id);
      else if (b2_graph.is_valid(cand_id)) cand_vec = b2_graph.get_vector(cand_id);
      if (cand_vec == nullptr) {
        // we cant prune this, but we need to keep it
        if (threadIdx.x == 0) {
          uint32_t slot = s_n_selected;
          if (slot < R) {
            s_selected[slot]      = cand_id;
            s_selected_dist[slot] = cand_dist;
            s_n_selected          = slot + 1;
          }
        }
        __syncthreads();
        continue;
      };

      if (threadIdx.x == 0) s_pruned = false;
      __syncthreads();

      uint32_t n_sel = s_n_selected;
      for (uint32_t j = tile_id; j < n_sel; j += n_tiles) {
        index_t sel_id = s_selected[j];
        data_t* sel_vec = nullptr;
        if      (b1_graph.is_valid(sel_id)) sel_vec = b1_graph.get_vector(sel_id);
        else if (b2_graph.is_valid(sel_id)) sel_vec = b2_graph.get_vector(sel_id);
        if (sel_vec == nullptr) continue;

        distance_t d = compute_distance<
            CONSTRUCT_GRAPH_CONFIG::graph_cfg_t::dist_func,
            data_t, distance_t, TILE_SIZE>(cand_vec, sel_vec, dim, my_tile);

        if (my_tile.thread_rank() == 0) {
          if (d * alpha < cand_dist) s_pruned = true;
        }
      }
      __syncthreads();

      if (!s_pruned) {
        if (threadIdx.x == 0) {
          uint32_t slot = s_n_selected;
          if (slot < R) {
            s_selected[slot]      = cand_id;
            s_selected_dist[slot] = cand_dist;
            s_n_selected          = slot + 1;
          }
        }
      }
      __syncthreads();
    }

    // Write back to B2's graph.
    uint32_t final_count = s_n_selected;
    for (uint32_t i = threadIdx.x; i < final_count; i += blockDim.x) {
      b2_graph.set_neighbor(actual_vertex, i, s_selected[i]);
      b2_graph.set_neighbor_dist(actual_vertex, i, s_selected_dist[i]);
    }
    if (threadIdx.x == 0) {
      b2_graph.set_edge_count(actual_vertex, static_cast<uint8_t>(final_count));
    }
    __syncthreads();
  }
}

}