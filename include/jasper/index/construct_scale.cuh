#pragma once

#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <vector>
#include <thread>
#include <exception>

#include "jasper/index/construct.cuh"

namespace jasper {

// Calculate the partition size that fits two partitions within the given memory budget.
// Formula: partition_size * (index_size_per_vec + workspace_size_per_vec) * 2 < budget_bytes
// index_size_per_vec  = edge storage + vector data per node
// workspace_size_per_vec = per-element construction workspace cost (same accounting as
//                          graph_construct_workspace::max_batch_size_for_budget)
template <typename GRAPH_CONSTRUCT_CONFIG>
__host__ size_t estimate_partition_size_for_budget(size_t budget_bytes, uint32_t dim) {
  using graph_cfg_t = typename GRAPH_CONSTRUCT_CONFIG::graph_cfg_t;
  using data_t      = typename graph_cfg_t::data_t;
  using edge_list_t = typename graph_cfg_t::edge_list_t;
  using index_t     = typename graph_cfg_t::index_t;
  using distance_t  = typename graph_cfg_t::distance_t;
  using entry_t     = thrust::pair<index_t, distance_t>;

  constexpr uint32_t L               = GRAPH_CONSTRUCT_CONFIG::L;
  constexpr uint32_t R               = GRAPH_CONSTRUCT_CONFIG::R;
  constexpr uint32_t max_result_size = GRAPH_CONSTRUCT_CONFIG::beam_search_max_result_size;
  constexpr uint32_t max_candidates  = max_result_size + R;

  uint32_t padded_dim = vector_view<data_t>::pad(dim);

  // Graph index storage per vector: edges + neighbor distance + edge count + vector data
  size_t index_size_per_vec =
      sizeof(edge_list_t) +
      sizeof(float) +
      sizeof(uint8_t) +
      static_cast<size_t>(padded_dim) * sizeof(data_t);

  // Construction workspace per vector (mirrors graph_construct_workspace per-item cost)
  constexpr size_t workspace_size_per_vec =
      sizeof(entry_t)            * L               // frontier
    + sizeof(entry_t)            * max_result_size // visited
    + sizeof(uint32_t)                             // visited_counts
    + sizeof(entry_t)            * max_candidates  // prune_candidates
    + sizeof(uint32_t)                             // prune_counts
    + sizeof(edge_pair<index_t>) * R               // reverse_edges
    + sizeof(index_t)            * R               // reverse_offsets_flags
    + sizeof(index_t)            * R;              // reverse_offsets

  size_t total_per_vec = index_size_per_vec + workspace_size_per_vec;
  if (total_per_vec == 0 || budget_bytes < 2 * total_per_vec) return 0;

  return budget_bytes / (2 * total_per_vec);
}

// Intermediate graph holding one independently-built index per partition.
// Partitions are merged into a larger graph after construction.
template <typename GRAPH_CFG>
struct intermediate_graph {
  using partition_graph_t = graph<GRAPH_CFG>;

  std::vector<partition_graph_t> partitions;
  size_t n_partitions;
  size_t partition_size;
  size_t workspace_budget_per_partition;

  // Allocate and fully construct one graph per partition with explicit sizing.
  // n_parts, part_size, and workspace_budget are provided directly by the caller.
  // GRAPH_CONSTRUCT_CONFIG::graph_cfg_t must equal GRAPH_CFG.
  template <typename GRAPH_CONSTRUCT_CONFIG>
  static intermediate_graph allocate_and_construct(
    graph_construct_params<GRAPH_CONSTRUCT_CONFIG> params,
    size_t n_parts,
    size_t part_size,
    size_t workspace_budget
  ) {
    static_assert(
        std::is_same<typename GRAPH_CONSTRUCT_CONFIG::graph_cfg_t, GRAPH_CFG>::value,
        "GRAPH_CONSTRUCT_CONFIG::graph_cfg_t must match GRAPH_CFG");

    size_t n_vectors = params.data_vectors.n_vectors;

    uint32_t batch_size =
        graph_construct_workspace<GRAPH_CONSTRUCT_CONFIG>::max_batch_size_for_budget(
            workspace_budget);
    batch_size = std::min(batch_size, params.max_batch_size);

    intermediate_graph ig;
    ig.partition_size                 = part_size;
    ig.workspace_budget_per_partition = workspace_budget;
    ig.n_partitions                   = n_parts;
    ig.partitions.resize(n_parts);

    for (size_t p = 0; p < n_parts; p++) {
      size_t start = p * part_size;
      size_t count = std::min(part_size, n_vectors - start);

      std::printf("[construct_scale] partition %zu / %zu (%zu vectors)\n",
                  p + 1, n_parts, count);

      auto sub_view = params.data_vectors.subview(
          static_cast<uint32_t>(start), static_cast<uint32_t>(count));

      graph_construct_params<GRAPH_CONSTRUCT_CONFIG> part_params;
      part_params.data_vectors   = sub_view;
      part_params.alpha          = params.alpha;
      part_params.max_batch_size = batch_size;
      part_params.on_host        = false;  // always construct on device

      ig.partitions[p] = construct_graph<GRAPH_CONSTRUCT_CONFIG>(part_params);
      ig.partitions[p].move_to(true);  // move to host pinned memory after construction
      ig.partitions[p].apply_global_offset(static_cast<typename GRAPH_CFG::index_t>(start));
    }

    return ig;
  }

  // Allocate and fully construct one graph per partition.
  // Derives n_parts, part_size, and workspace_budget from budget_bytes automatically.
  // GRAPH_CONSTRUCT_CONFIG::graph_cfg_t must equal GRAPH_CFG.
  template <typename GRAPH_CONSTRUCT_CONFIG>
  static intermediate_graph allocate_and_construct_for_budget(
    graph_construct_params<GRAPH_CONSTRUCT_CONFIG> params,
    size_t budget_bytes
  ) {
    uint32_t dim     = params.data_vectors.dim;
    size_t n_vectors = params.data_vectors.n_vectors;

    size_t part_size = estimate_partition_size_for_budget<GRAPH_CONSTRUCT_CONFIG>(
        budget_bytes, dim);
    if (part_size == 0) {
      throw std::runtime_error("[construct_scale] budget too small for any partition");
    }

    size_t workspace_budget = budget_bytes / 2;
    size_t n_parts = (n_vectors + part_size - 1) / part_size;

    return allocate_and_construct<GRAPH_CONSTRUCT_CONFIG>(
        params, n_parts, part_size, workspace_budget);
  }
};

}
