#pragma once

#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <vector>
#include <thread>
#include <exception>
#include <numeric>
#include <algorithm>
#include <fstream>

#include "jasper/index/construct.cuh"
#include "jasper/index/construct_scale_kernels.cuh"

#ifndef JASPER_DEBUG_SCALE_CONSTRUCT
#define JASPER_DEBUG_SCALE_CONSTRUCT 0
#endif

namespace jasper {

// Calculate the partition size that fits two partitions within the given memory budget.
// Formula: partition_size * (index_size_per_vec + workspace_size_per_vec) * 2 < budget_bytes
// index_size_per_vec  = edge storage + vector data per node
// workspace_size_per_vec = per-element construction workspace cost 
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
template <typename GRAPH_CFG, typename GRAPH_CONSTRUCT_CONFIG>
struct intermediate_graph {
  using partition_graph_t = graph<GRAPH_CFG>;

  std::vector<partition_graph_t> partitions;
  size_t n_partitions;
  size_t partition_size;
  size_t workspace_budget_per_partition;
  float alpha;

  // Allocate and fully construct one graph per partition with explicit sizing.
  // n_parts, part_size, and workspace_budget are provided directly by the caller.
  // GRAPH_CONSTRUCT_CONFIG::graph_cfg_t must equal GRAPH_CFG.
  static intermediate_graph allocate_and_construct(
    graph_construct_params<GRAPH_CONSTRUCT_CONFIG> params,
    size_t n_parts,
    size_t workspace_budget
  ) {
    static_assert(
        std::is_same<typename GRAPH_CONSTRUCT_CONFIG::graph_cfg_t, GRAPH_CFG>::value,
        "GRAPH_CONSTRUCT_CONFIG::graph_cfg_t must match GRAPH_CFG");

    size_t n_vectors = params.data_vectors.n_vectors;

    assert(n_parts > 0 && n_vectors > 0);
    size_t part_size = (n_vectors + n_parts - 1) / n_parts;

    uint32_t batch_size =
        graph_construct_workspace<GRAPH_CONSTRUCT_CONFIG>::max_batch_size_for_budget(
            workspace_budget);
    batch_size = std::min(batch_size, params.max_batch_size);

    intermediate_graph ig;
    ig.partition_size                 = part_size;
    ig.workspace_budget_per_partition = workspace_budget;
    ig.n_partitions                   = n_parts;
    ig.alpha                          = params.alpha;
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

    return allocate_and_construct(
        params, n_parts, workspace_budget);
  }

  __host__ void merge_partition_batch(
    partition_graph_t &B1_graph,
    partition_graph_t &B2_graph,
    uint32_t count,
    uint32_t batch_size,
    graph_construct_workspace<GRAPH_CONSTRUCT_CONFIG>& ws
  ) {
    using entry_t     = typename GRAPH_CONSTRUCT_CONFIG::entry_t;
    using index_t     = typename GRAPH_CONSTRUCT_CONFIG::index_t;
    using graph_cfg_t = typename GRAPH_CONSTRUCT_CONFIG::graph_cfg_t;
    constexpr uint32_t L          = GRAPH_CONSTRUCT_CONFIG::L;
    constexpr uint32_t R          = GRAPH_CONSTRUCT_CONFIG::R;
    constexpr uint32_t max_result = GRAPH_CONSTRUCT_CONFIG::beam_search_max_result_size;
    uint32_t max_candidates = max_result + R;
    uint32_t dim = B1_graph.dim;

    constexpr uint32_t BLOCK_SIZE = GRAPH_CONSTRUCT_CONFIG::block_size;
    constexpr uint32_t TILE_SIZE  = GRAPH_CONSTRUCT_CONFIG::tile_size;

    constexpr uint32_t beam_search_tile_size  = 4;
    constexpr uint32_t beam_search_block_size = 64;
    using beam_search_cfg = beam_search_config<
      graph_cfg_t, graph_cfg_t::dist_func,
      beam_search_block_size,
      true,
      GRAPH_CONSTRUCT_CONFIG::beam_search_max_search_width,
      beam_search_tile_size,
      GRAPH_CONSTRUCT_CONFIG::beam_search_max_result_size>;

    // Beam search: B1 batch queries → B2 graph.
    // The batch lies within one segment because partition_size <= vectors_per_segment.
    using segment_t = graph_segment<graph_cfg_t>;
    std::vector<segment_t> h_segs_b1(B1_graph.segments.begin(),
                                     B1_graph.segments.end());
    const uint32_t b1_seg   = graph<graph_cfg_t>::segment_of(count);
    const uint32_t b1_local = graph<graph_cfg_t>::local_of(count);
    auto query_view = h_segs_b1[b1_seg].vectors.subview(b1_local, batch_size);

    beam_search_params<beam_search_cfg> bp {
      .graph         = B2_graph,
      .query_vectors = query_view,
      .medoid        = B2_graph.medoid,
      .k             = L,
      .beam_width    = L,
      .limit         = GRAPH_CONSTRUCT_CONFIG::BEAM_SEARCH_LIMIT,
    };
    beam_search_result<graph_cfg_t> bs_result;
    bs_result.frontier       = ws.frontier;
    bs_result.visited        = ws.visited;
    bs_result.visited_counts = ws.visited_counts;
    beam_search<beam_search_cfg>(bp, bs_result);

#if JASPER_DEBUG_SCALE_CONSTRUCT
    // Debug: print first few beam search results
    {
      constexpr uint32_t dbg_limit = 5;
      uint32_t n_print = std::min(batch_size, dbg_limit);
      uint32_t dbg_b1_offset = static_cast<uint32_t>(partitions[B1].global_offset) + count;
      std::vector<entry_t> h_vis(static_cast<size_t>(n_print) * max_result);
      std::vector<uint32_t> h_vcounts(n_print);
      cudaMemcpy(h_vis.data(), bs_result.visited,
                 static_cast<size_t>(n_print) * max_result * sizeof(entry_t),
                 cudaMemcpyDeviceToHost);
      cudaMemcpy(h_vcounts.data(), bs_result.visited_counts,
                 n_print * sizeof(uint32_t), cudaMemcpyDeviceToHost);
      std::printf("[bs_result] b1_offset=%u count=%u\n", dbg_b1_offset, count);
      for (uint32_t i = 0; i < n_print; i++) {
        uint32_t nc = std::min(h_vcounts[i], 8u);
        std::printf("  q%u (%u results):", dbg_b1_offset + i, h_vcounts[i]);
        for (uint32_t j = 0; j < nc; j++) {
          const auto& e = h_vis[static_cast<size_t>(i) * max_result + j];
          std::printf(" (%u,%.3f)", e.first, e.second);
        }
        std::printf("\n");
      }
    }
#endif

    uint32_t b1_batch_offset = static_cast<uint32_t>(B1_graph.global_offset) + count;
    auto b1_view = B1_graph.view();
    auto b2_view = B2_graph.view();

    // Merge beam-search results into B1's candidate list.
    merge_candidates_kernel<GRAPH_CONSTRUCT_CONFIG, TILE_SIZE, BLOCK_SIZE>
        <<<batch_size, beam_search_block_size>>>(
            ws.visited,
            ws.visited_counts,
            b1_view,
            dim,
            b1_batch_offset,
            batch_size,
            max_result,
            max_candidates,
            ws.prune_candidates,
            ws.prune_counts);

    // Robust prune for B1 (resolves vectors from either partition).
    robust_prune_kernel_scale<GRAPH_CONSTRUCT_CONFIG, TILE_SIZE>
        <<<batch_size, beam_search_block_size>>>(
            ws.prune_candidates,
            ws.prune_counts,
            b1_view,
            b2_view,
            dim,
            b1_batch_offset,
            batch_size,
            max_candidates,
            alpha);

#if JASPER_DEBUG_SCALE_CONSTRUCT
    // Debug: print first few B1 neighbor lists after robust prune
    {
      using edge_list_t = typename graph_cfg_t::edge_list_t;
      constexpr uint32_t dbg_limit = 5;
      uint32_t n_print = std::min(batch_size, dbg_limit);
      std::vector<edge_list_t> h_edges(n_print);
      std::vector<uint8_t> h_ecounts(n_print);
      cudaMemcpy(h_edges.data(), h_segs_b1[b1_seg].edges + b1_local,
                 n_print * sizeof(edge_list_t), cudaMemcpyDeviceToHost);
      cudaMemcpy(h_ecounts.data(), h_segs_b1[b1_seg].edge_counts + b1_local,
                 n_print * sizeof(uint8_t), cudaMemcpyDeviceToHost);
      std::printf("[B1 post-prune] b1_offset=%u\n", b1_batch_offset);
      for (uint32_t i = 0; i < n_print; i++) {
        std::printf("  vec %u (%u neighbors):", b1_batch_offset + i, (uint32_t)h_ecounts[i]);
        for (uint32_t j = 0; j < h_ecounts[i]; j++)
          std::printf(" (%u,%.3f)", h_edges[i].edges[j], h_edges[i].dist[j]);
        std::printf("\n");
      }
    }
#endif

    // Collect reverse edges from B1's updated neighbor lists.
    uint32_t total_slots = batch_size * R;
    dim3 grid((total_slots + BLOCK_SIZE - 1) / BLOCK_SIZE);
    fill_reverse_edges<GRAPH_CONSTRUCT_CONFIG><<<grid, BLOCK_SIZE>>>(
        b1_view,
        ws.reverse_edges_ptr,
        batch_size,
        b1_batch_offset,
        R);

    // Sort all reverse edges by source global id.
    thrust::sort(
        thrust::device,
        ws.reverse_edges.begin(),
        ws.reverse_edges.begin() + total_slots,
        semi_sort_compare_edge_pair<index_t>{});

    // Trim sentinel entries (invalid neighbors sort to the end).
    edge_pair<index_t> sentinel = edge_pair<index_t>::sentinel();
    auto it = thrust::lower_bound(
        thrust::device,
        ws.reverse_edges.begin(),
        ws.reverse_edges.begin() + total_slots,
        sentinel,
        semi_sort_compare_edge_pair<index_t>{});
    index_t n_rev_edges = static_cast<index_t>(it - ws.reverse_edges.begin());

    if (n_rev_edges == 0) return;

    // Compute per-source offsets over the sorted edge array.
    index_t num_unique_sources = 0;
    get_unique_source_offsets<GRAPH_CONSTRUCT_CONFIG>(
        ws.reverse_edges,
        n_rev_edges,
        ws.reverse_offsets_flags,
        ws.reverse_offsets,
        num_unique_sources);

    if (num_unique_sources == 0) return;

    // Merge reverse edges into B2's neighbor lists.
    // Entries whose source is not in B2's range are skipped by the kernel.
    constexpr uint64_t process_kernel_grid_size = 1024;
    process_reverse_edges_kernel_scale<GRAPH_CONSTRUCT_CONFIG, TILE_SIZE>
        <<<process_kernel_grid_size, BLOCK_SIZE>>>(
            ws.reverse_edges_ptr,
            ws.reverse_offsets_ptr,
            b1_view,
            b2_view,
            dim,
            num_unique_sources,
            alpha);

    cudaDeviceSynchronize();

#if JASPER_DEBUG_SCALE_CONSTRUCT
    // Debug: print first few B2 neighbor lists after reverse edge insertion
    {
      using edge_list_t = typename graph_cfg_t::edge_list_t;
      constexpr uint32_t dbg_limit = 5;
      uint32_t b2_n_print = std::min((uint32_t)partitions[B2].n_vectors, dbg_limit);
      if (b2_n_print > 0) {
        std::vector<segment_t> h_segs_b2(partitions[B2].segments.begin(),
                                         partitions[B2].segments.end());
        std::vector<edge_list_t> h_edges(b2_n_print);
        std::vector<uint8_t> h_ecounts(b2_n_print);
        cudaMemcpy(h_edges.data(), h_segs_b2[0].edges,
                   b2_n_print * sizeof(edge_list_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_ecounts.data(), h_segs_b2[0].edge_counts,
                   b2_n_print * sizeof(uint8_t), cudaMemcpyDeviceToHost);
        uint32_t b2_offset = static_cast<uint32_t>(partitions[B2].global_offset);
        std::printf("[B2 post-reverse] b2_offset=%u\n", b2_offset);
        for (uint32_t i = 0; i < b2_n_print; i++) {
          std::printf("  vec %u (%u neighbors):", b2_offset + i, (uint32_t)h_ecounts[i]);
          for (uint32_t j = 0; j < h_ecounts[i]; j++)
            std::printf(" (%u,%.3f)", h_edges[i].edges[j], h_edges[i].dist[j]);
          std::printf("\n");
        }
      }
    }
#endif
  }

  // Merge two partitions together with our special sauce beam search + prune
  // Here we assume that both partitions are already loaded onto device memory.
  __host__ void merge_partition(
    partition_graph_t &B1_graph,
    partition_graph_t &B2_graph,
    uint32_t batch_size,
    graph_construct_workspace<GRAPH_CONSTRUCT_CONFIG>& ws
  ) {
    uint32_t count = 0;
    size_t n_vectors = B1_graph.n_vectors;

    while (count < n_vectors) {
      const uint32_t b1_local = graph<partition_graph_t>::local_of(count);
      const uint32_t seg_remaining = graph<partition_graph_t>::vectors_per_segment - b1_local;
      size_t cur_batch_size = std::min({(size_t)batch_size, n_vectors - count, (size_t)seg_remaining});
      merge_partition_batch(B1_graph, B2_graph, count, cur_batch_size, ws);
      count += cur_batch_size;
      std::printf("\r[merge_partition] %u / %u (%.1f%%)", count, static_cast<unsigned int>(n_vectors),
                100.0f * count / n_vectors);
      std::fflush(stdout);
    }
    std::printf("\n");
  }

  // Produce a merge order sequence and call merge_partition for each pair.
  //
  // Enumerates all unordered pairs (i, j) with i < j < n_partitions in an
  // order that preserves memory locality: within a cycle, consecutive
  // merge calls share one partition, so the resident partition can stay
  // resident across calls.
  __host__ void merge() {
    if (n_partitions <= 1) return;

    // Allocate merge workspace
    uint32_t batch_size =
        graph_construct_workspace<GRAPH_CONSTRUCT_CONFIG>::max_batch_size_for_budget(
            workspace_budget_per_partition);
    std::cout << "[merge] allocating workspace." << std::endl;
    auto ws = graph_construct_workspace<GRAPH_CONSTRUCT_CONFIG>::allocate(batch_size);
    std::cout << "[merge] Finished allocating workspace." << std::endl;

    const size_t P = n_partitions;
    const size_t total_pairs = P * (P - 1) / 2;
    size_t B1 = 0;
    size_t merge_count = 0;

    std::cout << "[merge] allocating device memory graphs." << std::endl;
    partition_graph_t B1_graph = partition_graph_t::allocate(
      partitions[B1].dim, partitions[B1].n_vectors, false);
    partition_graph_t B2_graph = partition_graph_t::allocate(
      partitions[B1].dim, partitions[B1].n_vectors, false);
    std::cout << "[merge] Finished allocating device memory graphs." << std::endl;

    partition_graph_t* resident = &B1_graph;
    partition_graph_t* incoming = &B2_graph;

    partitions[B1].copy_to(*resident);

    for (size_t itv = 1; itv <= P / 2; ++itv) {
      const size_t g = std::gcd(itv, P);
      const size_t cycle_len = P / g;

      const size_t calls_per_cycle = (cycle_len == 2) ? 1 : cycle_len;

      for (size_t cycle_idx = 0; cycle_idx < g; ++cycle_idx) {
        if (cycle_idx > 0) {
          // move old B1 to host
          resident->copy_to(partitions[B1]);
          // Switch residue class. (old_B1, new_B1) was already merged
          // at itv=1, so no merge call here -- just slide B1.
          B1 = (B1 + 1) % P;
          // move new B1 to device
          partitions[B1].copy_to(*resident);
        }
        for (size_t step = 0; step < calls_per_cycle; ++step) {
          const size_t B2 = (B1 + itv) % P;
          partitions[B2].copy_to(*incoming);
          std::printf("[merge] %zu / %zu: partitions %zu, %zu (itv=%zu)\n",
                      ++merge_count, total_pairs, B1, B2, itv);
          merge_partition(*resident, *incoming, batch_size, ws);
          resident->copy_to(partitions[B1]);

          using std::swap;
          B1 = B2;
          swap(resident, incoming); 
        }
      }
    }

    resident->copy_to(partitions[B1]); // offload the last resident partition
  }

  // Average out-degree across all partitions, computed without materializing
  // the concatenated graph. Partitions are assumed to be on host pinned memory.
  __host__ double avg_degree() const {
    using segment_t = graph_segment<GRAPH_CFG>;
    uint64_t total_edges   = 0;
    uint64_t total_vectors = 0;
    for (const auto& p : partitions) {
      std::vector<segment_t> p_segs(p.segments.begin(), p.segments.end());
      for (uint32_t ps = 0; ps < p.n_segments; ps++) {
        const segment_t& seg = p_segs[ps];
        uint32_t count = static_cast<uint32_t>(seg.n_vectors);
        for (uint32_t i = 0; i < count; i++) total_edges += seg.edge_counts[i];
        total_vectors += count;
      }
    }
    return total_vectors == 0
        ? 0.0
        : static_cast<double>(total_edges) / static_cast<double>(total_vectors);
  }

  // Stream the concatenated graph straight to the output file, one partition at
  // a time, freeing each partition as soon as it is written. This fuses the old
  // concat() + save_graph_to_file() steps so we never hold a second full copy of
  // the graph in memory (which would double peak host usage).
  //
  // The on-disk format is a flat sequence of nodes in global index order, so the
  // VPS-segment repacking that concat() performed is irrelevant here: partitions
  // are already laid out in global order (partition p covers globals
  // [p*part_size, ...) with global offsets already applied to edges), so writing
  // them in sequence reproduces the exact same file concat() would have produced.
  __host__ void save_to_file(const std::string& output_fname) {
    using segment_t   = graph_segment<GRAPH_CFG>;
    using data_t      = typename GRAPH_CFG::data_t;
    using edge_list_t = typename GRAPH_CFG::edge_list_t;
    using index_t     = typename GRAPH_CFG::index_t;
    using vector_view_t = typename GRAPH_CFG::vector_view_t;

    index_t total_vectors = 0;
    for (auto& p : partitions)
      total_vectors += static_cast<index_t>(p.n_vectors);

    uint32_t dim        = partitions[0].dim;
    uint32_t padded_dim = vector_view_t::pad(dim);

    std::ofstream outFile(output_fname, std::ios::binary);
    if (!outFile.is_open()) {
      throw std::runtime_error("Failed to open " + output_fname + " for graph save");
    }

    uint64_t big_n_vectors    = static_cast<uint64_t>(total_vectors);
    uint64_t medoid_as_uint64 = static_cast<uint64_t>(partitions[0].medoid);  // p0 offset==0
    uint64_t bytes_per_node   = sizeof(data_t) * dim + sizeof(uint8_t) + sizeof(edge_list_t);
    uint64_t total_file_size  = 4 * sizeof(uint64_t) + big_n_vectors * bytes_per_node;

    outFile.write(reinterpret_cast<const char*>(&total_file_size),  sizeof(uint64_t));
    outFile.write(reinterpret_cast<const char*>(&big_n_vectors),    sizeof(uint64_t));
    outFile.write(reinterpret_cast<const char*>(&medoid_as_uint64), sizeof(uint64_t));
    outFile.write(reinterpret_cast<const char*>(&bytes_per_node),   sizeof(uint64_t));

    for (auto& p : partitions) {
      std::vector<segment_t> p_segs(p.segments.begin(), p.segments.end());
      for (uint32_t ps = 0; ps < p.n_segments; ps++) {
        const segment_t& seg = p_segs[ps];
        uint32_t count = static_cast<uint32_t>(seg.n_vectors);
        for (uint32_t i = 0; i < count; i++) {
          outFile.write(reinterpret_cast<const char*>(seg.vectors.data + static_cast<size_t>(i) * padded_dim),
                        sizeof(data_t) * dim);
          outFile.write(reinterpret_cast<const char*>(&seg.edge_counts[i]),
                        sizeof(uint8_t));
          outFile.write(reinterpret_cast<const char*>(&seg.edges[i]),
                        sizeof(edge_list_t));
        }
      }
      p.deallocate();  // free as we go so peak host memory never doubles
    }

    // File omits the slot_to_id/deleted columns (legacy bytes_per_node), which
    // is fine here: fresh construction has no deletions yet and no prior
    // stable ids to preserve, and load_graph_from_file's legacy-format path
    // reconstructs exactly this state (identity ids, zero deletions,
    // next_id == n_vectors) — see its bpn_base fallback.
    outFile.close();

    std::cout << "Saved graph to " << output_fname << "\n"
              << "  n_vectors       : " << big_n_vectors << "\n"
              << "  medoid          : " << medoid_as_uint64 << "\n"
              << "  dim             : " << dim << "\n";
  }

};

}
