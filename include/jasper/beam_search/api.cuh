#pragma once

#include "jasper/beam_search/beam_search.cuh"
#include "jasper/beam_search/config.cuh"

namespace jasper {

// Runtime search parameters — the stuff users actually change between calls
struct SearchParams {
  uint32_t k          = 10;
  uint32_t beam_width = 64;
  uint32_t limit      = 512;
  bool     get_visited = false;
};

// ── The user-facing search function ────────────────────────────
// All compile-time stuff is baked into graph_config.
// The user just passes a graph, queries, and search params.
//
// Usage:
//
//   using cfg = jasper::graph_config<
//       uint32_t, 64, float, 128, float, jasper::DistanceFunc::L2>;
//
//   auto g = jasper::load_graph_from_file<cfg>("graph.bin");
//
//   auto results = jasper::search(g, d_queries, n_queries, {
//       .k          = 10,
//       .beam_width = 64,
//       .limit      = 300,
//   });
//
//   // results.frontier[i] = {index, distance} for each query's top-k

template <typename GraphCfg,
          uint32_t BLOCK_SIZE = 64,
          uint32_t MAX_SEARCH_WIDTH = 512,
          uint32_t TILE_SIZE = 4,
          uint32_t MAX_RESULT_SIZE = 1024>
auto search(const graph<GraphCfg>&                          g,
            typename GraphCfg::vector_t*                    d_queries,
            uint64_t                                        n_queries,
            const SearchParams&                             params = {}) {

  // Dispatch on get_visited at runtime by branching into two compile-time paths
  if (params.get_visited) {
    return search_impl<GraphCfg, BLOCK_SIZE, MAX_SEARCH_WIDTH,
                       TILE_SIZE, MAX_RESULT_SIZE, true>(
        g, d_queries, n_queries, params);
  } else {
    return search_impl<GraphCfg, BLOCK_SIZE, MAX_SEARCH_WIDTH,
                       TILE_SIZE, MAX_RESULT_SIZE, false>(
        g, d_queries, n_queries, params);
  }
}

// ── Implementation ─────────────────────────────────────────────
template <typename GraphCfg,
          uint32_t BLOCK_SIZE,
          uint32_t MAX_SEARCH_WIDTH,
          uint32_t TILE_SIZE,
          uint32_t MAX_RESULT_SIZE,
          bool GET_VISITED>
auto search_impl(const graph<GraphCfg>&       g,
                 typename GraphCfg::vector_t*  d_queries,
                 uint64_t                      n_queries,
                 const SearchParams&           params) {

  using Cfg = BeamSearchConfig<
      typename GraphCfg::index_t,
      typename GraphCfg::data_t,
      GraphCfg::data_dim,
      typename GraphCfg::distance_t,
      GraphCfg::n_neighbors,
      BLOCK_SIZE,
      GraphCfg::dist_func,
      GET_VISITED,
      MAX_SEARCH_WIDTH,
      TILE_SIZE,
      MAX_RESULT_SIZE>;

  BeamSearchParams<Cfg> bp {
    .graph          = g,
    .data_vectors   = g.vectors,
    .n_data_vectors = g.n_vectors,
    .query_vectors  = d_queries,
    .n_query_vectors = n_queries,
    .medoid         = g.medoid,
    .k              = params.k,
    .beam_width     = params.beam_width,
    .limit          = params.limit,
  };

  return beam_search<Cfg>(bp);
}

}