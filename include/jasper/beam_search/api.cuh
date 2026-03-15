#pragma once

#include "jasper/beam_search/beam_search.cuh"
#include "jasper/beam_search/config.cuh"

namespace jasper {

struct search_params {
  uint32_t k          = 10;
  uint32_t beam_width = 64;
  uint32_t limit      = 512;
  bool     get_visited = false;
};

template <typename GRAPH_CFG,
          uint32_t BLOCK_SIZE,
          uint32_t MAX_SEARCH_WIDTH,
          uint32_t TILE_SIZE,
          uint32_t MAX_RESULT_SIZE,
          bool GET_VISITED>
beam_search_result<GRAPH_CFG> search_impl(const graph<GRAPH_CFG>& g,
                 typename GRAPH_CFG::vector_view_t&  d_queries,
                 const search_params&                params) {

  using Cfg = beam_search_config<
      GRAPH_CFG,
      GRAPH_CFG::dist_func,
      BLOCK_SIZE,
      GET_VISITED,
      MAX_SEARCH_WIDTH,
      TILE_SIZE,
      MAX_RESULT_SIZE>;

  beam_search_params<Cfg> bp {
    .graph          = g,
    .data_vectors   = g.vectors,
    .query_vectors  = d_queries,
    .medoid         = g.medoid,
    .k              = params.k,
    .beam_width     = params.beam_width,
    .limit          = params.limit,
  };

  return beam_search<Cfg>(bp);
}

// template <typename GRAPH_CFG,
//           uint32_t BLOCK_SIZE = 64,
//           uint32_t MAX_SEARCH_WIDTH = 512,
//           uint32_t TILE_SIZE = 4,
//           uint32_t MAX_RESULT_SIZE = 1024>
// beam_search_result<GRAPH_CFG> search(const graph<GRAPH_CFG>& g,
//             typename GRAPH_CFG::vector_view_t                d_queries,
//             const search_params&                             params = {}) {
//   if (params.get_visited) {
//     return search_impl<GRAPH_CFG, BLOCK_SIZE, MAX_SEARCH_WIDTH,
//                        TILE_SIZE, MAX_RESULT_SIZE, true>(
//         g, d_queries, params);
//   } else {
//     return search_impl<GRAPH_CFG, BLOCK_SIZE, MAX_SEARCH_WIDTH,
//                        TILE_SIZE, MAX_RESULT_SIZE, false>(
//         g, d_queries, params);
//   }
// }

template <typename GRAPH_CFG,
          uint32_t BLOCK_SIZE,
          uint32_t MAX_SEARCH_WIDTH,
          uint32_t TILE_SIZE,
          uint32_t MAX_RESULT_SIZE,
          bool GET_VISITED>
beam_search_result<GRAPH_CFG> search_dispatch_visited(
    const graph<GRAPH_CFG>&               g,
    typename GRAPH_CFG::vector_view_t     d_queries,
    const search_params&                  params) {
  return search_impl<GRAPH_CFG, BLOCK_SIZE, MAX_SEARCH_WIDTH,
                     TILE_SIZE, MAX_RESULT_SIZE, GET_VISITED>(
      g, d_queries, params);
}

template <typename GRAPH_CFG,
          uint32_t BLOCK_SIZE,
          uint32_t MAX_SEARCH_WIDTH,
          uint32_t TILE_SIZE,
          uint32_t MAX_RESULT_SIZE>
beam_search_result<GRAPH_CFG> search_dispatch_width(
    const graph<GRAPH_CFG>&               g,
    typename GRAPH_CFG::vector_view_t     d_queries,
    const search_params&                  params) {
  if (params.get_visited) {
    return search_dispatch_visited<GRAPH_CFG, BLOCK_SIZE, MAX_SEARCH_WIDTH,
                                   TILE_SIZE, MAX_RESULT_SIZE, true>(
        g, d_queries, params);
  } else {
    return search_dispatch_visited<GRAPH_CFG, BLOCK_SIZE, MAX_SEARCH_WIDTH,
                                   TILE_SIZE, MAX_RESULT_SIZE, false>(
        g, d_queries, params);
  }
}

template <typename GRAPH_CFG,
          uint32_t BLOCK_SIZE = 64,
          uint32_t TILE_SIZE = 4,
          uint32_t MAX_RESULT_SIZE = 1024>
beam_search_result<GRAPH_CFG> search(
    const graph<GRAPH_CFG>&               g,
    typename GRAPH_CFG::vector_view_t     d_queries,
    const search_params&                  params = {}) {
  const uint32_t bw = params.beam_width;

  // Pick the smallest value from {128, 256, 512, 1024}
  // such that beam_width + 64 < MAX_SEARCH_WIDTH.
  if (bw + 64 < 128) {
    return search_dispatch_width<GRAPH_CFG, BLOCK_SIZE, 128,
                                 TILE_SIZE, MAX_RESULT_SIZE>(
        g, d_queries, params);
  } else if (bw + 64 < 256) {
    return search_dispatch_width<GRAPH_CFG, BLOCK_SIZE, 256,
                                 TILE_SIZE, MAX_RESULT_SIZE>(
        g, d_queries, params);
  } else if (bw + 64 < 512) {
    return search_dispatch_width<GRAPH_CFG, BLOCK_SIZE, 512,
                                 TILE_SIZE, MAX_RESULT_SIZE>(
        g, d_queries, params);
  } else if (bw + 64 < 1024) {
    return search_dispatch_width<GRAPH_CFG, BLOCK_SIZE, 1024,
                                 TILE_SIZE, MAX_RESULT_SIZE>(
        g, d_queries, params);
  } else {
    throw std::invalid_argument(
        "beam_width " + std::to_string(bw) +
        " is too large (beam_width + 64 must be < 1024)");
  }
}

}