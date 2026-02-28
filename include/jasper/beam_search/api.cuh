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

template <typename GraphCfg,
          uint32_t BLOCK_SIZE,
          uint32_t MAX_SEARCH_WIDTH,
          uint32_t TILE_SIZE,
          uint32_t MAX_RESULT_SIZE,
          bool GET_VISITED>
BeamSearchResult<GraphCfg> search_impl(const graph<GraphCfg>&             g,
                 typename GraphCfg::vector_view_t&  d_queries,
                 const SearchParams&                params) {

  using Cfg = BeamSearchConfig<
      GraphCfg,
      GraphCfg::dist_func,
      BLOCK_SIZE,
      GET_VISITED,
      MAX_SEARCH_WIDTH,
      TILE_SIZE,
      MAX_RESULT_SIZE>;

  BeamSearchParams<Cfg> bp {
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

template <typename GraphCfg,
          uint32_t BLOCK_SIZE = 64,
          uint32_t MAX_SEARCH_WIDTH = 512,
          uint32_t TILE_SIZE = 4,
          uint32_t MAX_RESULT_SIZE = 1024>
BeamSearchResult<GraphCfg> search(const graph<GraphCfg>&                          g,
            typename GraphCfg::vector_view_t                d_queries,
            const SearchParams&                             params = {}) {
  if (params.get_visited) {
    return search_impl<GraphCfg, BLOCK_SIZE, MAX_SEARCH_WIDTH,
                       TILE_SIZE, MAX_RESULT_SIZE, true>(
        g, d_queries, params);
  } else {
    return search_impl<GraphCfg, BLOCK_SIZE, MAX_SEARCH_WIDTH,
                       TILE_SIZE, MAX_RESULT_SIZE, false>(
        g, d_queries, params);
  }
}

}