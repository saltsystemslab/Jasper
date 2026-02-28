#pragma once

#include "jasper/distance/distance.cuh"
#include "jasper/index/vector.cuh"

namespace jasper {

template <typename GRAPH_CFG,
          distance_func DISTANCE_FUNC,
          uint32_t BLOCK_SIZE = 128,
          bool GET_VISITED = true,
          uint32_t MAX_SEARCH_WIDTH = 512,
          uint32_t TILE_SIZE = 4,
          uint32_t MAX_RESULT_SIZE=1024>
struct beam_search_config {
  using graph_cfg_t  = GRAPH_CFG;
  using index_t       = typename GRAPH_CFG::index_t;
  using data_t        = typename GRAPH_CFG::data_t;
  using distance_t    = typename GRAPH_CFG::distance_t;
  using graph_t       = graph<GRAPH_CFG>;
  using entry_t       = thrust::pair<index_t, distance_t>;
  using vector_view_t = vector_view<data_t>;

  static constexpr distance_func dist_func    = DISTANCE_FUNC;
  static constexpr uint32_t block_size        = BLOCK_SIZE;
  static constexpr bool     get_visited       = GET_VISITED;
  static constexpr uint32_t max_search_width  = MAX_SEARCH_WIDTH;
  static constexpr uint32_t tile_size         = TILE_SIZE;
  static constexpr uint32_t max_result_size   = MAX_RESULT_SIZE;
};

template <typename Cfg>
struct beam_search_params {
  // Graph
  typename Cfg::graph_t       graph;

  // Vectors
  typename Cfg::vector_view_t data_vectors;
  typename Cfg::vector_view_t query_vectors;

  // Search settings
  typename Cfg::index_t       medoid;
  uint32_t                    k;
  uint32_t                    beam_width;
  uint32_t                    limit;
};

template <typename GRAPH_CFG>
struct beam_search_result {
  using index_t       = typename GRAPH_CFG::index_t;
  using distance_t    = typename GRAPH_CFG::distance_t;
  using entry_t = thrust::pair<index_t, distance_t>;
  entry_t* frontier;   // [n_query_vectors * k]
  entry_t* visited;    // [n_query_vectors * max_result_size] (if get_visited)
  uint32_t*              visited_counts; // [n_query_vectors] (if get_visited)
};

}