#pragma once

#include "jasper/distance/distance.cuh"

namespace jasper {

template <typename INDEX_T,
          typename DATA_T,
          uint16_t DATA_DIM,
          typename DISTANCE_T,
          uint8_t N_NEIGHBORS,
          distance_func DISTANCE_FUNC,
          uint32_t BLOCK_SIZE = 128,
          bool GET_VISITED = true,
          uint32_t MAX_SEARCH_WIDTH = 512,
          uint32_t TILE_SIZE = 4>
struct BeamSearchConfig {
  using index_t    = INDEX_T;
  using data_t     = DATA_T;
  using distance_t = DISTANCE_T;
  using graph_t    = graph<INDEX_T, N_NEIGHBORS>;
  using entry_t    = thrust::pair<INDEX_T, DISTANCE_T>;
  using vector_t   = data_vector<DATA_T, DATA_DIM>;

  static constexpr distance_func dist_func    = DISTANCE_FUNC;
  static constexpr uint16_t data_dim          = DATA_DIM;
  static constexpr uint32_t block_size        = BLOCK_SIZE;
  static constexpr bool     get_visited       = GET_VISITED;
  static constexpr uint32_t max_search_width  = MAX_SEARCH_WIDTH;
  static constexpr uint32_t tile_size         = TILE_SIZE;
  static constexpr uint32_t max_result_size   = 1024;
};

template <typename Cfg>
struct BeamSearchParams {
  // Graph
  typename Cfg::graph_t*     graph;

  // Data
  typename Cfg::vector_t*    data_vectors;
  uint64_t                   n_data_vectors;

  // Queries
  typename Cfg::vector_t*    query_vectors;
  uint64_t                   n_query_vectors;

  // Search settings
  typename Cfg::index_t      medoid;
  uint32_t                   k;
  uint32_t                   beam_width;
  uint32_t                   limit;
};

template <typename Cfg>
struct BeamSearchResult {
  typename Cfg::entry_t* frontier;   // [n_query_vectors * k]
  typename Cfg::entry_t* visited;    // [n_query_vectors * max_result_size] (if get_visited)
  uint32_t*              visited_counts; // [n_query_vectors] (if get_visited)
};

}