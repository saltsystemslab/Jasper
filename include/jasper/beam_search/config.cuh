#pragma once

#include <thrust/pair.h>

#include "jasper/distance/distance.cuh"
#include "jasper/index/vector.cuh"

namespace jasper {

template <typename GRAPH_CFG,
          distance_func DISTANCE_FUNC,
          uint32_t BLOCK_SIZE = 128,
          uint32_t MAX_SEARCH_WIDTH = 512,
          uint32_t TILE_SIZE = 4>
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
  static constexpr uint32_t max_search_width  = MAX_SEARCH_WIDTH;
  static constexpr uint32_t tile_size         = TILE_SIZE;
};

template <typename Cfg>
struct beam_search_params {
  // Graph
  typename Cfg::graph_t       graph;

  // Query vectors
  // if use_range, the query vector will be selected from the graph [start, end).
  // otherwise use query_vectors.
  typename Cfg::vector_view_t query_vectors;

  bool use_range = false;
  typename Cfg::index_t query_start;
  typename Cfg::index_t query_end;

  // Search settings
  typename Cfg::index_t       medoid;
  uint32_t                    k;
  uint32_t                    beam_width;
  uint32_t                    limit;

  // Visited-list bookkeeping. Neither of these affects kernel codegen (no
  // shared/register array or CUB template is sized off them), so they are
  // plain runtime knobs rather than template parameters.
  bool                        get_visited     = false;
  uint32_t                    max_result_size = 1024;

  // Adaptive Beam Search termination (Al-Jazzazi et al. 2025, eq. 3). Negative
  // (the default) = off: search stops once the beam_width best candidates are
  // all expanded. γ ≥ 0 also stops as soon as the next candidate is at least
  // (1+γ)× farther than the k-th best result so far, so beam_width becomes a
  // cap on the candidate list rather than the stopping rule. Only the
  // directional kernel (graph_beam_search.cuh) implements it. A runtime knob
  // like get_visited, not a template axis.
  float                       adaptive_gamma  = -1.0f;
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