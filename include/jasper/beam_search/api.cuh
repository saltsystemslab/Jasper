#pragma once

#include "jasper/beam_search/beam_search.cuh"
#include "jasper/beam_search/directional_beam_search.cuh"
#include "jasper/beam_search/pq_beam_search.cuh"
#include "jasper/beam_search/rq_beam_search.cuh"
#include "jasper/beam_search/config.cuh"
#include "jasper/lsh/lsh_globals.cuh"
#include "jasper/pq/pq_codebooks.cuh"
#include "jasper/rabitq/edge_rq.cuh"

namespace jasper {

struct search_params {
  uint32_t k               = 10;
  uint32_t beam_width       = 64;
  uint32_t limit            = 512;
  bool     get_visited      = false;
  uint32_t max_result_size  = 2048;
  // < 0: stop on beam width. ≥ 0: Adaptive Beam Search with this γ, see
  // beam_search_params::adaptive_gamma. Directional searches only.
  float    adaptive_gamma   = -1.0f;
  // > 0: re-rank this many unexpanded candidates with exact distances when the
  // search stops, see beam_search_params::rerank. Directional searches only.
  uint32_t rerank           = 0;
};

template <typename GRAPH_CFG,
          uint32_t BLOCK_SIZE,
          uint32_t MAX_SEARCH_WIDTH,
          uint32_t TILE_SIZE>
beam_search_result<GRAPH_CFG> search_impl(
    const graph<GRAPH_CFG>&            g,
    typename GRAPH_CFG::vector_view_t& d_queries,
    const search_params&               params) {

  using Cfg = beam_search_config<
      GRAPH_CFG, GRAPH_CFG::dist_func,
      BLOCK_SIZE, MAX_SEARCH_WIDTH, TILE_SIZE>;

  beam_search_params<Cfg> bp {
    .graph           = g,
    .query_vectors   = d_queries,
    .use_range       = false,
    .medoid          = g.medoid,
    .k               = params.k,
    .beam_width      = params.beam_width,
    .limit           = params.limit,
    .get_visited     = params.get_visited,
    .max_result_size = params.max_result_size,
  };
  return beam_search<Cfg>(bp);
}

template <typename GRAPH_CFG, uint32_t BLOCK_SIZE, uint32_t MAX_SEARCH_WIDTH,
          uint32_t TILE_SIZE>
beam_search_result<GRAPH_CFG> search_dispatch_width(
    const graph<GRAPH_CFG>&           g,
    typename GRAPH_CFG::vector_view_t d_queries,
    const search_params&              params) {
  return search_impl<GRAPH_CFG, BLOCK_SIZE, MAX_SEARCH_WIDTH, TILE_SIZE>(
      g, d_queries, params);
}

template <typename GRAPH_CFG,
          uint32_t BLOCK_SIZE = 64,
          uint32_t TILE_SIZE  = 4>
beam_search_result<GRAPH_CFG> search(
    const graph<GRAPH_CFG>&           g,
    typename GRAPH_CFG::vector_view_t d_queries,
    const search_params&              params = {}) {
  // The plain kernel (beam_search.cuh) has only the beam-width rule; refuse
  // rather than silently run something other than what was asked for.
  if (params.adaptive_gamma >= 0.0f)
    throw std::invalid_argument(
        "adaptive_gamma is implemented for directional_search / pq_search / "
        "rq_search only, not for conventional search()");
  if (params.rerank > 0)
    throw std::invalid_argument(
        "rerank is implemented for directional_search / pq_search / "
        "rq_search only, not for conventional search()");
  const uint32_t bw = params.beam_width;
  if      (bw + 64 < 128)  return search_dispatch_width<GRAPH_CFG, BLOCK_SIZE, 128,  TILE_SIZE>(g, d_queries, params);
  else if (bw + 64 < 256)  return search_dispatch_width<GRAPH_CFG, BLOCK_SIZE, 256,  TILE_SIZE>(g, d_queries, params);
  else if (bw + 64 < 512)  return search_dispatch_width<GRAPH_CFG, BLOCK_SIZE, 512,  TILE_SIZE>(g, d_queries, params);
  else if (bw + 64 < 1024) return search_dispatch_width<GRAPH_CFG, BLOCK_SIZE, 1024, TILE_SIZE>(g, d_queries, params);
  else if (bw + 64 < 2048) return search_dispatch_width<GRAPH_CFG, BLOCK_SIZE, 2048, TILE_SIZE>(g, d_queries, params);
  else throw std::invalid_argument(
      "beam_width " + std::to_string(bw) +
      " is too large (beam_width + 64 must be < 2048)");
}

// ─────────────────────────── directional_search ────────────────────────────

template <typename GRAPH_CFG,
          uint32_t BLOCK_SIZE,
          uint32_t MAX_SEARCH_WIDTH,
          uint32_t TILE_SIZE>
beam_search_result<GRAPH_CFG> directional_search_impl(
    const graph<GRAPH_CFG>&                       g,
    const lsh_globals<GRAPH_CFG::k_ranks>&        globals,
    typename GRAPH_CFG::vector_view_t&            d_queries,
    const search_params&                          params) {
  static_assert(GRAPH_CFG::use_lsh,
                "directional_search requires graph_cfg::use_lsh");

  using Cfg = beam_search_config<
      GRAPH_CFG, GRAPH_CFG::dist_func,
      BLOCK_SIZE, MAX_SEARCH_WIDTH, TILE_SIZE>;

  beam_search_params<Cfg> bp {
    .graph           = g,
    .query_vectors   = d_queries,
    .use_range       = false,
    .medoid          = g.medoid,
    .k               = params.k,
    .beam_width      = params.beam_width,
    .limit           = params.limit,
    .get_visited     = params.get_visited,
    .max_result_size = params.max_result_size,
    .adaptive_gamma  = params.adaptive_gamma,
    .rerank          = params.rerank,
  };
  return directional_beam_search<Cfg>(bp, globals);
}

template <typename GRAPH_CFG, uint32_t BLOCK_SIZE, uint32_t MAX_SEARCH_WIDTH,
          uint32_t TILE_SIZE>
beam_search_result<GRAPH_CFG> directional_search_dispatch_width(
    const graph<GRAPH_CFG>&                g,
    const lsh_globals<GRAPH_CFG::k_ranks>& globals,
    typename GRAPH_CFG::vector_view_t      d_queries,
    const search_params&                   params) {
  return directional_search_impl<GRAPH_CFG, BLOCK_SIZE, MAX_SEARCH_WIDTH, TILE_SIZE>(
      g, globals, d_queries, params);
}

template <typename GRAPH_CFG,
          uint32_t BLOCK_SIZE = 128,
          uint32_t TILE_SIZE  = 4>
beam_search_result<GRAPH_CFG> directional_search(
    const graph<GRAPH_CFG>&                g,
    const lsh_globals<GRAPH_CFG::k_ranks>& globals,
    typename GRAPH_CFG::vector_view_t      d_queries,
    const search_params&                   params = {}) {
  static_assert(GRAPH_CFG::use_lsh,
                "directional_search requires graph_cfg::use_lsh");

  // The list holds beam_width + rerank candidates; that picks the tier.
  const uint32_t bw = params.beam_width + params.rerank;
  if      (bw + 64 < 128)  return directional_search_dispatch_width<GRAPH_CFG, BLOCK_SIZE, 128,  TILE_SIZE>(g, globals, d_queries, params);
  else if (bw + 64 < 256)  return directional_search_dispatch_width<GRAPH_CFG, BLOCK_SIZE, 256,  TILE_SIZE>(g, globals, d_queries, params);
  else if (bw + 64 < 512)  return directional_search_dispatch_width<GRAPH_CFG, BLOCK_SIZE, 512,  TILE_SIZE>(g, globals, d_queries, params);
  else if (bw + 64 < 1024) return directional_search_dispatch_width<GRAPH_CFG, BLOCK_SIZE, 1024, TILE_SIZE>(g, globals, d_queries, params);
  else if (bw + 64 < 2048) return directional_search_dispatch_width<GRAPH_CFG, BLOCK_SIZE, 2048, TILE_SIZE>(g, globals, d_queries, params);
  else throw std::invalid_argument(
      "beam_width + rerank " + std::to_string(bw) +
      " is too large (beam_width + rerank + 64 must be < 2048)");
}

// ───────────────────────────────── pq_search ───────────────────────────────
// Same beam-search skeleton as directional_search, but neighbor distances are
// estimated with Product-Quantization ADC instead of cross-polytope LSH.

template <typename GRAPH_CFG,
          uint32_t BLOCK_SIZE,
          uint32_t MAX_SEARCH_WIDTH,
          uint32_t TILE_SIZE>
beam_search_result<GRAPH_CFG> pq_search_impl(
    const graph<GRAPH_CFG>&                                          g,
    pq_codebooks_view<GRAPH_CFG::pq_m, GRAPH_CFG::pq_k>              codebooks,
    typename GRAPH_CFG::vector_view_t&                              d_queries,
    const search_params&                                           params) {
  static_assert(GRAPH_CFG::use_lsh,
                "pq_search requires graph_cfg::use_lsh (directional storage)");

  using Cfg = beam_search_config<
      GRAPH_CFG, GRAPH_CFG::dist_func,
      BLOCK_SIZE, MAX_SEARCH_WIDTH, TILE_SIZE>;

  beam_search_params<Cfg> bp {
    .graph           = g,
    .query_vectors   = d_queries,
    .use_range       = false,
    .medoid          = g.medoid,
    .k               = params.k,
    .beam_width      = params.beam_width,
    .limit           = params.limit,
    .get_visited     = params.get_visited,
    .max_result_size = params.max_result_size,
    .adaptive_gamma  = params.adaptive_gamma,
    .rerank          = params.rerank,
  };
  return pq_beam_search<Cfg>(bp, codebooks);
}

template <typename GRAPH_CFG, uint32_t BLOCK_SIZE, uint32_t MAX_SEARCH_WIDTH,
          uint32_t TILE_SIZE>
beam_search_result<GRAPH_CFG> pq_search_dispatch_width(
    const graph<GRAPH_CFG>&                             g,
    pq_codebooks_view<GRAPH_CFG::pq_m, GRAPH_CFG::pq_k> codebooks,
    typename GRAPH_CFG::vector_view_t                   d_queries,
    const search_params&                                params) {
  return pq_search_impl<GRAPH_CFG, BLOCK_SIZE, MAX_SEARCH_WIDTH, TILE_SIZE>(
      g, codebooks, d_queries, params);
}

template <typename GRAPH_CFG,
          uint32_t BLOCK_SIZE = 128,
          uint32_t TILE_SIZE  = 4>
beam_search_result<GRAPH_CFG> pq_search(
    const graph<GRAPH_CFG>&                             g,
    pq_codebooks_view<GRAPH_CFG::pq_m, GRAPH_CFG::pq_k> codebooks,
    typename GRAPH_CFG::vector_view_t                   d_queries,
    const search_params&                                params = {}) {
  static_assert(GRAPH_CFG::use_lsh,
                "pq_search requires graph_cfg::use_lsh (directional storage)");

  // The list holds beam_width + rerank candidates; that picks the tier.
  const uint32_t bw = params.beam_width + params.rerank;
  if      (bw + 64 < 128)  return pq_search_dispatch_width<GRAPH_CFG, BLOCK_SIZE, 128,  TILE_SIZE>(g, codebooks, d_queries, params);
  else if (bw + 64 < 256)  return pq_search_dispatch_width<GRAPH_CFG, BLOCK_SIZE, 256,  TILE_SIZE>(g, codebooks, d_queries, params);
  else if (bw + 64 < 512)  return pq_search_dispatch_width<GRAPH_CFG, BLOCK_SIZE, 512,  TILE_SIZE>(g, codebooks, d_queries, params);
  else if (bw + 64 < 1024) return pq_search_dispatch_width<GRAPH_CFG, BLOCK_SIZE, 1024, TILE_SIZE>(g, codebooks, d_queries, params);
  else if (bw + 64 < 2048) return pq_search_dispatch_width<GRAPH_CFG, BLOCK_SIZE, 2048, TILE_SIZE>(g, codebooks, d_queries, params);
  else throw std::invalid_argument(
      "beam_width + rerank " + std::to_string(bw) +
      " is too large (beam_width + rerank + 64 must be < 2048)");
}

// ───────────────────────────────── rq_search ───────────────────────────────
// Same beam-search skeleton again, with neighbor distances estimated from the
// per-edge RaBitQ codes (see rq_beam_search.cuh). Requires the graph's stored
// vectors to be in the rotated space the codes were built in, and the queries
// rotated with the same seed — truncating to rq_dim coordinates is the random
// projection P_d only there.

template <typename GRAPH_CFG,
          uint32_t BLOCK_SIZE,
          uint32_t MAX_SEARCH_WIDTH,
          uint32_t TILE_SIZE>
beam_search_result<GRAPH_CFG> rq_search_impl(
    const graph<GRAPH_CFG>&                                   g,
    rq_globals<GRAPH_CFG::rq_dim, GRAPH_CFG::rq_bits>         globals,
    typename GRAPH_CFG::vector_view_t&                        d_queries,
    const search_params&                                      params) {
  static_assert(GRAPH_CFG::use_lsh,
                "rq_search requires graph_cfg::use_lsh (directional storage)");

  using Cfg = beam_search_config<
      GRAPH_CFG, GRAPH_CFG::dist_func,
      BLOCK_SIZE, MAX_SEARCH_WIDTH, TILE_SIZE>;

  beam_search_params<Cfg> bp {
    .graph           = g,
    .query_vectors   = d_queries,
    .use_range       = false,
    .medoid          = g.medoid,
    .k               = params.k,
    .beam_width      = params.beam_width,
    .limit           = params.limit,
    .get_visited     = params.get_visited,
    .max_result_size = params.max_result_size,
    .adaptive_gamma  = params.adaptive_gamma,
    .rerank          = params.rerank,
  };
  return rq_beam_search<Cfg>(bp, globals);
}

template <typename GRAPH_CFG, uint32_t BLOCK_SIZE, uint32_t MAX_SEARCH_WIDTH,
          uint32_t TILE_SIZE>
beam_search_result<GRAPH_CFG> rq_search_dispatch_width(
    const graph<GRAPH_CFG>&                           g,
    rq_globals<GRAPH_CFG::rq_dim, GRAPH_CFG::rq_bits> globals,
    typename GRAPH_CFG::vector_view_t                 d_queries,
    const search_params&                              params) {
  return rq_search_impl<GRAPH_CFG, BLOCK_SIZE, MAX_SEARCH_WIDTH, TILE_SIZE>(
      g, globals, d_queries, params);
}

template <typename GRAPH_CFG,
          uint32_t BLOCK_SIZE = 128,
          uint32_t TILE_SIZE  = 4>
beam_search_result<GRAPH_CFG> rq_search(
    const graph<GRAPH_CFG>&                           g,
    rq_globals<GRAPH_CFG::rq_dim, GRAPH_CFG::rq_bits> globals,
    typename GRAPH_CFG::vector_view_t                 d_queries,
    const search_params&                              params = {}) {
  static_assert(GRAPH_CFG::use_lsh,
                "rq_search requires graph_cfg::use_lsh (directional storage)");

  // The list holds beam_width + rerank candidates; that picks the tier.
  const uint32_t bw = params.beam_width + params.rerank;
  if      (bw + 64 < 128)  return rq_search_dispatch_width<GRAPH_CFG, BLOCK_SIZE, 128,  TILE_SIZE>(g, globals, d_queries, params);
  else if (bw + 64 < 256)  return rq_search_dispatch_width<GRAPH_CFG, BLOCK_SIZE, 256,  TILE_SIZE>(g, globals, d_queries, params);
  else if (bw + 64 < 512)  return rq_search_dispatch_width<GRAPH_CFG, BLOCK_SIZE, 512,  TILE_SIZE>(g, globals, d_queries, params);
  else if (bw + 64 < 1024) return rq_search_dispatch_width<GRAPH_CFG, BLOCK_SIZE, 1024, TILE_SIZE>(g, globals, d_queries, params);
  else if (bw + 64 < 2048) return rq_search_dispatch_width<GRAPH_CFG, BLOCK_SIZE, 2048, TILE_SIZE>(g, globals, d_queries, params);
  else throw std::invalid_argument(
      "beam_width + rerank " + std::to_string(bw) +
      " is too large (beam_width + rerank + 64 must be < 2048)");
}

}  // namespace jasper
