// jasper/beam_search/directional_beam_search.cuh
//
// Cross-polytope LSH edge scorer for the generic directional beam-search
// kernel (see graph_beam_search.cuh). Defines `lsh_estimator`, which supplies
// the estimator API graph_beam_search_kernel is templated on: precomputing
// per-query state (here, just the calibration-derived 1/norm_denom), and
// scoring a popped node's out-edges using the packed sign/coordinate words +
// magnitude stored on each edge (graph.segments[...].edge_lshs).
#pragma once

#include "jasper/distance/distance.cuh"
#include "jasper/index/vector.cuh"
#include "jasper/index/graph.cuh"
#include "jasper/beam_search/config.cuh"
#include "jasper/beam_search/entry.cuh"
#include "jasper/beam_search/beam_search_common.cuh"
#include "jasper/beam_search/graph_beam_search.cuh"
#include "jasper/lsh/lsh_globals.cuh"

namespace jasper {

// ===== Helper: estimated distance for each new neighbor v of u =====
// Implements the estimator from sec. "Query Phase". With mag = ||v-u|| and r̂
// the (unit) residual direction v = u + mag·r̂, the estimated projection is
//   îp(x, r̂) ≈ ( Σ_r c[r] · sign[r] · x[coord[r]] ) / norm_denom
// where `smem_est_vec` supplies x:
//   L2            → x = (q - u):   ||q-v||² ≈ ||q-u||² + mag² − 2·mag·îp(q-u, r̂)
//   INNER_PRODUCT → x = q:         ⟨q,v⟩   ≈ ⟨q,u⟩ + mag·îp(q, r̂),
//                                  and the stored distance is −⟨q,v⟩ (as in
//                                  inner_product_distance) so smaller == closer.
template <typename GRAPH_CFG, distance_func DISTANCE_FUNC>
__device__ __forceinline__ void populate_estimated_distances(
    const typename GRAPH_CFG::data_t* __restrict__ smem_est_vec,
    float                                          exact_u,
    typename graph<GRAPH_CFG>::device_view&        graph,
    typename GRAPH_CFG::index_t                    u_gid,
    const lsh_globals<GRAPH_CFG::k_ranks>&         globals,
    float                                          inv_norm_denom,
    ENTRY_T*  __restrict__                         result_buffer,
    uint32_t                                       offset,
    uint8_t                                        n_edges)
{
  using INDEX_T  = typename GRAPH_CFG::index_t;
  using PACKED_T = typename GRAPH_CFG::packed_t;
  constexpr uint8_t  K_RANKS       = GRAPH_CFG::k_ranks;
  constexpr INDEX_T  INVALID_INDEX = static_cast<INDEX_T>(0x7FFFFFFFu);

  // Packed-word layout (compile-time): sign in the MSB, coord in the low bits.
  // packed_t is uint8_t (7-bit coord) or uint16_t (15-bit coord).
  constexpr PACKED_T SIGN_MASK    = static_cast<PACKED_T>(PACKED_T{1} << (sizeof(PACKED_T) * 8 - 1));
  constexpr PACKED_T COORD_MASK   = static_cast<PACKED_T>(~SIGN_MASK);
  constexpr uint32_t PACKED_BYTES = static_cast<uint32_t>(K_RANKS) * sizeof(PACKED_T);

  // Hoist u-side segment lookup.
  const INDEX_T  u_lid    = u_gid - graph.global_offset;
  const uint32_t u_segid  = static_cast<uint32_t>(u_lid / GRAPH_CFG::vectors_per_segment);
  const uint32_t u_locidx = static_cast<uint32_t>(u_lid % GRAPH_CFG::vectors_per_segment);
  const auto&    u_segment = graph.segments[u_segid];

  // SoA edge-LSH layout: all edges' packed words, then a parallel bf16 mag_sq
  // array. Edge e's K_RANKS words start at packed_base + e*K_RANKS; that offset
  // is 4-byte aligned (K_RANKS*sizeof(PACKED_T) is a multiple of 4 for supported
  // configs and edge_lsh_list is alignas(4)), so the vectorized 32-bit loads
  // below stay valid.
  const auto& u_lsh = u_segment.edge_lshs[u_locidx];
  const PACKED_T*      __restrict__ packed_base = u_lsh.packed;
  const __nv_bfloat16* __restrict__ mag_base    = u_lsh.mag_sq;

  // c_per_rank in registers.
  float c_reg[K_RANKS];
  #pragma unroll
  for (uint8_t r = 0; r < K_RANKS; ++r) c_reg[r] = globals.c_per_rank[r];

  for (uint32_t e = threadIdx.x; e < n_edges; e += blockDim.x) {
    ENTRY_T entry = result_buffer[offset + e];
    INDEX_T v     = static_cast<INDEX_T>(get_index(entry));
    if (v == INVALID_INDEX) continue;

    const PACKED_T* __restrict__ row = packed_base + static_cast<uint32_t>(e) * K_RANKS;

    // ─── Load K_RANKS packed words for this edge ──
    PACKED_T p[K_RANKS];
    if constexpr (PACKED_BYTES % 4u == 0u) {
      // Packed group is a whole number of 32-bit words → vectorize the loads
      // (row is 4-byte aligned via edge_lsh_list's alignas(4)).
      constexpr uint32_t N_WORDS = PACKED_BYTES / 4u;
      const uint32_t* __restrict__ p32 = reinterpret_cast<const uint32_t*>(row);
      uint32_t words[N_WORDS];
      #pragma unroll
      for (uint32_t w = 0; w < N_WORDS; ++w) words[w] = __ldg(p32 + w);
      #pragma unroll
      for (uint32_t r = 0; r < K_RANKS; ++r) {
        if constexpr (sizeof(PACKED_T) == 1u) {
          p[r] = static_cast<PACKED_T>((words[r >> 2] >> ((r & 3u) * 8u)) & 0xFFu);
        } else {
          p[r] = static_cast<PACKED_T>((words[r >> 1] >> ((r & 1u) * 16u)) & 0xFFFFu);
        }
      }
    } else {
      // Odd packed size (e.g. uint8_t with K_RANKS not a multiple of 4): load
      // each packed word directly.
      #pragma unroll
      for (uint8_t r = 0; r < K_RANKS; ++r) p[r] = __ldg(row + r);
    }

    // mag_sq lives in the parallel bf16 array, indexed by edge.
    const float mag_sq = __bfloat162float(__ldg(mag_base + e));

    // ─── Estimator ───
    float dot_acc = 0.0f;
    #pragma unroll
    for (uint8_t r = 0; r < K_RANKS; ++r) {
      const PACKED_T word  = p[r];
      const uint32_t coord = static_cast<uint32_t>(word & COORD_MASK);
      const float    sgn   = (word & SIGN_MASK) ? -1.0f : +1.0f;
      const float    x     = static_cast<float>(smem_est_vec[coord]);
      dot_acc += c_reg[r] * sgn * x;
    }
    const float dot_est = dot_acc * inv_norm_denom;   // îp(x, r̂)
    const float mag     = sqrtf(mag_sq);

    float est_dist;
    if constexpr (DISTANCE_FUNC == distance_func::L2) {
      // ||q-v||² ≈ ||q-u||² + mag² − 2·mag·⟨q-u, r̂⟩
      est_dist = exact_u + mag_sq - 2.0f * mag * dot_est;
    } else {  // INNER_PRODUCT
      // ⟨q,v⟩ ≈ ⟨q,u⟩ + mag·⟨q, r̂⟩; distance is negated (smaller == closer).
      const float ip = exact_u + mag * dot_est;
      est_dist = -ip;
    }

    result_buffer[offset + e] = set_distance(entry, est_dist);
  }
  __syncthreads();
}

// ===== Estimator: cross-polytope LSH ADC =====
// Fulfills the graph_beam_search_kernel estimator API (see graph_beam_search.cuh):
//   - globals_t            = lsh_globals<k_ranks>            (calibration constants)
//   - extra_smem_size()    = smem for (q - u), the L2 residual used to score edges
//   - visit()              = one exact I/O of u + (L2) store (q - u) for later use
//   - estimate_neighbors() = score u's out-edges via packed sign/coord + magnitude
template <typename GRAPH_CFG, distance_func DISTANCE_FUNC, uint32_t BLOCK_SIZE>
struct lsh_estimator {
  using DATA_T    = typename GRAPH_CFG::data_t;
  using INDEX_T   = typename GRAPH_CFG::index_t;
  using globals_t = lsh_globals<GRAPH_CFG::k_ranks>;

  __host__ static uint32_t extra_smem_size(uint32_t padded_dim) {
    return sizeof(DATA_T) * padded_dim;  // smem_qu_diff
  }

  DATA_T*   smem_qu_diff;
  globals_t globals;
  float     inv_norm_denom;
  float     exact_u;  // scratch: set by visit(), read by estimate_neighbors()

  __device__ lsh_estimator(unsigned char* extra_smem,
                            DATA_T* /*smem_query_vec*/,
                            uint32_t /*padded_dim*/,
                            const globals_t& g)
    : smem_qu_diff(reinterpret_cast<DATA_T*>(extra_smem)),
      globals(g),
      inv_norm_denom(1.0f / g.norm_denom) {}

  // Fetch u's full vector. Computes (q-u) into smem and EXACT ||q-u||² in one
  // pass — the ONE I/O per explored node.
  __device__ float visit(typename graph<GRAPH_CFG>::device_view& graph,
                          INDEX_T u_gid, DATA_T* smem_query_vec, uint32_t padded_dim) {
    DATA_T* u_vec = graph.get_vector(u_gid);
    // exact_u is ||q-u||² (L2) or the raw dot ⟨q,u⟩ (INNER_PRODUCT).
    exact_u = compute_qu_diff_and_exact<DATA_T, BLOCK_SIZE, DISTANCE_FUNC>(
        smem_query_vec, u_vec, smem_qu_diff, padded_dim);
    // Distance stored for u: identity for L2, negated dot for INNER_PRODUCT
    // (so that smaller == closer, matching inner_product_distance).
    return (DISTANCE_FUNC == distance_func::L2) ? exact_u : -exact_u;
  }

  // ESTIMATE ||q-v||² for the newly-appended neighbors using the LSH info
  // stored on edges (u, v) — no fetch of v's vector required.
  __device__ void estimate_neighbors(typename graph<GRAPH_CFG>::device_view& graph,
                                      INDEX_T u_gid, DATA_T* smem_query_vec,
                                      ENTRY_T* result_buffer, uint32_t offset,
                                      uint8_t n_edges) {
    // L2 dots the residual (q-u) against r̂; IP dots the query q directly.
    const DATA_T* __restrict__ smem_est_vec =
        (DISTANCE_FUNC == distance_func::L2) ? smem_qu_diff : smem_query_vec;
    populate_estimated_distances<GRAPH_CFG, DISTANCE_FUNC>(
        smem_est_vec, exact_u, graph, u_gid, globals, inv_norm_denom,
        result_buffer, offset, n_edges);
  }
};

// ===== Host: launcher =====
template <typename Cfg>
__host__ beam_search_result<typename Cfg::graph_cfg_t>
directional_beam_search(
    const beam_search_params<Cfg>&                p,
    const lsh_globals<Cfg::graph_cfg_t::k_ranks>& globals,
    cudaStream_t                                  stream = 0)
{
  using graph_cfg_t = typename Cfg::graph_cfg_t;
  static_assert(graph_cfg_t::use_lsh,
                "directional_beam_search requires graph_cfg::use_lsh");
  using Estimator = lsh_estimator<graph_cfg_t, Cfg::dist_func, Cfg::block_size>;
  return graph_beam_search<Cfg, Estimator>(p, globals, stream);
}

// Variant with pre-allocated result buffer, matching beam_search.cuh.
template <typename Cfg>
__host__ void directional_beam_search(
    const beam_search_params<Cfg>&                p,
    const lsh_globals<Cfg::graph_cfg_t::k_ranks>& globals,
    beam_search_result<typename Cfg::graph_cfg_t> result,
    cudaStream_t                                  stream = 0)
{
  using graph_cfg_t = typename Cfg::graph_cfg_t;
  static_assert(graph_cfg_t::use_lsh,
                "directional_beam_search requires graph_cfg::use_lsh");
  using Estimator = lsh_estimator<graph_cfg_t, Cfg::dist_func, Cfg::block_size>;
  graph_beam_search<Cfg, Estimator>(p, globals, result, stream);
}

}  // namespace jasper
