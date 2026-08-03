// jasper/beam_search/pq_beam_search.cuh
//
// Product-Quantization (ADC) edge scorer for the generic directional
// beam-search kernel (see graph_beam_search.cuh). Defines `pq_estimator`,
// which supplies the estimator API graph_beam_search_kernel is templated on:
// building the query's ADC lookup table once per search, and scoring a
// popped node's out-edges via table lookups over the learned PQ codes stored
// on each edge (graph.segments[...].edge_pqs). See docs/lsh_beam_search_v5.md.
#pragma once

#include <cub/cub.cuh>

#include "jasper/distance/distance.cuh"
#include "jasper/index/vector.cuh"
#include "jasper/index/graph.cuh"
#include "jasper/beam_search/config.cuh"
#include "jasper/beam_search/entry.cuh"
#include "jasper/beam_search/beam_search_common.cuh"
#include "jasper/beam_search/graph_beam_search.cuh"
#include "jasper/pq/pq_codebooks.cuh"

namespace jasper {

// ===== Build the query ADC lookup table (once per search) =====
// LUT[j*K + c] = <q_j, C_j[c]>, the projection of the query onto every centroid.
// δ = q (the query) for BOTH distance functions, so the table is constant across
// the whole search and is built exactly once — the popped node u enters the
// estimate only through the scalars <q,u> and ||v||² (see the scorer below),
// never through the LUT. This removes the per-node codebook re-read that used to
// dominate the L2 path.
template <typename DATA_T, uint32_t M, uint32_t K>
__device__ __forceinline__ void pq_build_lut(
    const DATA_T* __restrict__ query_vec,
    const float*  __restrict__ centroids,
    uint32_t                   dsub,
    float*        __restrict__ lut)
{
  // One WARP per centroid, lanes striding over the dsub coordinates. For a
  // fixed element the 32 lanes read Cj[0..31] — 32 contiguous global floats,
  // i.e. one coalesced transaction. (The old thread-per-element mapping had
  // adjacent lanes stride dsub floats apart, shattering each warp's read into
  // dsub sectors — the source of the "excessive global load" the profiler saw.)
  const uint32_t total   = M * K;
  const uint32_t lane    = threadIdx.x & 31u;
  const uint32_t warp    = threadIdx.x >> 5;
  const uint32_t n_warps = blockDim.x >> 5;
  for (uint32_t e = warp; e < total; e += n_warps) {
    const uint32_t j  = e / K;                       // subspace
    const float*  Cj = centroids + static_cast<size_t>(e) * dsub;  // (j*K+c)*dsub
    const DATA_T* qj = query_vec + static_cast<size_t>(j) * dsub;  // shared mem
    float acc = 0.0f;
    for (uint32_t t = lane; t < dsub; t += 32u)
      acc += static_cast<float>(qj[t]) * Cj[t];
    #pragma unroll
    for (int d = 16; d > 0; d >>= 1)
      acc += __shfl_xor_sync(0xFFFFFFFFu, acc, d);
    if (lane == 0) lut[e] = acc;
  }
  __syncthreads();
}

// ===== Score each new neighbor v of u via ADC table lookups =====
// The query LUT gives <q, ê> = Σ_j LUT[j, code_j] with ê the PQ reconstruction
// of the residual e = v - u, so <q,v> ≈ <q,u> + <q,ê>. From that:
//   INNER_PRODUCT → <q,v> ≈ qu_dot + Σ_j LUT[j,code_j]      (negated: smaller==closer)
//   L2            → ||q-v||² ≈ ||q||² - 2·<q,v> + ||v||²,
//                   using the EXACT stored ||v||² (graph.get_vector_norm) rather
//                   than a reconstructed norm — both faster and more accurate.
template <typename GRAPH_CFG, distance_func DISTANCE_FUNC, uint32_t M, uint32_t K>
__device__ __forceinline__ void pq_populate_estimated_distances(
    float                                   q_sq,      // ||q||²  (L2 only)
    float                                   qu_dot,    // <q,u>
    typename graph<GRAPH_CFG>::device_view& graph,
    typename GRAPH_CFG::index_t             u_gid,
    const float* __restrict__               lut,
    ENTRY_T*     __restrict__               result_buffer,
    uint32_t                                offset,
    uint8_t                                 n_edges)
{
  using INDEX_T = typename GRAPH_CFG::index_t;
  constexpr INDEX_T INVALID_INDEX = static_cast<INDEX_T>(0x7FFFFFFFu);

  // Hoist u-side segment lookup (same layout math as the LSH estimator).
  const INDEX_T  u_lid    = u_gid - graph.global_offset;
  const uint32_t u_segid  = static_cast<uint32_t>(u_lid / GRAPH_CFG::vectors_per_segment);
  const uint32_t u_locidx = static_cast<uint32_t>(u_lid % GRAPH_CFG::vectors_per_segment);
  const auto&    u_segment = graph.segments[u_segid];
  const uint8_t* __restrict__ code_base = u_segment.edge_pqs[u_locidx].code;

  for (uint32_t e = threadIdx.x; e < n_edges; e += blockDim.x) {
    ENTRY_T entry = result_buffer[offset + e];
    INDEX_T v     = static_cast<INDEX_T>(get_index(entry));
    if (v == INVALID_INDEX) continue;

    // Subspace-major codes: code_base[j * N_NEIGHBORS + e]. For a fixed j the
    // warp's edges read consecutive bytes -> one coalesced transaction.
    constexpr uint32_t N = GRAPH_CFG::n_neighbors;
    float acc = qu_dot;                       // <q,u> + Σ_j <q, C_j[code_j]>
    #pragma unroll
    for (uint32_t j = 0; j < M; ++j)
      acc += lut[j * K + code_base[j * N + e]];

    float est_dist;
    if constexpr (DISTANCE_FUNC == distance_func::L2) {
      est_dist = q_sq - 2.0f * acc + graph.get_vector_norm(v);
    } else {  // INNER_PRODUCT
      est_dist = -acc;
    }
    result_buffer[offset + e] = set_distance(entry, est_dist);
  }
  __syncthreads();
}

// ===== Estimator: Product-Quantization ADC =====
// Fulfills the graph_beam_search_kernel estimator API (see graph_beam_search.cuh):
//   - globals_t            = pq_codebooks_view<pq_m, pq_k>   (learned centroids)
//   - extra_smem_size()    = smem for the query's ADC lookup table
//   - constructor          = builds the ADC LUT (and ||q||² for L2) once per search
//   - visit()              = one exact I/O of u: <q,u>, plus (L2) ||u||² via graph norms
//   - estimate_neighbors() = score u's out-edges via LUT lookups over PQ codes
template <typename GRAPH_CFG, distance_func DISTANCE_FUNC, uint32_t BLOCK_SIZE>
struct pq_estimator {
  using DATA_T  = typename GRAPH_CFG::data_t;
  using INDEX_T = typename GRAPH_CFG::index_t;
  static constexpr uint32_t M = GRAPH_CFG::pq_m;
  static constexpr uint32_t K = GRAPH_CFG::pq_k;
  using globals_t = pq_codebooks_view<M, K>;

  __host__ static uint32_t extra_smem_size(uint32_t /*padded_dim*/) {
    return sizeof(float) * M * K;  // smem_lut
  }

  float* lut;
  float  q_sq;    // ||q||²  (L2 only); reused every iteration
  float  qu_dot;  // <q,u>   scratch: set by visit(), read by estimate_neighbors()

  // The ADC LUT projects the query onto every centroid (δ = q) and is constant
  // for the whole search, so build it ONCE here for both distance functions.
  __device__ pq_estimator(unsigned char* extra_smem,
                          DATA_T* smem_query_vec,
                          uint32_t padded_dim,
                          const globals_t& codebooks)
    : lut(reinterpret_cast<float*>(extra_smem)), q_sq(0.0f)
  {
    pq_build_lut<DATA_T, M, K>(
        smem_query_vec, codebooks.centroids, codebooks.dsub, lut);

    // ||q||² (L2 only) — reused every iteration to form ||q-v||² from <q,v>.
    // Plain block reduction over the shared query vector (compute_qu_diff_and_exact
    // can't be reused here: it __ldg's its second arg, which is illegal on smem).
    if constexpr (DISTANCE_FUNC == distance_func::L2) {
      float local = 0.0f;
      for (uint32_t i = threadIdx.x; i < padded_dim; i += BLOCK_SIZE) {
        const float x = static_cast<float>(smem_query_vec[i]);
        local += x * x;
      }
      #pragma unroll
      for (int d = 16; d > 0; d >>= 1)
        local += __shfl_xor_sync(0xFFFFFFFFu, local, d);
      constexpr uint32_t N_WARPS = BLOCK_SIZE / 32;
      __shared__ float s_qsq[N_WARPS > 0 ? N_WARPS : 1];
      const uint32_t lane = threadIdx.x & 31u;
      const uint32_t wid  = threadIdx.x >> 5;
      if (lane == 0) s_qsq[wid] = local;
      __syncthreads();
      if (wid == 0) {
        float v = (lane < N_WARPS) ? s_qsq[lane] : 0.0f;
        #pragma unroll
        for (int d = 16; d > 0; d >>= 1)
          v += __shfl_xor_sync(0xFFFFFFFFu, v, d);
        if (lane == 0) s_qsq[0] = v;
      }
      __syncthreads();
      q_sq = s_qsq[0];
    }
  }

  // Fetch u's vector; compute <q,u> in one pass. The estimate only needs this
  // scalar from u (not the full q-u vector). For L2 the exact ||q-u||² for the
  // frontier is recovered as ||q||² - 2<q,u> + ||u||².
  __device__ float visit(typename graph<GRAPH_CFG>::device_view& graph,
                          INDEX_T u_gid, DATA_T* smem_query_vec, uint32_t padded_dim) {
    DATA_T* u_vec = graph.get_vector(u_gid);
    qu_dot = compute_qu_diff_and_exact<DATA_T, BLOCK_SIZE, distance_func::INNER_PRODUCT>(
        smem_query_vec, u_vec, /*qu_diff=*/nullptr, padded_dim);
    if constexpr (DISTANCE_FUNC == distance_func::L2)
      return q_sq - 2.0f * qu_dot + graph.get_vector_norm(u_gid);
    else
      return -qu_dot;
  }

  // ESTIMATE distances via PQ ADC using the once-built query LUT. No per-node
  // table rebuild: u enters only through the scalar <q,u>.
  __device__ void estimate_neighbors(typename graph<GRAPH_CFG>::device_view& graph,
                                      INDEX_T u_gid, DATA_T* /*smem_query_vec*/,
                                      ENTRY_T* result_buffer, uint32_t offset,
                                      uint8_t n_edges) {
    pq_populate_estimated_distances<GRAPH_CFG, DISTANCE_FUNC, M, K>(
        q_sq, qu_dot, graph, u_gid, lut, result_buffer, offset, n_edges);
  }
};

// ===== Host: launcher =====
template <typename Cfg>
__host__ beam_search_result<typename Cfg::graph_cfg_t>
pq_beam_search(
    const beam_search_params<Cfg>&                                            p,
    pq_codebooks_view<Cfg::graph_cfg_t::pq_m, Cfg::graph_cfg_t::pq_k>          codebooks,
    cudaStream_t                                                              stream = 0)
{
  using graph_cfg_t = typename Cfg::graph_cfg_t;
  static_assert(graph_cfg_t::use_lsh,
                "pq_beam_search requires graph_cfg::use_lsh (directional storage)");
  using Estimator = pq_estimator<graph_cfg_t, Cfg::dist_func, Cfg::block_size>;
  return graph_beam_search<Cfg, Estimator>(p, codebooks, stream);
}

// Variant with pre-allocated result buffer, matching directional_beam_search.cuh.
template <typename Cfg>
__host__ void pq_beam_search(
    const beam_search_params<Cfg>&                                            p,
    pq_codebooks_view<Cfg::graph_cfg_t::pq_m, Cfg::graph_cfg_t::pq_k>          codebooks,
    beam_search_result<typename Cfg::graph_cfg_t>                             result,
    cudaStream_t                                                              stream = 0)
{
  using graph_cfg_t = typename Cfg::graph_cfg_t;
  static_assert(graph_cfg_t::use_lsh,
                "pq_beam_search requires graph_cfg::use_lsh (directional storage)");
  using Estimator = pq_estimator<graph_cfg_t, Cfg::dist_func, Cfg::block_size>;
  graph_beam_search<Cfg, Estimator>(p, codebooks, result, stream);
}

}  // namespace jasper
