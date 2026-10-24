// jasper/beam_search/rq_beam_search.cuh
//
// RaBitQ edge scorer for the generic directional beam-search kernel (see
// graph_beam_search.cuh). Defines `rq_estimator`, the third implementation of
// that kernel's estimator API alongside lsh_estimator (cross-polytope) and
// pq_estimator (ADC).
//
// ── The estimate ───────────────────────────────────────────────────────────
// This is the RaBitQ ">32x compression" construction with the popped node u
// playing the role of the centroid c, i.e. the law of cosines around u:
//
//   ||q-v||² = ||q-u||² + ||v-u||² - 2·<v-u, q-u>                        (62)
//            = A + mag² - 2·mag·√A·cos θ
//
// with A = ||q-u||² and mag = ||v-u||² both known EXACTLY — A from the one
// full-vector read the beam search already does per popped node, mag from a
// per-edge bf16. Only cos θ is estimated, and only there does the compression
// enter: project both sides onto d = rq_dim dimensions and apply RaBitQ to the
// projected unit vectors,
//
//   cos θ ≈ <unit(P_d(v-u)), unit(P_d(q-u))> ≈ <x̄, r[:d]> · rescale / ||r[:d]||
//                                                                    (63)-(65)
//
// where r = q-u, x̄[j] = level[j] + cb is the unnormalized RaBitQ
// reconstruction of the edge's direction, and rescale = 1/<x̄, û> is the stored
// factor that makes the ratio unbiased. Substituting:
//
//   est = A + mag² - (2√A / ||r[:d]||) · mag · rescale · <x̄, r[:d]>
//
// Everything in the first parenthesis is per popped node; mag, rescale and the
// dot are per edge.
//
// What makes this cheap is that P_d is coordinate truncation: the stored
// vectors are in a Haar-rotated space (--rotate / prerotate), where the first d
// coordinates are a uniformly random d-frame — T_d·R and an explicit random
// d×D projection are the same random object, not approximations of each other
// (see rabitq/edge_rq.cuh). So P_d(q-u) is literally the first d entries of the
// residual the kernel already writes to shared memory, and the query side of
// the projection costs nothing: no matrix, no per-node matvec, no stored
// projections. The whole estimator state is then one d-long shared buffer.
//
// <x̄, r_d> is accumulated directly from the code words — d shared-memory
// broadcasts and d FMAs per edge. (A nibble lookup table would cut that to
// d·bits/4 lookups, but at 64 edges per node and one edge per thread the
// difference is tens of cycles against a ~500-cycle vector fetch, and it costs
// a per-node table build plus the index algebra to keep in sync with the code
// layout.)
//
// For INNER_PRODUCT the same decomposition runs against q rather than the
// residual — <q,v> = <q,u> + mag·||q||·cos θ — so the estimator vector is the
// query's own prefix, already in shared memory, and this path needs no extra
// shared memory at all.
#pragma once

#include <cub/cub.cuh>

#include "jasper/distance/distance.cuh"
#include "jasper/index/vector.cuh"
#include "jasper/index/graph.cuh"
#include "jasper/beam_search/config.cuh"
#include "jasper/beam_search/entry.cuh"
#include "jasper/beam_search/beam_search_common.cuh"
#include "jasper/beam_search/graph_beam_search.cuh"
#include "jasper/rabitq/edge_rq.cuh"

namespace jasper {

// ===== Block-wide float sum =====
// Used for the two small norms this estimator needs (||r[:d]||², ||q||²).
// One slot per warp; safe to call repeatedly because the trailing barrier
// covers both the read and the write of the scratch.
template <uint32_t BLOCK_SIZE>
__device__ __forceinline__ float rq_block_sum(float local) {
  #pragma unroll
  for (int d = 16; d > 0; d >>= 1)
    local += __shfl_xor_sync(0xFFFFFFFFu, local, d);

  constexpr uint32_t N_WARPS = BLOCK_SIZE / 32;
  __shared__ float s_warp[N_WARPS > 0 ? N_WARPS : 1];
  const uint32_t lane = threadIdx.x & 31u;
  const uint32_t wid  = threadIdx.x >> 5;
  if (lane == 0) s_warp[wid] = local;
  __syncthreads();
  if (wid == 0) {
    float v = (lane < N_WARPS) ? s_warp[lane] : 0.0f;
    #pragma unroll
    for (int d = 16; d > 0; d >>= 1)
      v += __shfl_xor_sync(0xFFFFFFFFu, v, d);
    if (lane == 0) s_warp[0] = v;
  }
  __syncthreads();
  return s_warp[0];
}

// ===== Score each new neighbor v of u from its RaBitQ code =====
// est_L2 = A + mag² − node_scale·mag·rescale·<x̄, r_d>
// est_IP = −( <q,u> + node_scale·mag·rescale·<x̄, q_d> )   (negated: smaller == closer)
//
// node_scale folds the per-node part of eq (65):
//   L2 → 2·√A / ||r_d||        IP → ||q|| / ||q_d||
//
// `est_vec` is the rq_dim-long estimator vector in shared memory: the residual
// prefix (L2) or the query prefix (IP).
//
// `u_rq` is u's edge list, staged in shared memory by the search kernel or the
// graph's own (distortion.cuh); plain loads, not __ldg, so either works.
template <typename GRAPH_CFG, distance_func DISTANCE_FUNC>
__device__ __forceinline__ void rq_populate_estimated_distances(
    float                                   exact_u,    // A (L2) or <q,u> (IP)
    float                                   node_scale,
    const typename GRAPH_CFG::edge_rq_list_t& u_rq,
    const typename GRAPH_CFG::data_t* __restrict__ est_vec,
    ENTRY_T*     __restrict__               result_buffer,
    uint32_t                                offset,
    uint8_t                                 n_edges)
{
  using INDEX_T   = typename GRAPH_CFG::index_t;
  using edge_rq_t = typename GRAPH_CFG::edge_rq_list_t;
  constexpr INDEX_T  INVALID_INDEX = static_cast<INDEX_T>(0x7FFFFFFFu);
  constexpr uint32_t N_NEIGHBORS   = GRAPH_CFG::n_neighbors;
  constexpr uint32_t N_WORDS       = edge_rq_t::n_words;
  constexpr uint32_t DIMS_PER_WORD = edge_rq_t::dims_per_word;
  constexpr uint32_t LEVEL_MASK    = edge_rq_t::level_mask;
  constexpr uint32_t RQ_BITS       = GRAPH_CFG::rq_bits;
  // x̄[j] = level[j] + cb, the unnormalized RaBitQ reconstruction. At 1 bit per
  // dimension cb = -0.5, so (level + cb) is just ±½ — a select, not a multiply.
  constexpr float    CB =
      rq_globals<GRAPH_CFG::rq_dim, GRAPH_CFG::rq_bits>::cb();

  const uint32_t*      __restrict__ code_base    = u_rq.code;
  const __half*        __restrict__ rescale_base = u_rq.rescale;
  const __nv_bfloat16* __restrict__ mag_base     = u_rq.mag_sq;

  for (uint32_t e = threadIdx.x; e < n_edges; e += blockDim.x) {
    ENTRY_T entry = result_buffer[offset + e];
    INDEX_T v     = static_cast<INDEX_T>(get_index(entry));
    if (v == INVALID_INDEX) continue;

    // <x̄, est_vec> straight off the code. Word-plane-major codes mean a warp of
    // edges reads consecutive words for a fixed plane — one coalesced
    // transaction — and every lane reads the same est_vec[] element at each
    // step, which shared memory broadcasts.
    float dot = 0.0f;
    #pragma unroll
    for (uint32_t w = 0; w < N_WORDS; ++w) {
      const uint32_t word = code_base[w * N_NEIGHBORS + e];
      #pragma unroll
      for (uint32_t t = 0; t < DIMS_PER_WORD; ++t) {
        const float level =
            static_cast<float>((word >> (t * RQ_BITS)) & LEVEL_MASK);
        dot += (level + CB)
             * static_cast<float>(est_vec[w * DIMS_PER_WORD + t]);
      }
    }

    const float mag_sq  = __bfloat162float(mag_base[e]);
    const float rescale = __half2float(rescale_base[e]);
    const float mag     = sqrtf(mag_sq);
    const float cross   = node_scale * mag * rescale * dot;

    float est_dist;
    if constexpr (DISTANCE_FUNC == distance_func::L2) {
      est_dist = exact_u + mag_sq - cross;
    } else {  // INNER_PRODUCT
      est_dist = -(exact_u + cross);
    }
    result_buffer[offset + e] = set_distance(entry, est_dist);
  }
  __syncthreads();
}

// ===== Estimator: RaBitQ =====
// Fulfills the graph_beam_search_kernel estimator API (see graph_beam_search.cuh):
//   - globals_t            = rq_globals<rq_dim, rq_bits>  (build-side constants)
//   - extra_smem_size()    = rq_dim residual prefix (L2) / nothing (IP)
//   - constructor          = (IP) ||q|| / ||q_d||, which is query-invariant
//   - edge_codes_t / edge_codes() = u's edge_rq_list in graph memory
//   - visit()              = one exact I/O of u; (L2) the per-node cosine scale
//   - estimate_neighbors() = dot each out-edge's code against the estimator vector
template <typename GRAPH_CFG, distance_func DISTANCE_FUNC, uint32_t BLOCK_SIZE>
struct rq_estimator {
  using DATA_T       = typename GRAPH_CFG::data_t;
  using INDEX_T      = typename GRAPH_CFG::index_t;
  using edge_codes_t = typename GRAPH_CFG::edge_rq_list_t;

  __device__ static const edge_codes_t* edge_codes(
      const typename graph<GRAPH_CFG>::device_view& graph, INDEX_T u_gid) {
    const INDEX_T u_lid = u_gid - graph.global_offset;
    return &graph.segments[u_lid / GRAPH_CFG::vectors_per_segment]
                .edge_rqs[u_lid % GRAPH_CFG::vectors_per_segment];
  }

  static constexpr uint32_t RQ_DIM  = GRAPH_CFG::rq_dim;
  static constexpr uint32_t RQ_BITS = GRAPH_CFG::rq_bits;
  static constexpr bool     IS_L2   = (DISTANCE_FUNC == distance_func::L2);

  using globals_t = rq_globals<RQ_DIM, RQ_BITS>;
  static_assert(RQ_DIM % 8u == 0u, "rq_dim must be a multiple of 8");
  static_assert(BLOCK_SIZE >= 32 && BLOCK_SIZE % 32 == 0,
                "rq_estimator reduces over whole warps");

  // Shared memory: the L2 path needs the residual's rq_dim-long prefix
  // (16B-aligned for compute_qu_diff_and_exact's vectorized stores, and capped
  // there by STORE_DIMS so it never scales with the data dimension). The IP
  // path dots against the query itself, which is already in shared memory, so
  // it normally needs nothing — unless rq_dim outruns the data, where it needs
  // its own zero-padded copy rather than reading off the end of the query.
  static constexpr uint32_t RD_BYTES =
      (static_cast<uint32_t>(sizeof(DATA_T)) * RQ_DIM + 15u) & ~15u;
  __host__ static uint32_t extra_smem_size(uint32_t padded_dim) {
    if constexpr (IS_L2) return RD_BYTES;
    return (RQ_DIM > padded_dim) ? RD_BYTES : 0u;
  }

  DATA_T*       smem_qu_diff;  // (q-u)[:rq_dim], written by visit(); L2 only
  const DATA_T* est_vec;       // what the codes are dotted against
  float         node_scale;    // per-node (L2) or per-query (IP) cosine scale
  float         exact_u;       // set by visit(), read by estimate_neighbors()

  // ||est_vec[:rq_dim]||² — the norm of the PROJECTED estimator vector, which
  // is what normalizes the cosine (the full-dimension norms come from A and
  // mag).
  __device__ __forceinline__ float proj_norm_sq() {
    float local = 0.0f;
    for (uint32_t i = threadIdx.x; i < RQ_DIM; i += BLOCK_SIZE) {
      const float v = static_cast<float>(est_vec[i]);
      local += v * v;
    }
    return rq_block_sum<BLOCK_SIZE>(local);
  }

  __device__ rq_estimator(unsigned char* extra_smem,
                          DATA_T* smem_query_vec,
                          uint32_t padded_dim,
                          const globals_t& /*globals*/)
    : smem_qu_diff(reinterpret_cast<DATA_T*>(extra_smem)),
      est_vec(IS_L2 || RQ_DIM > padded_dim ? smem_qu_diff : smem_query_vec),
      node_scale(0.0f),
      exact_u(0.0f)
  {
    // rq_dim wider than the data: no projection is possible, so this is plain
    // 1-bit RaBitQ over the whole vector, with the surplus coordinates pinned
    // to zero. Writing those zeros ONCE here is what lets the per-node and
    // per-edge loops below keep compile-time bounds and read them for free —
    // a zero coordinate adds nothing to the norm or to the dot, so the result
    // matches a code built at rq_dim == padded_dim exactly. The encoder pins
    // the same tail (rq_populate), so both sides agree bit for bit.
    if (RQ_DIM > padded_dim) {
      for (uint32_t i = padded_dim + threadIdx.x; i < RQ_DIM; i += BLOCK_SIZE)
        smem_qu_diff[i] = static_cast<DATA_T>(0.0f);
      // IP dots against the query, which is only padded_dim long, so it needs
      // its own zero-padded copy rather than reading past the end of it.
      if constexpr (!IS_L2) {
        for (uint32_t i = threadIdx.x; i < padded_dim; i += BLOCK_SIZE)
          smem_qu_diff[i] = smem_query_vec[i];
      }
      __syncthreads();
    }

    // INNER_PRODUCT dots against q itself, so the scale is query-invariant:
    // compute it once here instead of per popped node.
    if constexpr (!IS_L2) {
      float local = 0.0f;
      for (uint32_t i = threadIdx.x; i < padded_dim; i += BLOCK_SIZE) {
        const float v = static_cast<float>(smem_query_vec[i]);
        local += v * v;
      }
      const float q_sq      = rq_block_sum<BLOCK_SIZE>(local);
      const float q_proj_sq = proj_norm_sq();
      // ||q|| / ||q_d||, guarded against a zero query.
      node_scale = sqrtf(q_sq) * rsqrtf(q_proj_sq + 1e-20f);
    }
  }

  // Fetch u's full vector — the ONE I/O per explored node. L2 also lands the
  // residual's first rq_dim coordinates in shared memory (STORE_DIMS caps the
  // store there), and those ARE P_d(q-u).
  __device__ float visit(typename graph<GRAPH_CFG>::device_view& graph,
                         INDEX_T u_gid, DATA_T* smem_query_vec, uint32_t padded_dim) {
    DATA_T* u_vec = graph.get_vector(u_gid);
    exact_u = compute_qu_diff_and_exact<DATA_T, BLOCK_SIZE, DISTANCE_FUNC,
                                        /*STORE_DIMS=*/RQ_DIM>(
        smem_query_vec, u_vec, smem_qu_diff, padded_dim);

    if constexpr (IS_L2) {
      // 2·√A / ||r_d||. At A == 0 (query sits on u) this is 0 and the estimate
      // collapses to mag², which is the exact answer there.
      const float proj_sq = proj_norm_sq();
      node_scale = 2.0f * sqrtf(exact_u) * rsqrtf(proj_sq + 1e-20f);
      return exact_u;
    } else {
      return -exact_u;
    }
  }

  // ESTIMATE ||q-v||² (or -<q,v>) for the newly-appended neighbors from the
  // RaBitQ code stored on each edge (u, v) — no fetch of v's vector.
  __device__ void estimate_neighbors(typename graph<GRAPH_CFG>::device_view& /*graph*/,
                                     DATA_T* /*smem_query_vec*/,
                                     const edge_codes_t& codes,
                                     ENTRY_T* result_buffer, uint32_t offset,
                                     uint8_t n_edges) {
    rq_populate_estimated_distances<GRAPH_CFG, DISTANCE_FUNC>(
        exact_u, node_scale, codes, est_vec, result_buffer, offset, n_edges);
  }
};

// ===== Host: launcher =====
template <typename Cfg>
__host__ beam_search_result<typename Cfg::graph_cfg_t>
rq_beam_search(
    const beam_search_params<Cfg>& p,
    rq_globals<Cfg::graph_cfg_t::rq_dim, Cfg::graph_cfg_t::rq_bits> globals,
    cudaStream_t                   stream = 0)
{
  using graph_cfg_t = typename Cfg::graph_cfg_t;
  static_assert(graph_cfg_t::use_lsh,
                "rq_beam_search requires graph_cfg::use_lsh (directional storage)");
  using Estimator = rq_estimator<graph_cfg_t, Cfg::dist_func, Cfg::block_size>;
  return graph_beam_search<Cfg, Estimator>(p, globals, stream);
}

// Variant with pre-allocated result buffer, matching pq_beam_search.cuh.
template <typename Cfg>
__host__ void rq_beam_search(
    const beam_search_params<Cfg>& p,
    rq_globals<Cfg::graph_cfg_t::rq_dim, Cfg::graph_cfg_t::rq_bits> globals,
    beam_search_result<typename Cfg::graph_cfg_t>                   result,
    cudaStream_t                                                    stream = 0)
{
  using graph_cfg_t = typename Cfg::graph_cfg_t;
  static_assert(graph_cfg_t::use_lsh,
                "rq_beam_search requires graph_cfg::use_lsh (directional storage)");
  using Estimator = rq_estimator<graph_cfg_t, Cfg::dist_func, Cfg::block_size>;
  graph_beam_search<Cfg, Estimator>(p, globals, result, stream);
}

}  // namespace jasper
