// jasper/rabitq/rq_kernels.cuh
//
// Build-side encoder for the directional RaBitQ path: quantizes every directed
// edge's residual direction into the graph's edge_rq storage. The search-side
// counterpart is rq_estimator in beam_search/rq_beam_search.cuh.
//
// One block per node, one WARP per out-edge (4 edges in flight at the default
// 128 threads). A warp is the natural tile here: it covers RQ_DIM <= 128
// coordinates in at most four strides and lets the whole encode run on warp
// collectives, so unlike pq_populate there is no block-wide argmin to serialize
// on and no block-uniformity requirement beyond the edge count.
#pragma once

#include <cassert>
#include <cstdint>
#include <limits>
#include <type_traits>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "jasper/rabitq/edge_rq.cuh"
#include "jasper/rabitq/encode.cuh"

namespace jasper {

// Zero one edge slot's code, rescale and magnitude. Called on slots that carry
// no edge, so a reused graph buffer never exposes a previous build's codes.
// Single-threaded: the caller picks one thread.
template <typename graph_cfg, typename graph_t, uint32_t N_WORDS>
__device__ __forceinline__ void clear_rq_slot(
    graph_t& g, typename graph_cfg::index_t gid, uint8_t edge_idx) {
  #pragma unroll
  for (uint32_t w = 0; w < N_WORDS; ++w) g.set_rq_word(gid, edge_idx, w, 0u);
  g.set_rq_rescale(gid, edge_idx, __float2half(0.0f));
  g.set_rq_mag_sq (gid, edge_idx, __float2bfloat16(0.0f));
}

// ─── encode: RaBitQ-quantize every graph edge's residual direction ─────────
// grid = n_vectors blocks. For edge (u -> v):
//   mag_sq  = ||v-u||²                       (full dimension, exact)
//   code    = RaBitQ( unit((v-u)[:RQ_DIM]) ) (the projected direction)
//   rescale = 1 / <x̄, û>                     (RaBitQ's unbiasedness factor)
//
// Requires the stored vectors to be in the rotated space: the first RQ_DIM
// coordinates of a Haar-rotated vector are a uniformly random RQ_DIM-frame, so
// truncating there IS the projection P_d (see edge_rq.cuh). On unrotated data
// the same code would quantize raw axis-aligned coordinates instead — not an
// error, just quietly much worse.
//
// The magnitude stays a full-dimension quantity on purpose: the projected space
// is used ONLY for the direction, and is paired with exact norms (eq. 65).
template <typename graph_cfg, typename graph_t, uint32_t RQ_DIM, uint32_t RQ_BITS,
          uint32_t BLOCK_SIZE = 128>
__global__ void rq_populate(graph_t g, float t_const) {
  using index_t   = typename graph_cfg::index_t;
  using data_t    = typename graph_cfg::data_t;
  using edge_rq_t = typename graph_cfg::edge_rq_list_t;
  static_assert(std::is_same_v<data_t, __half>,
                "rq_populate requires data_t == __half");
  static_assert(BLOCK_SIZE % 32 == 0 && BLOCK_SIZE >= 32,
                "rq_populate needs a whole number of warps");

  constexpr uint32_t N_WARPS       = BLOCK_SIZE / 32;
  constexpr uint32_t N_WORDS       = edge_rq_t::n_words;
  constexpr uint32_t DIMS_PER_WORD = edge_rq_t::dims_per_word;
  constexpr uint32_t LEVEL_MASK    = edge_rq_t::level_mask;
  constexpr uint32_t N_NEIGHBORS   = graph_cfg::n_neighbors;
  constexpr index_t  INVALID       = std::numeric_limits<index_t>::max();

  auto block = cg::this_thread_block();
  auto tile  = cg::tiled_partition<32>(block);
  const uint32_t warp_id = block.thread_rank() / 32u;
  const uint32_t lane    = tile.thread_rank();

  // Per-warp encode workspace. RQ_DIM is a compile-time constant, so this is
  // static shared memory and the launcher needs no dynamic smem at all.
  __shared__ float   s_res [N_WARPS][RQ_DIM];
  __shared__ uint8_t s_code[N_WARPS][RQ_DIM];

  const uint32_t padded_dim = g.get_padded_dim();
  const index_t  u_lid      = static_cast<index_t>(blockIdx.x);
  const index_t  u_gid      = u_lid + g.global_offset;

  const data_t* __restrict__ uv = g.get_vector(u_gid);
  const uint8_t cnt = g.get_edge_count(u_gid);

  for (uint32_t e = warp_id; e < cnt; e += N_WARPS) {
    const uint8_t e8 = static_cast<uint8_t>(e);
    const index_t nb = g.get_neighbor(u_gid, e8);
    if (nb == INVALID) {
      if (lane == 0) clear_rq_slot<graph_cfg, graph_t, N_WORDS>(g, u_gid, e8);
      continue;
    }
    const data_t* __restrict__ nv = g.get_vector(nb);

    // One pass over the residual: the EXACT full-dimension ||v-u||² (which the
    // estimator pairs with the projected cosine) and the projected prefix that
    // gets coded come out together.
    float mag_acc = 0.0f, proj_acc = 0.0f;
    for (uint32_t i = lane; i < padded_dim; i += 32u) {
      const float r = __half2float(nv[i]) - __half2float(uv[i]);
      mag_acc += r * r;
      if (i < RQ_DIM) {
        s_res[warp_id][i] = r;
        proj_acc += r * r;
      }
    }
    const float mag_sq  = cg::reduce(tile, mag_acc,  cg::plus<float>());
    const float proj_sq = cg::reduce(tile, proj_acc, cg::plus<float>());
    tile.sync();

    // Degenerate: v and u agree on every projected coordinate, so the residual
    // carries no direction to quantize. Write rescale = 0, which makes the
    // estimator drop the cosine term and fall back to ||q-u||² + mag² exactly.
    // (Both reduce() results are warp-uniform, so this branch is too.)
    if (proj_sq <= 0.0f) {
      if (lane == 0) {
        clear_rq_slot<graph_cfg, graph_t, N_WORDS>(g, u_gid, e8);
        g.set_rq_mag_sq(u_gid, e8, __float2bfloat16(mag_sq));
      }
      continue;
    }

    const float ip_norm_inv = calc_multi_code<RQ_BITS, 32>(
        &s_res[warp_id][0], &s_code[warp_id][0], RQ_DIM,
        static_cast<double>(t_const), sqrtf(proj_sq), tile);

    // Pack the per-dimension levels into 32-bit words: dimension d occupies
    // bits [d*RQ_BITS, (d+1)*RQ_BITS) of word d/DIMS_PER_WORD, matching
    // edge_rq_list::get_level.
    for (uint32_t w = lane; w < N_WORDS; w += 32u) {
      uint32_t word = 0;
      #pragma unroll
      for (uint32_t t = 0; t < DIMS_PER_WORD; ++t) {
        const uint32_t d = w * DIMS_PER_WORD + t;
        word |= (static_cast<uint32_t>(s_code[warp_id][d]) & LEVEL_MASK)
                << (t * RQ_BITS);
      }
      g.set_rq_word(u_gid, e8, w, word);
    }

    if (lane == 0) {
      g.set_rq_rescale(u_gid, e8, __float2half(ip_norm_inv));
      g.set_rq_mag_sq (u_gid, e8, __float2bfloat16(mag_sq));
    }
  }

  // Zero the unused tail slots [cnt, N_NEIGHBORS). Never read at search time
  // (the scoring loop stops at edge_count), but it keeps dumps honest and a
  // reused buffer clean.
  for (uint32_t e = cnt + block.thread_rank(); e < N_NEIGHBORS; e += BLOCK_SIZE)
    clear_rq_slot<graph_cfg, graph_t, N_WORDS>(g, u_gid, static_cast<uint8_t>(e));
}

}  // namespace jasper
