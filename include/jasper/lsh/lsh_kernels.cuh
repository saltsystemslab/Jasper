#pragma once

#include <cassert>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace jasper {

template <typename graph_cfg, typename graph_t>
__global__ void populate_lsh(graph_t g) {
  using index_t = typename graph_cfg::index_t;
  using data_t  = typename graph_cfg::data_t;
  constexpr uint32_t n_neighbors = graph_cfg::n_neighbors;
  static_assert(std::is_same_v<data_t, __half>,
                "populate_lsh requires data_t == __half");

  const index_t  query_lid = static_cast<index_t>(blockIdx.x);
  const index_t  query_gid = query_lid + g.global_offset;
  const uint32_t padded_dim = g.get_padded_dim();
  const uint8_t  k_ranks    = graph_cfg::k_ranks;

  // The dim index is packed into the low COORD_BITS of the sort key and later
  // stored in a packed_t LSH coord slot (uint8_t → 7 bits, uint16_t → 15 bits).
  using packed_t = typename graph_cfg::packed_t;
  constexpr uint32_t COORD_BITS = sizeof(packed_t) * 8u - 1u;
  constexpr uint32_t COORD_MASK = (1u << COORD_BITS) - 1u;
  constexpr uint32_t SIGN_SHIFT = COORD_BITS;
  constexpr uint32_t MAG_SHIFT  = COORD_BITS + 1u;

  assert(padded_dim <= COORD_MASK + 1u &&
         "populate_lsh: padded_dim exceeds the dim-index width of packed_t");

  extern __shared__ unsigned char smem_lsh[];
  data_t   *relative_vec = reinterpret_cast<data_t*>(smem_lsh);
  size_t    off          = (padded_dim * sizeof(data_t)   + 15u) & ~15u;
  uint32_t *keys         = reinterpret_cast<uint32_t*>(smem_lsh + off);
  off                   += (padded_dim * sizeof(uint32_t) + 15u) & ~15u;
  uint32_t *warp_scratch = reinterpret_cast<uint32_t*>(smem_lsh + off);

  const uint32_t lane   = threadIdx.x & 31u;
  const uint32_t wid    = threadIdx.x >> 5;
  const uint32_t nwarps = blockDim.x  >> 5;

  data_t *query_vec = g.get_vector(query_gid);

  for (uint32_t neighbor_idx = 0; neighbor_idx < n_neighbors; ++neighbor_idx) {
    const index_t neighbor_gid =
        g.get_neighbor(query_gid, static_cast<uint8_t>(neighbor_idx));
    data_t *neighbor_vec = g.get_vector(neighbor_gid);

    // ─── Build Δ, packed keys, AND accumulate ||Δ||² in one pass ─────
    // Key layout (uint32_t, MSB → LSB), parametrized by packed_t width:
    //   magnitude : |d| as half bits, at bits [MAG_SHIFT .. MAG_SHIFT+14] (sort key)
    //   sign      : bit SIGN_SHIFT (1 = negative)
    //   dim index : low COORD_BITS bits (0 .. COORD_MASK)
    float l2sq_local = 0.0f;
    for (uint32_t i = threadIdx.x; i < padded_dim; i += blockDim.x) {
      data_t   d        = neighbor_vec[i] - query_vec[i];
      relative_vec[i]   = d;
      uint16_t bits     = __half_as_ushort(d);
      uint32_t abs_key  = bits & 0x7FFFu;
      uint32_t sign_bit = (bits >> 15) & 1u;
      keys[i] = (abs_key << MAG_SHIFT) | (sign_bit << SIGN_SHIFT) | (i & COORD_MASK);

      float f      = __half2float(d);
      l2sq_local  += f * f;
    }

    // ─── Block-reduce ||Δ||² → s_mag_sq ──────────────────────────────
    #pragma unroll
    for (int d = 16; d > 0; d >>= 1)
      l2sq_local += __shfl_xor_sync(0xFFFFFFFFu, l2sq_local, d);

    if (lane == 0) warp_scratch[wid] = __float_as_uint(l2sq_local);
    __syncthreads();

    __shared__ float s_mag_sq;
    if (wid == 0) {
      float v = (lane < nwarps) ? __uint_as_float(warp_scratch[lane]) : 0.0f;
      #pragma unroll
      for (int d = 16; d > 0; d >>= 1)
        v += __shfl_xor_sync(0xFFFFFFFFu, v, d);
      if (lane == 0) s_mag_sq = v;
    }
    __syncthreads();

    // ─── Top-k argmax + mask (unchanged) ─────────────────────────────
    for (uint8_t r = 0; r < k_ranks; ++r) {
      uint32_t local_max = 0;
      for (uint32_t i = threadIdx.x; i < padded_dim; i += blockDim.x) {
        const uint32_t k = keys[i];
        local_max = (k > local_max) ? k : local_max;
      }
      #pragma unroll
      for (int d = 16; d > 0; d >>= 1) {
        const uint32_t other = __shfl_xor_sync(0xFFFFFFFFu, local_max, d);
        local_max = (other > local_max) ? other : local_max;
      }
      if (lane == 0) warp_scratch[wid] = local_max;
      __syncthreads();
      if (wid == 0) {
        uint32_t v = (lane < nwarps) ? warp_scratch[lane] : 0;
        #pragma unroll
        for (int d = 16; d > 0; d >>= 1) {
          const uint32_t other = __shfl_xor_sync(0xFFFFFFFFu, v, d);
          v = (other > v) ? other : v;
        }
        if (lane == 0) warp_scratch[0] = v;
      }
      __syncthreads();
      const uint32_t block_max = warp_scratch[0];

      if (threadIdx.x == 0) {
        const uint32_t win_dim = block_max & COORD_MASK;
        // block_max's low (COORD_BITS+1) bits already match the packed_t coord|
        // sign layout, so casting to packed_t drops the magnitude bits and
        // yields the word directly — one write instead of two RMWs.
        g.set_lsh_packed(query_gid, static_cast<uint8_t>(neighbor_idx), r,
                         static_cast<packed_t>(block_max));
        keys[win_dim] = 0;
      }
      __syncthreads();
    }

    // ─── Store mag_sq for this edge as bf16 ──────────────────────────
    if (threadIdx.x == 0) {
      g.set_lsh_mag_sq(query_gid, static_cast<uint8_t>(neighbor_idx),
                       __float2bfloat16(s_mag_sq));
    }
    __syncthreads();
  }
}

}