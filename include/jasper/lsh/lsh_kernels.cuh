#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace jasper {

template <typename graph_cfg, typename graph_t>
__global__ void populate_lsh(graph_t g) {
  using index_t = typename graph_cfg::index_t;
  using data_t  = typename graph_cfg::data_t;
  constexpr uint32_t n_neighbors = graph_cfg::n_neighbors;

  static_assert(std::is_same_v<data_t, __half>,
              "populate_lsh requires graph_cfg::data_t to be __half "
              "(the |x| bit trick assumes IEEE fp16 layout)");

  // query_id is local to this partition; device_view APIs want global ids.
  const index_t query_lid = static_cast<index_t>(blockIdx.x);
  const index_t query_gid = query_lid + g.global_offset;

  const uint32_t padded_dim = g.get_padded_dim();
  const uint8_t  k_ranks    = graph_cfg::k_ranks;
  // Caller guarantees: padded_dim <= 32768 (15-bit dim field)
  //                    k_ranks    <= n_neighbors limits set on edge_lsh_list_t
  //                    blockDim.x is a multiple of 32

  // Shared memory layout, each region 16B-aligned:
  //   relative_vec[padded_dim]   bf16, kept for downstream stages
  //   keys[padded_dim]           uint32 packed, zeroed as winners are extracted
  //   warp_scratch[nwarps]       uint32 inter-warp slots
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

    // ---- Build Δ and packed keys ----
    // Key layout (max() gives argmax of |Δ|):
    //   [31:16] |Δ| as bf16 bits with sign masked off
    //   [15]    sign bit of Δ (1 = negative)
    //   [14:0]  dim index
    for (uint32_t i = threadIdx.x; i < padded_dim; i += blockDim.x) {
      data_t   d        = neighbor_vec[i] - query_vec[i];
      relative_vec[i]   = d;
      uint16_t bits     = __half_as_ushort(d);
      uint32_t abs_key  = bits & 0x7FFFu;
      uint32_t sign_bit = (bits >> 15) & 1u;
      keys[i] = (abs_key << 16) | (sign_bit << 15) | (i & 0x7FFFu);
    }
    __syncthreads();

    // ---- Extract top k_ranks via iterative argmax + mask ----
    for (uint8_t r = 0; r < k_ranks; ++r) {
      // 1. Per-thread max over its strided slice.
      uint32_t local_max = 0;
      for (uint32_t i = threadIdx.x; i < padded_dim; i += blockDim.x) {
        const uint32_t k = keys[i];
        local_max = (k > local_max) ? k : local_max;
      }

      // 2. Intra-warp reduction.
      #pragma unroll
      for (int delta = 16; delta > 0; delta >>= 1) {
        const uint32_t other = __shfl_xor_sync(0xFFFFFFFFu, local_max, delta);
        local_max = (other > local_max) ? other : local_max;
      }

      // 3. Inter-warp reduction.
      if (lane == 0) warp_scratch[wid] = local_max;
      __syncthreads();

      if (wid == 0) {
        uint32_t v = (lane < nwarps) ? warp_scratch[lane] : 0;
        #pragma unroll
        for (int delta = 16; delta > 0; delta >>= 1) {
          const uint32_t other = __shfl_xor_sync(0xFFFFFFFFu, v, delta);
          v = (other > v) ? other : v;
        }
        if (lane == 0) warp_scratch[0] = v;
      }
      __syncthreads();
      const uint32_t block_max = warp_scratch[0];

      // 4. Write rank-r winner and mask it out for the next iteration.
      if (threadIdx.x == 0) {
        const uint32_t win_dim = block_max & 0x7FFFu;
        const bool     neg     = (block_max >> 15) & 1u;
        g.set_lsh_coord(query_gid, static_cast<uint8_t>(neighbor_idx), r,
                            static_cast<uint16_t>(win_dim));
        g.set_lsh_sign (query_gid, static_cast<uint8_t>(neighbor_idx), r,
                            /*is_positive=*/ !neg);
        keys[win_dim] = 0;
      }
      __syncthreads();
    }
  }
}

}