#pragma once

#include <math.h>
#include <cuda_fp16.h>
#include <curand_kernel.h>
#include <type_traits>
#include <limits>

namespace jasper {

template <uint32_t K_RANKS>
struct lsh_globals {
  float c_per_rank [K_RANKS];   // calibration constants
  float norm_denom;             // sum_r c_per_rank[r]^2
};

template <typename graph_cfg, typename graph_t>
__global__ void accumulate_rank_sums(
    graph_t g,
    float*    __restrict__ rank_sum,   // [k_ranks], pre-zeroed
    uint32_t* __restrict__ n_valid,    // scalar, pre-zeroed
    uint64_t  seed
) {
  using index_t = typename graph_cfg::index_t;
  using data_t  = typename graph_cfg::data_t;
  static_assert(std::is_same_v<data_t, __half>,
                "accumulate_rank_sums requires data_t == __half");

  constexpr uint8_t k_ranks = graph_cfg::k_ranks;
  constexpr index_t INVALID = std::numeric_limits<index_t>::max();

  const uint32_t padded_dim = g.get_padded_dim();

  // Shared layout (16B-aligned regions):
  //   keys[padded_dim]      uint32 packed keys
  //   warp_scratch[nwarps]  uint32 inter-warp slots (reused for L2 and max reductions)
  extern __shared__ unsigned char smem_lsh_global[];
  uint32_t *keys         = reinterpret_cast<uint32_t*>(smem_lsh_global);
  size_t    off          = (padded_dim * sizeof(uint32_t) + 15u) & ~15u;
  uint32_t *warp_scratch = reinterpret_cast<uint32_t*>(smem_lsh_global + off);

  __shared__ index_t s_query_gid;
  __shared__ index_t s_neighbor_gid;
  __shared__ float   s_inv_norm;   // 1 / ||Δ||  — calibrate against r̂, not Δ

  // ---- Sample a valid (query, neighbor) on thread 0 ----
  if (threadIdx.x == 0) {
    curandStatePhilox4_32_10_t rng;
    curand_init(seed, /*subsequence=*/blockIdx.x, /*offset=*/0, &rng);

    s_query_gid = INVALID;
    #pragma unroll 1
    for (int attempt = 0; attempt < 8; ++attempt) {
      const index_t qlid = static_cast<index_t>(curand(&rng) % g.n_vectors);
      const index_t qgid = qlid + g.global_offset;
      const uint8_t cnt  = g.get_edge_count(qgid);
      if (cnt == 0) continue;

      const uint8_t eidx = static_cast<uint8_t>(curand(&rng) % cnt);
      const index_t ngid = g.get_neighbor(qgid, eidx);
      if (ngid == INVALID) continue;

      s_query_gid    = qgid;
      s_neighbor_gid = ngid;
      break;
    }
  }
  __syncthreads();
  if (s_query_gid == INVALID) return;

  data_t *query_vec    = g.get_vector(s_query_gid);
  data_t *neighbor_vec = g.get_vector(s_neighbor_gid);

  const uint32_t lane   = threadIdx.x & 31u;
  const uint32_t wid    = threadIdx.x >> 5;
  const uint32_t nwarps = blockDim.x  >> 5;

  // ---- Pack Δ into uint32 keys AND accumulate ||Δ||² in one pass ----
  float l2sq_local = 0.0f;
  for (uint32_t i = threadIdx.x; i < padded_dim; i += blockDim.x) {
    data_t   d        = neighbor_vec[i] - query_vec[i];
    uint16_t bits     = __half_as_ushort(d);
    uint32_t abs_key  = bits & 0x7FFFu;
    uint32_t sign_bit = (bits >> 15) & 1u;
    keys[i] = (abs_key << 16) | (sign_bit << 15) | (i & 0x7FFFu);

    float f      = __half2float(d);
    l2sq_local  += f * f;
  }

  // ---- Block-reduce ||Δ||² → 1/||Δ|| in s_inv_norm ----
  #pragma unroll
  for (int delta = 16; delta > 0; delta >>= 1)
    l2sq_local += __shfl_xor_sync(0xFFFFFFFFu, l2sq_local, delta);

  if (lane == 0) warp_scratch[wid] = __float_as_uint(l2sq_local);
  __syncthreads();

  if (wid == 0) {
    float v = (lane < nwarps) ? __uint_as_float(warp_scratch[lane]) : 0.0f;
    #pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1)
      v += __shfl_xor_sync(0xFFFFFFFFu, v, delta);
    if (lane == 0) {
      // rsqrtf handles the 1/sqrt in one instruction.
      // +1e-20f guards against a degenerate Δ == 0 sample.
      s_inv_norm = rsqrtf(v + 1e-20f);
    }
  }
  __syncthreads();

  // Skip degenerate samples (query and neighbor identical) so they don't
  // poison the calibration with bogus inflated magnitudes.
  if (s_inv_norm > 1e9f) return;

  // ---- Top-k argmax + mask, accumulating |r̂[r]| = |Δ[r]| / ||Δ|| ----
  for (uint8_t r = 0; r < k_ranks; ++r) {
    uint32_t local_max = 0;
    for (uint32_t i = threadIdx.x; i < padded_dim; i += blockDim.x) {
      const uint32_t k = keys[i];
      local_max = (k > local_max) ? k : local_max;
    }

    #pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) {
      const uint32_t other = __shfl_xor_sync(0xFFFFFFFFu, local_max, delta);
      local_max = (other > local_max) ? other : local_max;
    }

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

    if (threadIdx.x == 0) {
      const uint16_t abs_bits = static_cast<uint16_t>(block_max >> 16);
      const float    mag_d    = __half2float(__ushort_as_half(abs_bits));
      // Normalize: |r̂[r]| = |Δ[r]| / ||Δ||.
      atomicAdd(&rank_sum[r], mag_d * s_inv_norm);

      const uint32_t win_dim = block_max & 0x7FFFu;
      keys[win_dim] = 0;
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) atomicAdd(n_valid, 1u);
}

template <uint32_t K_RANKS>
__global__ void finalize_lsh_globals(
    const float*    __restrict__ rank_sum,
    const uint32_t* __restrict__ n_valid,
    lsh_globals<K_RANKS>* __restrict__ out
) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;

  const float inv_n = 1.0f / static_cast<float>(*n_valid);
  float denom = 0.0f;
  #pragma unroll
  for (uint32_t r = 0; r < K_RANKS; ++r) {
    const float c = rank_sum[r] * inv_n;
    out->c_per_rank[r] = c;
    denom += c * c;
  }
  out->norm_denom = denom;
}

// template <typename graph_cfg>
// void generate_lsh_globals(
//     typename graph<graph_cfg>::device_view g,
//     lsh_globals<graph_cfg::k_ranks>* d_globals,   // device pointer
//     uint32_t      n_samples = 4096,
//     uint64_t      seed = 42,
//     cudaStream_t  stream = 0
// ) {
//   constexpr uint8_t  k_ranks       = graph_cfg::k_ranks;
//   constexpr uint32_t block_threads = 128;
//   static_assert(block_threads % 32 == 0);

//   float*    d_rank_sum = nullptr;
//   uint32_t* d_n_valid  = nullptr;
//   cudaMalloc(&d_rank_sum, k_ranks * sizeof(float));
//   cudaMalloc(&d_n_valid,  sizeof(uint32_t));
//   cudaMemsetAsync(d_rank_sum, 0, k_ranks * sizeof(float), stream);
//   cudaMemsetAsync(d_n_valid,  0, sizeof(uint32_t),        stream);

//   const uint32_t padded_dim = g.get_padded_dim();
//   const uint32_t nwarps     = block_threads / 32;
//   const size_t   smem_bytes =
//         ((padded_dim * sizeof(uint32_t) + 15) & ~15)
//       +  nwarps * sizeof(uint32_t);

//   accumulate_rank_sums<graph_cfg>
//       <<<n_samples, block_threads, smem_bytes, stream>>>(
//           g, d_rank_sum, d_n_valid, seed);

//   finalize_lsh_globals<k_ranks>
//       <<<1, 1, 0, stream>>>(d_rank_sum, d_n_valid, d_globals);

//   cudaStreamSynchronize(stream);   // before free; or use cudaFreeAsync
//   cudaFree(d_rank_sum);
//   cudaFree(d_n_valid);
// }

}