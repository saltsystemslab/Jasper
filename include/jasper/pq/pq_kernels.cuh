#pragma once

#include <cassert>
#include <cstdint>
#include <cfloat>

#include <cuda_fp16.h>
#include <curand_kernel.h>
#include <type_traits>
#include <limits>

namespace jasper {

// ─── small helpers ────────────────────────────────────────────────────────

// splitmix64-style hash, used to pick pseudo-random training points for
// centroid initialization and empty-cluster reseeding (Math.random is not
// available inside kernels and we want reproducibility from a seed).
static __device__ __forceinline__ uint64_t pq_hash(uint64_t x, uint64_t seed) {
  x += seed + 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}

// Block-wide argmin over (dist, idx), one candidate per thread. BLOCK must be a
// multiple of 32 and <= 1024. Ties break toward the smaller index for
// determinism. s_dist/s_idx must hold >= BLOCK/32 elements each.
template <uint32_t BLOCK>
__device__ __forceinline__ int pq_block_argmin(
    float my_dist, int my_idx, float* s_dist, int* s_idx) {
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    const float od = __shfl_xor_sync(0xFFFFFFFFu, my_dist, o);
    const int   oi = __shfl_xor_sync(0xFFFFFFFFu, my_idx,  o);
    if (od < my_dist || (od == my_dist && oi < my_idx)) { my_dist = od; my_idx = oi; }
  }
  const uint32_t lane = threadIdx.x & 31u;
  const uint32_t wid  = threadIdx.x >> 5;
  constexpr uint32_t NW = BLOCK / 32;
  if (lane == 0) { s_dist[wid] = my_dist; s_idx[wid] = my_idx; }
  __syncthreads();
  if (wid == 0) {
    float v  = (lane < NW) ? s_dist[lane] : FLT_MAX;
    int   vi = (lane < NW) ? s_idx[lane]  : 0x7FFFFFFF;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      const float od = __shfl_xor_sync(0xFFFFFFFFu, v,  o);
      const int   oi = __shfl_xor_sync(0xFFFFFFFFu, vi, o);
      if (od < v || (od == v && oi < vi)) { v = od; vi = oi; }
    }
    if (lane == 0) { s_dist[0] = v; s_idx[0] = vi; }
  }
  __syncthreads();
  return s_idx[0];
}

// ─── training: sample edge residuals ──────────────────────────────────────
// One block per training slot. Picks a random valid directed edge (u -> nb)
// and writes the residual (nb - u) as __half into X[p*padded_dim ..]. This is
// exactly the quantity the search must reconstruct (e in the law-of-cosines
// decomposition), so training on it teaches the codebooks the graph's real
// edge distribution.
template <typename graph_cfg, typename graph_t>
__global__ void pq_sample_residuals(
    graph_t g, __half* __restrict__ X, uint32_t n_train, uint64_t seed) {
  using index_t = typename graph_cfg::index_t;
  using data_t  = typename graph_cfg::data_t;
  static_assert(std::is_same_v<data_t, __half>,
                "pq_sample_residuals requires data_t == __half");

  const uint32_t p = blockIdx.x;
  if (p >= n_train) return;
  const uint32_t padded_dim = g.get_padded_dim();
  constexpr index_t INVALID = std::numeric_limits<index_t>::max();

  __shared__ index_t s_q, s_n;
  if (threadIdx.x == 0) {
    curandStatePhilox4_32_10_t rng;
    curand_init(seed, p, 0, &rng);
    s_q = INVALID;
    #pragma unroll 1
    for (int attempt = 0; attempt < 8; ++attempt) {
      const index_t qlid = static_cast<index_t>(curand(&rng) % g.n_vectors);
      const index_t qgid = qlid + g.global_offset;
      const uint8_t cnt  = g.get_edge_count(qgid);
      if (cnt == 0) continue;
      const uint8_t e    = static_cast<uint8_t>(curand(&rng) % cnt);
      const index_t ngid = g.get_neighbor(qgid, e);
      if (ngid == INVALID) continue;
      s_q = qgid; s_n = ngid; break;
    }
  }
  __syncthreads();

  __half* __restrict__ row = X + static_cast<size_t>(p) * padded_dim;
  if (s_q == INVALID) {
    for (uint32_t i = threadIdx.x; i < padded_dim; i += blockDim.x)
      row[i] = __float2half(0.0f);
    return;
  }
  const data_t* __restrict__ qv = g.get_vector(s_q);
  const data_t* __restrict__ nv = g.get_vector(s_n);
  for (uint32_t i = threadIdx.x; i < padded_dim; i += blockDim.x)
    row[i] = nv[i] - qv[i];
}

// ─── training: k-means (per-subspace Lloyd's) ─────────────────────────────

// Initialize every centroid from a pseudo-random training sub-vector.
// grid = M*K blocks (blockIdx = j*K + c), threads copy the dsub coords.
template <uint32_t M, uint32_t K>
__global__ void pq_kmeans_init(
    const __half* __restrict__ X, uint32_t n_train, uint32_t padded_dim,
    float* __restrict__ centroids, uint64_t seed) {
  const uint32_t dsub = padded_dim / M;
  const uint32_t jc   = blockIdx.x;
  const uint32_t j    = jc / K;
  const uint32_t r    = static_cast<uint32_t>(pq_hash(jc, seed) % n_train);
  const __half* __restrict__ xr =
      X + static_cast<size_t>(r) * padded_dim + static_cast<size_t>(j) * dsub;
  float* __restrict__ Cjc = centroids + static_cast<size_t>(jc) * dsub;
  for (uint32_t t = threadIdx.x; t < dsub; t += blockDim.x)
    Cjc[t] = __half2float(xr[t]);
}

// Assignment step: each training point's sub-vector goes to its nearest
// centroid. grid = n_train blocks, blockDim = K (one thread per centroid).
// smem: padded_dim floats (residual) + K/32 floats + K/32 ints (reduction).
template <uint32_t M, uint32_t K>
__global__ void pq_kmeans_assign(
    const __half* __restrict__ X, uint32_t n_train, uint32_t padded_dim,
    const float* __restrict__ centroids, uint8_t* __restrict__ labels) {
  const uint32_t dsub = padded_dim / M;
  const uint32_t p    = blockIdx.x;
  if (p >= n_train) return;

  extern __shared__ float pq_smem[];
  float* s_res  = pq_smem;
  float* s_dist = s_res + padded_dim;
  int*   s_idx  = reinterpret_cast<int*>(s_dist + (K / 32));

  const __half* __restrict__ xrow = X + static_cast<size_t>(p) * padded_dim;
  for (uint32_t i = threadIdx.x; i < padded_dim; i += blockDim.x)
    s_res[i] = __half2float(xrow[i]);
  __syncthreads();

  const uint32_t c = threadIdx.x;  // one centroid per thread (blockDim == K)
  for (uint32_t j = 0; j < M; ++j) {
    const float* __restrict__ Cj =
        centroids + (static_cast<size_t>(j) * K + c) * dsub;
    const float* __restrict__ rj = s_res + static_cast<size_t>(j) * dsub;
    float d = 0.0f;
    for (uint32_t t = 0; t < dsub; ++t) { const float df = rj[t] - Cj[t]; d += df * df; }
    const int best = pq_block_argmin<K>(d, static_cast<int>(c), s_dist, s_idx);
    if (threadIdx.x == 0)
      labels[static_cast<size_t>(p) * M + j] = static_cast<uint8_t>(best);
    __syncthreads();
  }
}

// Accumulate per-centroid coordinate sums and point counts (for the mean).
// grid = n_train, blockDim arbitrary; atomics into sum[M*K*dsub], count[M*K].
template <uint32_t M, uint32_t K>
__global__ void pq_kmeans_accumulate(
    const __half* __restrict__ X, uint32_t n_train, uint32_t padded_dim,
    const uint8_t* __restrict__ labels,
    float* __restrict__ sum, uint32_t* __restrict__ count) {
  const uint32_t dsub = padded_dim / M;
  const uint32_t p    = blockIdx.x;
  if (p >= n_train) return;

  const __half*  __restrict__ xrow = X + static_cast<size_t>(p) * padded_dim;
  const uint8_t* __restrict__ lab  = labels + static_cast<size_t>(p) * M;
  for (uint32_t i = threadIdx.x; i < padded_dim; i += blockDim.x) {
    const uint32_t j   = i / dsub;
    const uint32_t off = i - j * dsub;
    const uint8_t  c   = lab[j];
    atomicAdd(&sum[(static_cast<size_t>(j) * K + c) * dsub + off],
              __half2float(xrow[i]));
    if (off == 0) atomicAdd(&count[j * K + c], 1u);
  }
}

// Update step: centroid = mean of assigned points, or reseed from a random
// training sub-vector if the cluster is empty. grid = M*K blocks.
template <uint32_t M, uint32_t K>
__global__ void pq_kmeans_finalize(
    const __half* __restrict__ X, uint32_t n_train, uint32_t padded_dim,
    const float* __restrict__ sum, const uint32_t* __restrict__ count,
    float* __restrict__ centroids, uint64_t seed, uint32_t iter) {
  const uint32_t dsub = padded_dim / M;
  const uint32_t jc   = blockIdx.x;
  const uint32_t j    = jc / K;
  const uint32_t cnt  = count[jc];
  float* __restrict__ Cjc = centroids + static_cast<size_t>(jc) * dsub;

  if (cnt > 0) {
    const float inv = 1.0f / static_cast<float>(cnt);
    const float* __restrict__ Sjc = sum + static_cast<size_t>(jc) * dsub;
    for (uint32_t t = threadIdx.x; t < dsub; t += blockDim.x)
      Cjc[t] = Sjc[t] * inv;
  } else {
    const uint32_t r = static_cast<uint32_t>(
        pq_hash(jc ^ (static_cast<uint64_t>(iter) * 0x9E3779B9u), seed) % n_train);
    const __half* __restrict__ xr =
        X + static_cast<size_t>(r) * padded_dim + static_cast<size_t>(j) * dsub;
    for (uint32_t t = threadIdx.x; t < dsub; t += blockDim.x)
      Cjc[t] = __half2float(xr[t]);
  }
}

// Precompute per-vector squared L2 norm ||x||² for every graph vector, indexed
// by local id (gid - global_offset). grid = n_vectors blocks; block-reduce over
// padded_dim (padding coords are zero, so they don't affect the sum). Used by
// the PQ L2 estimator, which reconstructs ||q-v||² with the EXACT ||v||² rather
// than rebuilding a residual LUT per popped node.
template <typename graph_cfg, typename graph_t>
__global__ void pq_compute_vector_norms(graph_t g, float* __restrict__ norms) {
  using index_t = typename graph_cfg::index_t;
  const index_t  lid        = static_cast<index_t>(blockIdx.x);
  const index_t  gid        = lid + g.global_offset;
  const uint32_t padded_dim = g.get_padded_dim();
  const auto* __restrict__ x = g.get_vector(gid);

  float local = 0.0f;
  for (uint32_t i = threadIdx.x; i < padded_dim; i += blockDim.x) {
    const float v = __half2float(x[i]); local += v * v;
  }
  #pragma unroll
  for (int o = 16; o > 0; o >>= 1)
    local += __shfl_xor_sync(0xFFFFFFFFu, local, o);

  __shared__ float s_warp[32];
  const uint32_t lane = threadIdx.x & 31u;
  const uint32_t wid  = threadIdx.x >> 5;
  const uint32_t nw   = (blockDim.x + 31u) / 32u;
  if (lane == 0) s_warp[wid] = local;
  __syncthreads();
  if (wid == 0) {
    float v = (lane < nw) ? s_warp[lane] : 0.0f;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
      v += __shfl_xor_sync(0xFFFFFFFFu, v, o);
    if (lane == 0) norms[lid] = v;
  }
}

// ─── encode: assign every graph edge to its PQ code ───────────────────────
// grid = n_vectors blocks, blockDim = K. Mirrors pq_kmeans_assign but loops
// over a node's out-edges and writes the M-byte code into edge_pqs.
template <typename graph_cfg, typename graph_t, uint32_t M, uint32_t K>
__global__ void pq_populate(graph_t g, const float* __restrict__ centroids) {
  using index_t = typename graph_cfg::index_t;
  using data_t  = typename graph_cfg::data_t;
  static_assert(std::is_same_v<data_t, __half>,
                "pq_populate requires data_t == __half");
  constexpr uint32_t n_neighbors = graph_cfg::n_neighbors;

  const uint32_t padded_dim = g.get_padded_dim();
  const uint32_t dsub       = padded_dim / M;
  const index_t  u_lid      = static_cast<index_t>(blockIdx.x);
  const index_t  u_gid      = u_lid + g.global_offset;

  extern __shared__ float pq_smem[];
  float* s_res  = pq_smem;
  float* s_dist = s_res + padded_dim;
  int*   s_idx  = reinterpret_cast<int*>(s_dist + (K / 32));

  const data_t* __restrict__ uv = g.get_vector(u_gid);
  const uint8_t cnt = g.get_edge_count(u_gid);

  for (uint32_t e = 0; e < n_neighbors; ++e) {
    if (e >= cnt) break;
    const index_t nb = g.get_neighbor(u_gid, static_cast<uint8_t>(e));
    const data_t* __restrict__ nv = g.get_vector(nb);

    for (uint32_t i = threadIdx.x; i < padded_dim; i += blockDim.x)
      s_res[i] = __half2float(nv[i] - uv[i]);
    __syncthreads();

    const uint32_t c = threadIdx.x;  // one centroid per thread (blockDim == K)
    for (uint32_t j = 0; j < M; ++j) {
      const float* __restrict__ Cj =
          centroids + (static_cast<size_t>(j) * K + c) * dsub;
      const float* __restrict__ rj = s_res + static_cast<size_t>(j) * dsub;
      float d = 0.0f;
      for (uint32_t t = 0; t < dsub; ++t) { const float df = rj[t] - Cj[t]; d += df * df; }
      const int best = pq_block_argmin<K>(d, static_cast<int>(c), s_dist, s_idx);
      if (threadIdx.x == 0)
        g.set_pq_code(u_gid, static_cast<uint8_t>(e), j, static_cast<uint8_t>(best));
      __syncthreads();
    }
  }
}

}  // namespace jasper
