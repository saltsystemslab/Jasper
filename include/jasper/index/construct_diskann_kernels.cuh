#pragma once

#include <cstdint>
#include <limits>

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "jasper/index/construct.cuh"

namespace jasper {

// ─── Center a chunk in place: v[i] -= mean ───────────────────────────────────
//
// Cross-polytope LSH assumes data centered at the origin. Datasets with a large
// non-zero mean (e.g. all-positive uint8) otherwise have one dominant direction
// shared by every vector, collapsing the argmax onto a single bucket. Applied to
// a temporary hashing copy only; the stored vectors are never modified.
template <typename DATA_T>
__global__ void subtract_mean_kernel(
    DATA_T* v,                            // [n * padded_dim], modified in place
    const DATA_T* __restrict__ mean,      // [dim]
    uint32_t n,
    uint32_t dim,
    uint32_t padded_dim
) {
  const size_t total = static_cast<size_t>(n) * dim;
  for (size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       tid < total; tid += static_cast<size_t>(gridDim.x) * blockDim.x) {
    const uint32_t d   = static_cast<uint32_t>(tid % dim);
    const size_t   row = tid / dim;
    v[row * padded_dim + d] = v[row * padded_dim + d] - mean[d];
  }
}

// ─── Polytope (cross-polytope) LSH shard assignment ──────────────────────────
//
// Cross-polytope LSH hashes a vector to the nearest vertex of the scaled
// cross-polytope {±e_i}: the vertex is the largest-magnitude coordinate with
// its sign. Nearby vectors share the same dominant coordinate, so they hash to
// the same bucket. To duplicate a point into `n_dup` shards we take its top
// `n_dup` coordinates by magnitude (multi-probe), each a signed vertex.
//
// bucket(coord, sign) = 2 * coord + sign  ∈ [0, 2 * dim)
// shard               = bucket % n_parts
//
// This mirrors the top-k argmax scheme in lsh_kernels.cuh, applied to the raw
// vector instead of an edge delta. One block per vector.
//
// The key packs |value| into the high bits and the coordinate index into the
// low 16 bits, so argmax over the packed key orders by magnitude first and
// breaks ties by index deterministically. Requires padded_dim <= 65536 so the
// index tiebreak never outranks a nonzero magnitude.
//
// Callers should hash a randomly rotated copy of the data: cross-polytope LSH
// is only balanced in an isotropic space (see assign_shards).
template <typename DATA_T, typename INDEX_T>
__global__ void polytope_lsh_assign_kernel(
    const DATA_T* __restrict__ vectors,  // [n_vectors * padded_dim]
    uint32_t n_vectors,
    uint32_t dim,
    uint32_t padded_dim,
    uint32_t n_parts,
    uint32_t n_dup,
    INDEX_T* __restrict__ shard_of       // [n_vectors * n_dup], INVALID for dup hits
) {
  static_assert(std::is_same_v<DATA_T, __half>,
                "polytope_lsh_assign_kernel requires data_t == __half");
  constexpr INDEX_T INVALID = std::numeric_limits<INDEX_T>::max();

  const uint32_t vid = blockIdx.x;
  if (vid >= n_vectors) return;

  extern __shared__ unsigned char smem_pl[];
  uint32_t* keys         = reinterpret_cast<uint32_t*>(smem_pl);   // [padded_dim]
  const uint32_t nwarps  = blockDim.x >> 5;
  uint32_t* warp_scratch = keys + padded_dim;                       // [nwarps]

  const DATA_T* v   = vectors + static_cast<size_t>(vid) * padded_dim;
  const uint32_t lane = threadIdx.x & 31u;
  const uint32_t wid  = threadIdx.x >> 5;

  // Pack keys for the real dimensions; pad lanes get 0.
  for (uint32_t i = threadIdx.x; i < padded_dim; i += blockDim.x) {
    uint32_t key = 0;
    if (i < dim) {
      uint16_t bits    = __half_as_ushort(v[i]);
      uint32_t abs_key = bits & 0x7FFFu;  // monotonic in |value| for IEEE half
      key = (abs_key << 16) | (i & 0xFFFFu);
    }
    keys[i] = key;
  }
  __syncthreads();

  __shared__ uint32_t s_winner;
  __shared__ uint32_t s_nsel;
  if (threadIdx.x == 0) s_nsel = 0;
  __syncthreads();

  for (uint32_t r = 0; r < n_dup; ++r) {
    // Block-wide argmax over the packed keys.
    uint32_t local_max = 0;
    for (uint32_t i = threadIdx.x; i < padded_dim; i += blockDim.x) {
      const uint32_t k = keys[i];
      local_max = (k > local_max) ? k : local_max;
    }
    #pragma unroll
    for (int d = 16; d > 0; d >>= 1) {
      const uint32_t o = __shfl_xor_sync(0xFFFFFFFFu, local_max, d);
      local_max = (o > local_max) ? o : local_max;
    }
    if (lane == 0) warp_scratch[wid] = local_max;
    __syncthreads();
    if (wid == 0) {
      uint32_t vv = (lane < nwarps) ? warp_scratch[lane] : 0;
      #pragma unroll
      for (int d = 16; d > 0; d >>= 1) {
        const uint32_t o = __shfl_xor_sync(0xFFFFFFFFu, vv, d);
        vv = (o > vv) ? o : vv;
      }
      if (lane == 0) s_winner = vv;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      const uint32_t win   = s_winner;
      const uint32_t coord = win & 0xFFFFu;
      const uint16_t bits  = __half_as_ushort(v[coord]);
      const uint32_t sign  = (bits >> 15) & 1u;
      const uint32_t bucket = 2u * coord + sign;
      INDEX_T shard = static_cast<INDEX_T>(bucket % n_parts);

      // Drop duplicates: a point already placed in this shard by an earlier
      // rank contributes it only once.
      bool dup = false;
      for (uint32_t t = 0; t < r; ++t) {
        if (shard_of[static_cast<size_t>(vid) * n_dup + t] == shard) { dup = true; break; }
      }
      shard_of[static_cast<size_t>(vid) * n_dup + r] = dup ? INVALID : shard;
      s_nsel = r + 1;
      keys[coord] = 0;  // mask the winner for the next rank
    }
    __syncthreads();
  }
}

// ─── Remap a freshly built shard's neighbor ids from local to global ─────────
//
// A shard graph is built over its own contiguous [0, shard_n) local ids; its
// stored neighbors are therefore local. This rewrites every neighbor in place
// to the global id via id_map (local -> global), on the device, so we never pay
// a host-side pass over ~n_dup*N edges. One thread per node (grid-stride).
template <typename GRAPH_CFG>
__global__ void remap_shard_edges_kernel(
    typename graph<GRAPH_CFG>::device_view g,
    const typename GRAPH_CFG::index_t* __restrict__ id_map,
    uint32_t n_nodes
) {
  using index_t = typename GRAPH_CFG::index_t;
  constexpr index_t INVALID = std::numeric_limits<index_t>::max();
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
       i < n_nodes; i += gridDim.x * blockDim.x) {
    uint8_t c = g.get_edge_count(i);
    for (uint8_t j = 0; j < c; j++) {
      index_t nb = g.get_neighbor(i, j);
      if (nb != INVALID) g.set_neighbor(i, j, id_map[nb]);
    }
  }
}

}  // namespace jasper
