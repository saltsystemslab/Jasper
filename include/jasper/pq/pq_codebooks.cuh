#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

namespace jasper {

// The learned Product-Quantization model, shared by every edge of a graph.
//
// For M subspaces and K = 256 centroids each, the model stores:
//   centroids[(j*K + c)*dsub + d] = the d-th coordinate of centroid c of
//                                   subspace j (dsub = padded_dim / M).
//   cnorm[j*K + c]                = ||C_j[c]||²  (used to reconstruct ||e||²).
//
// This is the analogue of lsh_globals for the cross-polytope path, but it is
// far larger (256·padded_dim floats), so it lives in device memory and is
// passed to kernels through the lightweight `view` below.
template <uint32_t M, uint32_t K = 256>
struct pq_codebooks_view {
  const float* __restrict__ centroids;  // [M*K*dsub]
  const float* __restrict__ cnorm;       // [M*K]
  uint32_t                   dsub;        // padded_dim / M
};

template <uint32_t M, uint32_t K = 256>
struct pq_codebooks {
  static constexpr uint32_t m = M;
  static constexpr uint32_t k = K;

  float*   d_centroids = nullptr;  // device [M*K*dsub]
  float*   d_cnorm     = nullptr;  // device [M*K]
  uint32_t dsub        = 0;        // padded_dim / M

  __host__ pq_codebooks_view<M, K> view() const {
    return pq_codebooks_view<M, K>{d_centroids, d_cnorm, dsub};
  }

  __host__ size_t centroids_count() const {
    return static_cast<size_t>(M) * K * dsub;
  }

  // Allocate device storage for a given subspace dimension. Zero-initialized.
  __host__ static pq_codebooks allocate(uint32_t dsub) {
    auto check = [](cudaError_t err, const char* what) {
      if (err != cudaSuccess)
        throw std::runtime_error(std::string(what) + " failed: " +
                                 cudaGetErrorString(err));
    };
    pq_codebooks cb{};
    cb.dsub = dsub;
    check(cudaMalloc(&cb.d_centroids,
                     static_cast<size_t>(M) * K * dsub * sizeof(float)),
          "cudaMalloc(pq centroids)");
    check(cudaMalloc(&cb.d_cnorm, static_cast<size_t>(M) * K * sizeof(float)),
          "cudaMalloc(pq cnorm)");
    check(cudaMemset(cb.d_centroids, 0,
                     static_cast<size_t>(M) * K * dsub * sizeof(float)),
          "cudaMemset(pq centroids)");
    check(cudaMemset(cb.d_cnorm, 0, static_cast<size_t>(M) * K * sizeof(float)),
          "cudaMemset(pq cnorm)");
    return cb;
  }

  __host__ void free() {
    if (d_centroids) cudaFree(d_centroids);
    if (d_cnorm)     cudaFree(d_cnorm);
    d_centroids = nullptr;
    d_cnorm     = nullptr;
    dsub        = 0;
  }
};

}  // namespace jasper
