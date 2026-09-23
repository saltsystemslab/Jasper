#pragma once

#include <cmath>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <cublas_v2.h>
#include <cuda_fp16.h>

namespace jasper {

namespace detail {

template <typename T>
constexpr cudaDataType cublas_dtype() {
  if constexpr (std::is_same_v<T, float>)  return CUDA_R_32F;
  if constexpr (std::is_same_v<T, __half>) return CUDA_R_16F;
}

}  // namespace detail

// ── Householder QR on a column-major d×d matrix ────────────────
// Computes Q and R in-place. A is destroyed and replaced with R.
// Q is built by accumulating reflectors.
template <typename Scalar = float>
__host__ void householder_qr(int d, Scalar* A, Scalar* Q) {
  // Q = I
  std::memset(Q, 0, sizeof(Scalar) * d * d);
  for (int i = 0; i < d; i++) Q[i + i * d] = Scalar(1);

  std::vector<Scalar> v(d);

  for (int k = 0; k < d; k++) {
    // Extract column k below the diagonal
    Scalar norm_sq = 0;
    for (int i = k; i < d; i++) {
      v[i] = A[i + k * d];
      norm_sq += v[i] * v[i];
    }
    Scalar norm = std::sqrt(norm_sq);
    if (norm < Scalar(1e-12)) continue;

    // Choose sign to avoid cancellation
    Scalar sign = (A[k + k * d] >= 0) ? Scalar(1) : Scalar(-1);
    v[k] += sign * norm;

    // Recompute ||v||^2
    Scalar v_norm_sq = 0;
    for (int i = k; i < d; i++) v_norm_sq += v[i] * v[i];
    Scalar tau = Scalar(2) / v_norm_sq;

    // Apply H = I - tau * v * v^T to A (columns k..d-1)
    for (int j = k; j < d; j++) {
      Scalar dot = 0;
      for (int i = k; i < d; i++) dot += v[i] * A[i + j * d];
      for (int i = k; i < d; i++) A[i + j * d] -= tau * v[i] * dot;
    }

    // Apply H to Q (all columns)
    for (int j = 0; j < d; j++) {
      Scalar dot = 0;
      for (int i = k; i < d; i++) dot += v[i] * Q[i + j * d];
      for (int i = k; i < d; i++) Q[i + j * d] -= tau * v[i] * dot;
    }
  }
}

// ── Generate random orthogonal matrix ──────────────────────────
// Fills m[d×d] with Q and m_transpose[d×d] with Q^T (both column-major).
// Uses the standard method: QR decompose a random Gaussian matrix,
// then fix signs so the diagonal of R is positive (ensures uniqueness).
template <typename Scalar = float>
__host__ void set_rotation_matrix(
    int d,
    Scalar* m,
    Scalar* m_transpose,
    uint64_t seed = 0) {

  std::mt19937_64 rng(seed);
  std::normal_distribution<Scalar> N(0, 1);

  // A = random Gaussian matrix (column-major)
  std::vector<Scalar> A(d * d);
  for (int j = 0; j < d; j++)
    for (int i = 0; i < d; i++)
      A[i + j * d] = N(rng);

  // Q will hold the orthogonal matrix, A becomes R
  std::vector<Scalar> Q(d * d);
  householder_qr(d, A.data(), Q.data());

  // Fix signs: if R(i,i) < 0, flip column i of Q
  for (int i = 0; i < d; i++) {
    if (A[i + i * d] < Scalar(0)) {
      for (int j = 0; j < d; j++) {
        Q[j + i * d] *= Scalar(-1);
      }
    }
  }

  // Copy Q and Q^T to output (both column-major)
  for (int j = 0; j < d; j++) {
    for (int i = 0; i < d; i++) {
      m[i + j * d]           = Q[i + j * d];
      m_transpose[i + j * d] = Q[j + i * d];  // transpose: swap i,j
    }
  }
}

// ── Random D→d projection with orthonormal rows (the JLT P_d) ──
// "The random orthogonal matrix which projects a D-dimensional vector to
// d-dimensional space" of the RaBitQ ">32x compression" construction. Rows are
// Gaussian, orthonormalized by modified Gram-Schmidt, so P_d·P_dᵀ = I_d and
// E‖P_d x‖² = (d/D)·‖x‖².
//
// NOT used by the directional RaBitQ path, which gets the same distribution for
// free: the first d rows of the Haar-random R from set_rotation_matrix are
// themselves a uniform d-frame, so truncating a rotated vector IS a sample of
// P_d (see rabitq/edge_rq.cuh). This exists for the case that argument cannot
// cover — D large enough that materializing R at all is infeasible, since the
// Householder QR above is O(D³) in time and D² in space while this is O(D·d²)
// and D·d (123 KB vs 4.3 GB at D = 32768, d = 32).
//
// Output layout is (D x d) column-major — equivalently, row j of P_d occupies
// the contiguous span m[j*D .. j*D+D), which is also what a cuBLAS OP_T GEMM
// wants directly.
template <typename Scalar = float>
__host__ void set_projection_matrix(
    int D,
    int d,
    Scalar* m,
    uint64_t seed = 0) {

  if (d > D)
    throw std::runtime_error(
        "set_projection_matrix: d=" + std::to_string(d) +
        " exceeds D=" + std::to_string(D) + "; a projection cannot add rank");

  std::mt19937_64 rng(seed);
  std::normal_distribution<Scalar> N(0, 1);

  for (int j = 0; j < d; j++)
    for (int i = 0; i < D; i++)
      m[static_cast<size_t>(j) * D + i] = N(rng);

  // Modified Gram-Schmidt over the d rows.
  for (int j = 0; j < d; j++) {
    Scalar* rj = m + static_cast<size_t>(j) * D;
    for (int k = 0; k < j; k++) {
      const Scalar* rk = m + static_cast<size_t>(k) * D;
      Scalar dot = 0;
      for (int i = 0; i < D; i++) dot += rk[i] * rj[i];
      for (int i = 0; i < D; i++) rj[i] -= dot * rk[i];
    }
    Scalar norm_sq = 0;
    for (int i = 0; i < D; i++) norm_sq += rj[i] * rj[i];
    Scalar norm = std::sqrt(norm_sq);
    // d << D with Gaussian rows, so a dependent row is a measure-zero event;
    // fall back to a basis vector rather than dividing by ~0.
    if (norm < Scalar(1e-12)) {
      for (int i = 0; i < D; i++) rj[i] = Scalar(0);
      rj[j] = Scalar(1);
      continue;
    }
    const Scalar inv = Scalar(1) / norm;
    for (int i = 0; i < D; i++) rj[i] *= inv;
  }
}

// Rotate a batch of data vectors by applying the orthogonal rotation matrix.
// Input/output are row-major [n_data_vectors, dim], rotation matrix is
// column-major [dim, dim].
//
// We use the transpose trick to make this work between different data orientations
// https://leimao.github.io/blog/cuBLAS-Transpose-Column-Major-Relationship/
//
// Both `float` and `__half` instantiations route through cublasGemmEx with
// 32-bit accumulation for accuracy; the float overload below is kept for
// the legacy cublasSgemm path.
template <typename DataT>
__host__ inline void rotate_data_vec(
  cublasHandle_t handle,
  DataT *d_data_vectors,         // row major
  DataT *d_rotated_data_vectors, // row major
  uint64_t n_data_vectors,
  uint32_t dim,
  DataT *d_P                     // (dim x dim) col major
) {
  static_assert(std::is_same_v<DataT, float> || std::is_same_v<DataT, __half>,
                "rotate_data_vec only supports float or __half");

  const float alpha = 1.0f, beta = 0.0f;
  cudaDataType io_type = detail::cublas_dtype<DataT>();
  cublasStatus_t stat = cublasGemmEx(
      handle,
      CUBLAS_OP_T, CUBLAS_OP_N,
      dim, n_data_vectors, dim,
      &alpha,
      d_P,                    io_type, dim,
      d_data_vectors,         io_type, dim,
      &beta,
      d_rotated_data_vectors, io_type, dim,
      CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT);
  if (stat != CUBLAS_STATUS_SUCCESS) {
    printf("cublasGemmEx (rotate_data_vec) failed: %d\n", stat);
  }
  cudaDeviceSynchronize();
}

// Rotate a single data vector (the centroid path in rabitq quantize).
// Equivalent to a GEMV but expressed as a single-column GEMM so it can use
// the same Ex API.
template <typename DataT>
__host__ inline void rotate_single_data_vec(
    cublasHandle_t handle,
    const DataT *d_data_vector,    // (dim)
    DataT *d_rotated_data_vector,  // (dim)
    uint32_t dim,
    const DataT *d_P               // (dim x dim) col major
) {
  static_assert(std::is_same_v<DataT, float> || std::is_same_v<DataT, __half>,
                "rotate_single_data_vec only supports float or __half");

  const float alpha = 1.0f, beta = 0.0f;
  cudaDataType io_type = detail::cublas_dtype<DataT>();
  cublasStatus_t stat = cublasGemmEx(
      handle,
      CUBLAS_OP_N, CUBLAS_OP_N,
      dim, 1, dim,
      &alpha,
      d_P,                   io_type, dim,
      d_data_vector,         io_type, dim,
      &beta,
      d_rotated_data_vector, io_type, dim,
      CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT);
  if (stat != CUBLAS_STATUS_SUCCESS) {
    printf("cublasGemmEx (rotate_single_data_vec) failed: %d\n", stat);
  }
  cudaDeviceSynchronize();
}

} // namespace jasper