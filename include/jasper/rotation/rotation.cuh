#pragma once

#include <cmath>
#include <cstring>
#include <random>
#include <vector>

#include <cublas_v2.h>

namespace jasper {

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

// Rotate data vector by applying the orthogonal rotation matrix.
// The input (d_data_vectors) and output (d_rotated_data_vectors) are
// in row major. The rotation matrix (d_P) is in column major.
//
// We use the transpose trick to make this work between different data orientation
// https://leimao.github.io/blog/cuBLAS-Transpose-Column-Major-Relationship/
__host__ inline void rotate_data_vec(
  cublasHandle_t handle,
  float *d_data_vectors,         // row major   
  float *d_rotated_data_vectors, // row major
  uint64_t n_data_vectors,
  uint32_t dim,
  float *d_P // (dim x dim) col major
) {
  const float alpha = 1.0f;
  const float beta  = 0.0f;
  cublasStatus_t stat = cublasSgemm(
      handle,
      CUBLAS_OP_T, CUBLAS_OP_N,
      dim,
      n_data_vectors,
      dim,
      &alpha,
      d_P, dim,
      d_data_vectors, dim,
      &beta,
      d_rotated_data_vectors, dim
  );
  if (stat != CUBLAS_STATUS_SUCCESS) {
    printf("cublasSgemm failed: %d\n", stat);
  }
  cudaDeviceSynchronize();
}

// rotate a single data vector.
__host__ inline void rotate_single_data_vec(
    cublasHandle_t handle,
    const float *d_data_vector,       // (dim)
    float *d_rotated_data_vector,     // (dim)
    uint32_t dim,
    const float *d_P                  // (dim x dim)
) {
  const float alpha = 1.0f;
  const float beta  = 0.0f;
  cublasStatus_t stat = cublasSgemv(
    handle,
    CUBLAS_OP_N,
    dim,              
    dim,              
    &alpha,
    d_P, dim,         
    d_data_vector, 1, 
    &beta,
    d_rotated_data_vector, 1
  );
  if (stat != CUBLAS_STATUS_SUCCESS) {
    printf("cublasSgemm failed: %d\n", stat);
  }
  cudaDeviceSynchronize();
}

} // namespace jasper