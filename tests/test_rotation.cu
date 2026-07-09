#include <gtest/gtest.h>
#include <cmath>
#include <vector>

#include "jasper/rotation/rotation.cuh"

namespace jasper::test {

// ── Helpers ────────────────────────────────────────────────────

// Multiply two column-major d×d matrices: C = A * B
void matmul(int d, const float* A, const float* B, float* C) {
  for (int i = 0; i < d; i++) {
    for (int j = 0; j < d; j++) {
      float sum = 0;
      for (int k = 0; k < d; k++) {
        sum += A[i + k * d] * B[k + j * d];
      }
      C[i + j * d] = sum;
    }
  }
}

// Check if matrix is identity (within tolerance)
void expect_identity(int d, const float* M, float tol, const char* label) {
  for (int i = 0; i < d; i++) {
    for (int j = 0; j < d; j++) {
      float expected = (i == j) ? 1.0f : 0.0f;
      EXPECT_NEAR(M[i + j * d], expected, tol)
          << label << "[" << i << "," << j << "]";
    }
  }
}

// Compute determinant via LU (simple, no pivoting — fine for orthogonal matrices)
float determinant(int d, const float* M) {
  std::vector<float> LU(M, M + d * d);
  float det = 1.0f;
  for (int k = 0; k < d; k++) {
    det *= LU[k + k * d];
    if (std::abs(LU[k + k * d]) < 1e-12f) return 0.0f;
    for (int i = k + 1; i < d; i++) {
      float factor = LU[i + k * d] / LU[k + k * d];
      for (int j = k + 1; j < d; j++) {
        LU[i + j * d] -= factor * LU[k + j * d];
      }
    }
  }
  return det;
}

// ── Tests ──────────────────────────────────────────────────────

class RotationTest : public ::testing::Test {};

TEST_F(RotationTest, QTransposeQ_IsIdentity) {
  // Q^T * Q should be I (orthogonality)
  constexpr int D = 16;
  std::vector<float> Q(D * D), Qt(D * D), result(D * D);

  jasper::set_rotation_matrix(D, Q.data(), Qt.data(), /*seed=*/42);

  matmul(D, Qt.data(), Q.data(), result.data());
  expect_identity(D, result.data(), 1e-5f, "Q^T * Q");
}

TEST_F(RotationTest, QQTranspose_IsIdentity) {
  // Q * Q^T should also be I
  constexpr int D = 16;
  std::vector<float> Q(D * D), Qt(D * D), result(D * D);

  jasper::set_rotation_matrix(D, Q.data(), Qt.data(), /*seed=*/42);

  matmul(D, Q.data(), Qt.data(), result.data());
  expect_identity(D, result.data(), 1e-5f, "Q * Q^T");
}

TEST_F(RotationTest, QtIsTransposeOfQ) {
  // Verify m_transpose is actually the transpose of m
  constexpr int D = 8;
  std::vector<float> Q(D * D), Qt(D * D);

  jasper::set_rotation_matrix(D, Q.data(), Qt.data(), /*seed=*/123);

  for (int i = 0; i < D; i++) {
    for (int j = 0; j < D; j++) {
      EXPECT_NEAR(Q[i + j * D], Qt[j + i * D], 1e-6f)
          << "Q[" << i << "," << j << "] != Qt[" << j << "," << i << "]";
    }
  }
}

TEST_F(RotationTest, Determinant_IsPlusOrMinusOne) {
  // An orthogonal matrix has det = +1 or -1
  // After sign correction, it should be +1 (proper rotation)
  constexpr int D = 16;
  std::vector<float> Q(D * D), Qt(D * D);

  jasper::set_rotation_matrix(D, Q.data(), Qt.data(), /*seed=*/42);

  float det = determinant(D, Q.data());
  EXPECT_NEAR(std::abs(det), 1.0f, 1e-4f)
      << "det(Q) = " << det;
}

TEST_F(RotationTest, PreservesNorm) {
  // Orthogonal matrix preserves vector norms: ||Qx|| = ||x||
  constexpr int D = 8;
  std::vector<float> Q(D * D), Qt(D * D);

  jasper::set_rotation_matrix(D, Q.data(), Qt.data(), /*seed=*/99);

  // Test vector
  std::vector<float> x = {1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<float> Qx(D, 0.0f);

  // Qx = Q * x (column-major)
  for (int i = 0; i < D; i++) {
    for (int j = 0; j < D; j++) {
      Qx[i] += Q[i + j * D] * x[j];
    }
  }

  float norm_x = 0, norm_Qx = 0;
  for (int i = 0; i < D; i++) {
    norm_x += x[i] * x[i];
    norm_Qx += Qx[i] * Qx[i];
  }

  EXPECT_NEAR(norm_Qx, norm_x, 1e-4f);
}

TEST_F(RotationTest, DeterministicWithSameSeed) {
  constexpr int D = 8;
  std::vector<float> Q1(D * D), Qt1(D * D);
  std::vector<float> Q2(D * D), Qt2(D * D);

  jasper::set_rotation_matrix(D, Q1.data(), Qt1.data(), /*seed=*/77);
  jasper::set_rotation_matrix(D, Q2.data(), Qt2.data(), /*seed=*/77);

  for (int i = 0; i < D * D; i++) {
    EXPECT_FLOAT_EQ(Q1[i], Q2[i]) << "Mismatch at index " << i;
  }
}

TEST_F(RotationTest, DifferentSeedsDifferentMatrices) {
  constexpr int D = 8;
  std::vector<float> Q1(D * D), Qt1(D * D);
  std::vector<float> Q2(D * D), Qt2(D * D);

  jasper::set_rotation_matrix(D, Q1.data(), Qt1.data(), /*seed=*/1);
  jasper::set_rotation_matrix(D, Q2.data(), Qt2.data(), /*seed=*/2);

  bool any_different = false;
  for (int i = 0; i < D * D; i++) {
    if (std::abs(Q1[i] - Q2[i]) > 1e-6f) {
      any_different = true;
      break;
    }
  }
  EXPECT_TRUE(any_different);
}

TEST_F(RotationTest, LargerDimension) {
  // Smoke test for a bigger matrix
  constexpr int D = 128;
  std::vector<float> Q(D * D), Qt(D * D), result(D * D);

  jasper::set_rotation_matrix(D, Q.data(), Qt.data(), /*seed=*/0);

  matmul(D, Qt.data(), Q.data(), result.data());
  expect_identity(D, result.data(), 1e-3f, "Q^T * Q (128x128)");
}

TEST_F(RotationTest, Dimension1) {
  // Edge case: 1×1 matrix should be +1 or -1
  constexpr int D = 1;
  float Q, Qt;

  jasper::set_rotation_matrix(D, &Q, &Qt, /*seed=*/42);

  EXPECT_NEAR(std::abs(Q), 1.0f, 1e-6f);
  EXPECT_FLOAT_EQ(Q, Qt);  // transpose of 1×1 is itself
}

} // namespace jasper::test