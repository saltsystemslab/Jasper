#include <gtest/gtest.h>
#include "helpers.cuh"
#include "jasper/rabitq/distance.cuh"
#include "jasper/rabitq/layout.cuh"
#include "jasper/rabitq/quantize.cuh"
#include "jasper/rotation/rotation.cuh"

#include <cooperative_groups.h>
#include <cublas_v2.h>

#include <cmath>
#include <numeric>
#include <random>
#include <vector>

namespace jasper::test {

// ═══════════════════════════════════════════════════════════════
// Bit extraction tests (host-only, no GPU needed)
// ═══════════════════════════════════════════════════════════════

class BitExtractionTest : public ::testing::Test {};

TEST_F(BitExtractionTest, Extract1Bit_AllZeros) {
  uint8_t code[2] = {0x00, 0x00};
  for (uint32_t i = 0; i < 16; i++) {
    EXPECT_EQ(extract_1bit(code, i), 0u);
  }
}

TEST_F(BitExtractionTest, Extract1Bit_AllOnes) {
  uint8_t code[2] = {0xFF, 0xFF};
  for (uint32_t i = 0; i < 16; i++) {
    EXPECT_EQ(extract_1bit(code, i), 1u);
  }
}

TEST_F(BitExtractionTest, Extract1Bit_Alternating) {
  // 0b10101010 = 0xAA → bits 1,3,5,7 are set
  uint8_t code[1] = {0xAA};
  EXPECT_EQ(extract_1bit(code, 0), 0u);
  EXPECT_EQ(extract_1bit(code, 1), 1u);
  EXPECT_EQ(extract_1bit(code, 2), 0u);
  EXPECT_EQ(extract_1bit(code, 3), 1u);
  EXPECT_EQ(extract_1bit(code, 4), 0u);
  EXPECT_EQ(extract_1bit(code, 5), 1u);
  EXPECT_EQ(extract_1bit(code, 6), 0u);
  EXPECT_EQ(extract_1bit(code, 7), 1u);
}

TEST_F(BitExtractionTest, Extract2Bit_KnownValues) {
  // byte = 0b11_10_01_00 = 0xE4
  // dim 0 → 00, dim 1 → 01, dim 2 → 10, dim 3 → 11
  uint8_t code[1] = {0xE4};
  EXPECT_EQ(extract_2bit(code, 0), 0u);
  EXPECT_EQ(extract_2bit(code, 1), 1u);
  EXPECT_EQ(extract_2bit(code, 2), 2u);
  EXPECT_EQ(extract_2bit(code, 3), 3u);
}

TEST_F(BitExtractionTest, Extract4Bit_KnownValues) {
  // byte = 0xA3 → low nibble=3, high nibble=A(10)
  uint8_t code[1] = {0xA3};
  EXPECT_EQ(extract_4bit(code, 0), 3u);
  EXPECT_EQ(extract_4bit(code, 1), 10u);
}

TEST_F(BitExtractionTest, GenericMatchesSpecialized_1bit) {
  std::mt19937 rng(42);
  uint8_t code[16];
  for (auto& b : code) b = rng() & 0xFF;

  for (uint32_t i = 0; i < 128; i++) {
    EXPECT_EQ(extract_bits(code, i, 1), extract_1bit(code, i))
        << "Mismatch at dim " << i;
  }
}

TEST_F(BitExtractionTest, GenericMatchesSpecialized_2bit) {
  std::mt19937 rng(42);
  uint8_t code[16];
  for (auto& b : code) b = rng() & 0xFF;

  for (uint32_t i = 0; i < 64; i++) {
    EXPECT_EQ(extract_bits(code, i, 2), extract_2bit(code, i))
        << "Mismatch at dim " << i;
  }
}

TEST_F(BitExtractionTest, GenericMatchesSpecialized_4bit) {
  std::mt19937 rng(42);
  uint8_t code[16];
  for (auto& b : code) b = rng() & 0xFF;

  for (uint32_t i = 0; i < 32; i++) {
    EXPECT_EQ(extract_bits(code, i, 4), extract_4bit(code, i))
        << "Mismatch at dim " << i;
  }
}

// ═══════════════════════════════════════════════════════════════
// rabitq_data_store layout tests
// ═══════════════════════════════════════════════════════════════

class DataStoreTest : public ::testing::Test {};

TEST_F(DataStoreTest, NodeBytesCorrect_1bit) {
  rabitq_data_store<float> s;
  s.dim = 128;
  s.bits_per_dim = 1;
  // code = 128/8 = 16 bytes, + 2 floats (add, rescale) = 16+8 = 24
  EXPECT_EQ(s.code_bytes(), 16u);
  EXPECT_EQ(s.node_bytes(), 24u);
}

TEST_F(DataStoreTest, NodeBytesCorrect_2bit) {
  rabitq_data_store<float> s;
  s.dim = 128;
  s.bits_per_dim = 2;
  // code = 256/8 = 32, + 8 = 40
  EXPECT_EQ(s.code_bytes(), 32u);
  EXPECT_EQ(s.node_bytes(), 40u);
}

TEST_F(DataStoreTest, NodeBytesCorrect_4bit) {
  rabitq_data_store<float> s;
  s.dim = 96;
  s.bits_per_dim = 4;
  // code = 384/8 = 48, + 8 = 56
  EXPECT_EQ(s.code_bytes(), 48u);
  EXPECT_EQ(s.node_bytes(), 56u);
}

TEST_F(DataStoreTest, NodeBytesOddDim) {
  rabitq_data_store<float> s;
  s.dim = 97;
  s.bits_per_dim = 1;
  // code = (97+7)/8 = 13 bytes (then pad to 16 bytes), + 8 = 24
  EXPECT_EQ(s.code_bytes(), 16u);
  EXPECT_EQ(s.node_bytes(), 24u);
}

TEST_F(DataStoreTest, TotalBytesCorrect) {
  rabitq_data_store<float> s;
  s.dim = 128;
  s.bits_per_dim = 1;
  s.n_vectors = 1000;
  EXPECT_EQ(s.total_bytes(), 24000u);
}

TEST_F(DataStoreTest, HostAllocateAndAccess) {
  auto store = rabitq_data_store<float>::allocate_host(10, 64, 1);
  ASSERT_NE(store.data, nullptr);
  EXPECT_EQ(store.n_vectors, 10u);
  EXPECT_EQ(store.dim, 64u);
  EXPECT_EQ(store.bits_per_dim, 1u);

  // Write to node 0
  auto node = store[0];
  *node.f_add = 1.5f;
  *node.f_rescale = 2.5f;
  node.bin_code[0] = 0xAB;

  // Read back
  auto node_r = store[0];
  EXPECT_FLOAT_EQ(*node_r.f_add, 1.5f);
  EXPECT_FLOAT_EQ(*node_r.f_rescale, 2.5f);
  EXPECT_EQ(node_r.bin_code[0], 0xAB);

  // Write to node 9, verify no overlap with node 0
  auto node9 = store[9];
  *node9.f_add = 99.0f;
  EXPECT_FLOAT_EQ(*store[0].f_add, 1.5f);  // unchanged
  EXPECT_FLOAT_EQ(*store[9].f_add, 99.0f);

  cudaFreeHost(store.data);
}

TEST_F(DataStoreTest, HostToDeviceRoundtrip) {
  ASSERT_EQ(cudaSetDevice(0), cudaSuccess);

  auto h_store = rabitq_data_store<float>::allocate_host(5, 32, 1);
  for (uint32_t i = 0; i < 5; i++) {
    auto node = h_store[i];
    *node.f_add = static_cast<float>(i) * 10.0f;
    *node.f_rescale = static_cast<float>(i) * 0.1f;
    for (uint32_t j = 0; j < h_store.code_bytes(); j++) {
      node.bin_code[j] = static_cast<uint8_t>(i + j);
    }
  }

  auto d_store = h_store.to_device();
  auto h_back = d_store.to_host();

  for (uint32_t i = 0; i < 5; i++) {
    auto orig = h_store[i];
    auto back = h_back[i];
    EXPECT_FLOAT_EQ(*orig.f_add, *back.f_add) << "Node " << i;
    EXPECT_FLOAT_EQ(*orig.f_rescale, *back.f_rescale) << "Node " << i;
    for (uint32_t j = 0; j < h_store.code_bytes(); j++) {
      EXPECT_EQ(orig.bin_code[j], back.bin_code[j])
          << "Node " << i << " byte " << j;
    }
  }

  cudaFreeHost(h_store.data);
  cudaFree(d_store.data);
  cudaFreeHost(h_back.data);
}

// ═══════════════════════════════════════════════════════════════
// Quantization tests (require GPU)
// ═══════════════════════════════════════════════════════════════

class QuantizeTest : public ::testing::Test {
protected:
  void SetUp() override { CUDA_CHECK(cudaSetDevice(0)); }
  void TearDown() override { CUDA_CHECK(cudaDeviceSynchronize()); }
};

TEST_F(QuantizeTest, SmallDataset_1bit_Runs) {
  constexpr uint32_t N = 16;
  constexpr uint32_t DIM = 32;

  // Random vectors
  std::mt19937 rng(42);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  std::vector<float> h_vecs(N * DIM);
  for (auto& v : h_vecs) v = dist(rng);

  DeviceBuf<float> d_vecs(h_vecs);
  vector_view<float> vecs(d_vecs.ptr, DIM, N, false);

  // Centroid = mean
  std::vector<float> h_centroid(DIM, 0.0f);
  for (uint32_t i = 0; i < N; i++)
    for (uint32_t j = 0; j < DIM; j++)
      h_centroid[j] += h_vecs[i * DIM + j];
  for (auto& c : h_centroid) c /= N;
  DeviceBuf<float> d_centroid(h_centroid);

  // Identity rotation
  std::vector<float> h_rot(DIM * DIM, 0.0f);
  for (uint32_t i = 0; i < DIM; i++) h_rot[i + i * DIM] = 1.0f;
  DeviceBuf<float> d_rot(h_rot);

  auto store = rabitq_quantize<1>(vecs, d_rot.ptr, d_centroid.ptr, METRIC_L2);

  // Basic checks
  EXPECT_EQ(store.n_vectors, N);
  EXPECT_EQ(store.dim, DIM);
  EXPECT_EQ(store.bits_per_dim, 1u);

  // Copy back and verify f_add, f_rescale are finite
  auto h_store = store.to_host();
  for (uint32_t i = 0; i < N; i++) {
    auto node = h_store[i];
    EXPECT_TRUE(std::isfinite(*node.f_add))
        << "Node " << i << " f_add=" << *node.f_add;
    EXPECT_TRUE(std::isfinite(*node.f_rescale))
        << "Node " << i << " f_rescale=" << *node.f_rescale;
  }

  cudaFree(store.data);
  cudaFreeHost(h_store.data);
}

TEST_F(QuantizeTest, BinaryCodes_Are1Bit) {
  // With 1-bit quantization, every extracted bit should be 0 or 1
  constexpr uint32_t N = 8;
  constexpr uint32_t DIM = 64;

  std::mt19937 rng(99);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  std::vector<float> h_vecs(N * DIM);
  for (auto& v : h_vecs) v = dist(rng);
  DeviceBuf<float> d_vecs(h_vecs);
  vector_view<float> vecs(d_vecs.ptr, DIM, N, false);

  std::vector<float> h_centroid(DIM, 0.0f);
  DeviceBuf<float> d_centroid(h_centroid);

  std::vector<float> h_rot(DIM * DIM, 0.0f);
  for (uint32_t i = 0; i < DIM; i++) h_rot[i + i * DIM] = 1.0f;
  DeviceBuf<float> d_rot(h_rot);

  auto store = rabitq_quantize<1>(vecs, d_rot.ptr, d_centroid.ptr, METRIC_L2);
  auto h_store = store.to_host();

  for (uint32_t i = 0; i < N; i++) {
    auto node = h_store[i];
    for (uint32_t d = 0; d < DIM; d++) {
      uint8_t bit = extract_1bit(node.bin_code, d);
      EXPECT_LE(bit, 1u) << "Node " << i << " dim " << d;
    }
  }

  cudaFree(store.data);
  cudaFreeHost(h_store.data);
}

TEST_F(QuantizeTest, Rescale_IsNegative_L2) {
  // For L2 metric, rescale should be negative (= ipnorm_inv * -2 * l2_norm)
  constexpr uint32_t N = 8;
  constexpr uint32_t DIM = 32;

  std::mt19937 rng(77);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  std::vector<float> h_vecs(N * DIM);
  for (auto& v : h_vecs) v = dist(rng);
  DeviceBuf<float> d_vecs(h_vecs);
  vector_view<float> vecs(d_vecs.ptr, DIM, N, false);

  std::vector<float> h_centroid(DIM, 0.0f);
  DeviceBuf<float> d_centroid(h_centroid);

  std::vector<float> h_rot(DIM * DIM, 0.0f);
  for (uint32_t i = 0; i < DIM; i++) h_rot[i + i * DIM] = 1.0f;
  DeviceBuf<float> d_rot(h_rot);

  auto store = rabitq_quantize<1>(vecs, d_rot.ptr, d_centroid.ptr, METRIC_L2);
  auto h_store = store.to_host();

  for (uint32_t i = 0; i < N; i++) {
    auto node = h_store[i];
    EXPECT_LT(*node.f_rescale, 0.0f)
        << "Node " << i << " rescale should be negative for L2";
  }

  cudaFree(store.data);
  cudaFreeHost(h_store.data);
}

TEST_F(QuantizeTest, IdenticalVectors_SameQuantization) {
  constexpr uint32_t N = 4;
  constexpr uint32_t DIM = 16;

  // All vectors identical
  std::vector<float> h_vecs(N * DIM);
  for (uint32_t i = 0; i < N; i++)
    for (uint32_t j = 0; j < DIM; j++)
      h_vecs[i * DIM + j] = static_cast<float>(j + 1);

  DeviceBuf<float> d_vecs(h_vecs);
  vector_view<float> vecs(d_vecs.ptr, DIM, N, false);

  std::vector<float> h_centroid(DIM, 0.0f);
  DeviceBuf<float> d_centroid(h_centroid);

  std::vector<float> h_rot(DIM * DIM, 0.0f);
  for (uint32_t i = 0; i < DIM; i++) h_rot[i + i * DIM] = 1.0f;
  DeviceBuf<float> d_rot(h_rot);

  auto store = rabitq_quantize<1>(vecs, d_rot.ptr, d_centroid.ptr, METRIC_L2);
  auto h_store = store.to_host();

  // All nodes should have identical f_add, f_rescale, and codes
  auto ref = h_store[0];
  for (uint32_t i = 1; i < N; i++) {
    auto node = h_store[i];
    EXPECT_FLOAT_EQ(*ref.f_add, *node.f_add) << "Node " << i;
    EXPECT_FLOAT_EQ(*ref.f_rescale, *node.f_rescale) << "Node " << i;
    for (uint32_t j = 0; j < h_store.code_bytes(); j++) {
      EXPECT_EQ(ref.bin_code[j], node.bin_code[j])
          << "Node " << i << " byte " << j;
    }
  }

  cudaFree(store.data);
  cudaFreeHost(h_store.data);
}

TEST_F(QuantizeTest, WithRotation_StillFinite) {
  constexpr uint32_t N = 8;
  constexpr uint32_t DIM = 32;

  std::mt19937 rng(123);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  std::vector<float> h_vecs(N * DIM);
  for (auto& v : h_vecs) v = dist(rng);
  DeviceBuf<float> d_vecs(h_vecs);
  vector_view<float> vecs(d_vecs.ptr, DIM, N, false);

  std::vector<float> h_centroid(DIM, 0.0f);
  DeviceBuf<float> d_centroid(h_centroid);

  // Random orthogonal rotation
  std::vector<float> Q(DIM * DIM), Qt(DIM * DIM);
  set_rotation_matrix(DIM, Q.data(), Qt.data(), 42);
  DeviceBuf<float> d_rot(Q);

  auto store = rabitq_quantize<1>(vecs, d_rot.ptr, d_centroid.ptr, METRIC_L2);
  auto h_store = store.to_host();

  for (uint32_t i = 0; i < N; i++) {
    auto node = h_store[i];
    EXPECT_TRUE(std::isfinite(*node.f_add)) << "Node " << i;
    EXPECT_TRUE(std::isfinite(*node.f_rescale)) << "Node " << i;
    EXPECT_NE(*node.f_rescale, 0.0f) << "Node " << i;
  }

  cudaFree(store.data);
  cudaFreeHost(h_store.data);
}

TEST_F(QuantizeTest, InnerProduct_DifferentFactors) {
  constexpr uint32_t N = 8;
  constexpr uint32_t DIM = 32;

  std::mt19937 rng(55);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  std::vector<float> h_vecs(N * DIM);
  for (auto& v : h_vecs) v = dist(rng);
  DeviceBuf<float> d_vecs(h_vecs);
  vector_view<float> vecs(d_vecs.ptr, DIM, N, false);

  std::vector<float> h_centroid(DIM, 0.0f);
  DeviceBuf<float> d_centroid(h_centroid);

  std::vector<float> h_rot(DIM * DIM, 0.0f);
  for (uint32_t i = 0; i < DIM; i++) h_rot[i + i * DIM] = 1.0f;
  DeviceBuf<float> d_rot(h_rot);

  auto store_l2 = rabitq_quantize<1>(vecs, d_rot.ptr, d_centroid.ptr, METRIC_L2);
  auto store_ip = rabitq_quantize<1>(vecs, d_rot.ptr, d_centroid.ptr, METRIC_IP);

  auto h_l2 = store_l2.to_host();
  auto h_ip = store_ip.to_host();

  // L2 and IP should produce different f_add values
  bool any_different = false;
  for (uint32_t i = 0; i < N; i++) {
    if (std::abs(*h_l2[i].f_add - *h_ip[i].f_add) > 1e-6f) {
      any_different = true;
      break;
    }
  }
  EXPECT_TRUE(any_different) << "L2 and IP should produce different factors";

  cudaFree(store_l2.data);
  cudaFree(store_ip.data);
  cudaFreeHost(h_l2.data);
  cudaFreeHost(h_ip.data);
}

TEST_F(QuantizeTest, RotateSingle_MatchesBatchRotation) {
  // The centroid path must rotate the same way as the data/query path (P^T x),
  // or residuals are taken around the wrong point.
  constexpr uint32_t DIM = 48;

  std::mt19937 rng(11);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  std::vector<float> h_x(DIM);
  for (auto& v : h_x) v = dist(rng);
  std::vector<float> Q(DIM * DIM), Qt(DIM * DIM);
  set_rotation_matrix(DIM, Q.data(), Qt.data(), 5);

  DeviceBuf<float> d_x(h_x), d_P(Q), d_batch(DIM), d_single(DIM);
  cublasHandle_t handle;
  cublasCreate(&handle);
  rotate_data_vec(handle, d_x.ptr, d_batch.ptr, 1, DIM, d_P.ptr);
  rotate_single_data_vec(handle, d_x.ptr, d_single.ptr, DIM, d_P.ptr);
  cublasDestroy(handle);

  auto batch = d_batch.to_host();
  auto single = d_single.to_host();
  for (uint32_t i = 0; i < DIM; i++) {
    EXPECT_NEAR(batch[i], single[i], 1e-4f) << "dim " << i;
  }
}

TEST_F(QuantizeTest, QuantizeRotated_WindowsMatchSinglePass) {
  // Batched builds quantize into windows of one store; the bytes must match a
  // single pass over all vectors.
  constexpr uint32_t N = 10;
  constexpr uint32_t DIM = 40;
  constexpr uint32_t SPLIT = 3;

  std::mt19937 rng(21);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  std::vector<float> h_vecs(N * DIM), h_centroid(DIM);
  for (auto& v : h_vecs) v = dist(rng);
  for (auto& c : h_centroid) c = 0.1f * dist(rng);
  DeviceBuf<float> d_vecs(h_vecs), d_centroid(h_centroid);

  const double t_const = rabitq_t_const(DIM, 4);
  auto whole = rabitq_data_store<float>::allocate_device(N, DIM, 4);
  auto split = rabitq_data_store<float>::allocate_device(N, DIM, 4);

  rabitq_quantize_rotated<4>(d_vecs.ptr, N, d_centroid.ptr, t_const, METRIC_L2, whole);

  auto head = split;
  head.n_vectors = SPLIT;
  auto rest = split;
  rest.data += static_cast<size_t>(SPLIT) * split.node_bytes();
  rest.n_vectors = N - SPLIT;
  rabitq_quantize_rotated<4>(d_vecs.ptr, SPLIT, d_centroid.ptr, t_const, METRIC_L2, head);
  rabitq_quantize_rotated<4>(d_vecs.ptr + SPLIT * DIM, N - SPLIT, d_centroid.ptr,
                             t_const, METRIC_L2, rest);

  auto h_whole = whole.to_host();
  auto h_split = split.to_host();
  for (size_t b = 0; b < h_whole.total_bytes(); b++) {
    ASSERT_EQ(h_whole.data[b], h_split.data[b]) << "byte " << b;
  }

  cudaFree(whole.data);
  cudaFree(split.data);
  cudaFreeHost(h_whole.data);
  cudaFreeHost(h_split.data);
}

// ═══════════════════════════════════════════════════════════════
// Distance estimator tests (require GPU)
// ═══════════════════════════════════════════════════════════════

// RaBitQ estimate from every query to every data vector, one block per query.
template <uint32_t BITS, uint32_t TILE_SIZE = 8>
__global__ void estimate_all_kernel(rabitq_data_store<float> codes,
                                    const float* rot_queries,
                                    const RabitqQueryFactor* factors,
                                    float* out) {
  extern __shared__ float smem_query[];
  const uint32_t qid = blockIdx.x;
  const uint32_t dim = codes.dim;
  for (uint32_t i = threadIdx.x; i < codes.query_padded_dim(); i += blockDim.x) {
    smem_query[i] = (i < dim) ? rot_queries[qid * dim + i] : 0.0f;
  }
  __syncthreads();

  auto tile = cg::tiled_partition<TILE_SIZE>(cg::this_thread_block());
  const RabitqQueryFactor fac = factors[qid];
  for (uint32_t i = tile.meta_group_rank(); i < codes.n_vectors;
       i += tile.meta_group_size()) {
    float d = rabitq_estimate_distance<BITS>(codes, i, smem_query, fac, tile);
    if (tile.thread_rank() == 0) out[qid * codes.n_vectors + i] = d;
  }
}

class EstimatorTest : public ::testing::Test {
protected:
  static constexpr uint32_t N = 512;
  static constexpr uint32_t Q = 8;
  static constexpr uint32_t DIM = 97;  // odd: exercises row and code padding
  const uint32_t padded_dim = vector_view<float>::pad(DIM);

  std::vector<float> h_vecs;     // [N * padded_dim], zero pad lanes
  std::vector<float> h_queries;  // [Q * padded_dim], zero pad lanes
  std::vector<float> h_centroid; // [DIM]
  std::vector<float> h_rot;      // [DIM * DIM] column-major

  void SetUp() override {
    CUDA_CHECK(cudaSetDevice(0));
    std::mt19937 rng(7);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    // Gaussian around a mean far from the origin, so the estimate is only
    // good if vectors are centered on the right (rotated) centroid.
    auto fill = [&](std::vector<float>& v, uint32_t n) {
      v.assign(static_cast<size_t>(n) * padded_dim, 0.0f);
      for (uint32_t i = 0; i < n; i++)
        for (uint32_t j = 0; j < DIM; j++)
          v[i * padded_dim + j] = 3.0f + dist(rng);
    };
    fill(h_vecs, N);
    fill(h_queries, Q);

    h_centroid.assign(DIM, 0.0f);
    for (uint32_t i = 0; i < N; i++)
      for (uint32_t j = 0; j < DIM; j++)
        h_centroid[j] += h_vecs[i * padded_dim + j] / N;

    h_rot.resize(DIM * DIM);
    std::vector<float> h_rot_t(DIM * DIM);
    set_rotation_matrix(DIM, h_rot.data(), h_rot_t.data(), 42);
  }
  void TearDown() override { CUDA_CHECK(cudaDeviceSynchronize()); }

  // [Q * N] RaBitQ estimates with BITS bits per dim.
  template <uint32_t BITS>
  std::vector<float> estimates(MetricType metric) {
    DeviceBuf<float> d_vecs(h_vecs), d_queries(h_queries);
    DeviceBuf<float> d_centroid(h_centroid), d_rot(h_rot), d_rot_centroid(DIM);
    vector_view<float> vecs(d_vecs.ptr, DIM, N, false);
    vector_view<float> queries(d_queries.ptr, DIM, Q, false);

    auto codes = rabitq_quantize<BITS>(vecs, d_rot.ptr, d_centroid.ptr, metric);

    cublasHandle_t handle;
    cublasCreate(&handle);
    rotate_single_data_vec(handle, d_centroid.ptr, d_rot_centroid.ptr, DIM, d_rot.ptr);
    cublasDestroy(handle);
    auto prepared = rabitq_prepare_queries(queries, d_rot.ptr, d_rot_centroid.ptr,
                                           BITS, metric);

    DeviceBuf<float> d_out(static_cast<size_t>(Q) * N);
    estimate_all_kernel<BITS><<<Q, 128, codes.query_padded_dim() * sizeof(float)>>>(
        codes, prepared.rot_queries, prepared.factors, d_out.ptr);
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    auto out = d_out.to_host();

    prepared.deallocate();
    cudaFree(codes.data);
    return out;
  }

  // Exact value the estimator targets, in the original space (the rotation
  // preserves both): ||o - q||^2 for L2, 1 - <o, q> for IP.
  double exact(MetricType metric, uint32_t q, uint32_t i) const {
    double acc = 0;
    for (uint32_t j = 0; j < DIM; j++) {
      double a = h_queries[q * padded_dim + j], b = h_vecs[i * padded_dim + j];
      acc += (metric == METRIC_L2) ? (a - b) * (a - b) : a * b;
    }
    return (metric == METRIC_L2) ? acc : 1.0 - acc;
  }

  // ||o - c|| * ||q - c||: the scale of the estimator's error term.
  double residual_scale(uint32_t q, uint32_t i) const {
    double qq = 0, oo = 0;
    for (uint32_t j = 0; j < DIM; j++) {
      double dq = h_queries[q * padded_dim + j] - h_centroid[j];
      double dov = h_vecs[i * padded_dim + j] - h_centroid[j];
      qq += dq * dq;
      oo += dov * dov;
    }
    return std::sqrt(qq * oo);
  }

  double mean_l2_rel_err(const std::vector<float>& est) const {
    double sum = 0;
    for (uint32_t q = 0; q < Q; q++)
      for (uint32_t i = 0; i < N; i++) {
        double e = exact(METRIC_L2, q, i);
        sum += std::abs(est[q * N + i] - e) / e;
      }
    return sum / (static_cast<double>(Q) * N);
  }
};

TEST_F(EstimatorTest, L2_ErrorShrinksWithBits) {
  const double e1 = mean_l2_rel_err(estimates<1>(METRIC_L2));
  const double e2 = mean_l2_rel_err(estimates<2>(METRIC_L2));
  const double e4 = mean_l2_rel_err(estimates<4>(METRIC_L2));
  const double e8 = mean_l2_rel_err(estimates<8>(METRIC_L2));
  std::cout << "mean relative L2 error: 1b=" << e1 << " 2b=" << e2
            << " 4b=" << e4 << " 8b=" << e8 << std::endl;

  EXPECT_LT(e1, 0.15);
  EXPECT_LT(e4, 0.04);
  EXPECT_LT(e8, 0.01);
  EXPECT_LT(e2, e1);
  EXPECT_LT(e4, e2);
  EXPECT_LT(e8, e4);
}

TEST_F(EstimatorTest, InnerProduct_TracksExact) {
  auto est = estimates<4>(METRIC_IP);
  double sum = 0;
  for (uint32_t q = 0; q < Q; q++)
    for (uint32_t i = 0; i < N; i++)
      sum += std::abs(est[q * N + i] - exact(METRIC_IP, q, i)) / residual_scale(q, i);
  const double err = sum / (static_cast<double>(Q) * N);
  std::cout << "mean normalized IP error (4b): " << err << std::endl;
  EXPECT_LT(err, 0.04);
}

TEST_F(EstimatorTest, PreparedQueries_FactorsMatchHost) {
  // Identity rotation: rotated query == query, so the factors are checkable
  // directly against the definitions.
  std::vector<float> eye(DIM * DIM, 0.0f);
  for (uint32_t i = 0; i < DIM; i++) eye[i + i * DIM] = 1.0f;
  DeviceBuf<float> d_queries(h_queries), d_eye(eye), d_centroid(h_centroid);
  vector_view<float> queries(d_queries.ptr, DIM, Q, false);

  for (MetricType metric : {METRIC_L2, METRIC_IP}) {
    auto prepared = rabitq_prepare_queries(queries, d_eye.ptr, d_centroid.ptr, 4, metric);
    std::vector<RabitqQueryFactor> fac(Q);
    std::vector<float> rot(static_cast<size_t>(Q) * DIM);
    cudaMemcpy(fac.data(), prepared.factors, sizeof(RabitqQueryFactor) * Q,
               cudaMemcpyDeviceToHost);
    cudaMemcpy(rot.data(), prepared.rot_queries, sizeof(float) * rot.size(),
               cudaMemcpyDeviceToHost);
    prepared.deallocate();

    for (uint32_t q = 0; q < Q; q++) {
      double sqr = 0, ip = 0, sum = 0;
      for (uint32_t j = 0; j < DIM; j++) {
        double x = h_queries[q * padded_dim + j];
        EXPECT_NEAR(rot[q * DIM + j], x, 1e-4) << "query " << q << " dim " << j;
        sqr += (x - h_centroid[j]) * (x - h_centroid[j]);
        ip  += x * h_centroid[j];
        sum += x;
      }
      const double add = (metric == METRIC_L2) ? sqr : -ip;
      EXPECT_NEAR(fac[q].add, add, 1e-3 * std::max(1.0, std::abs(add)));
      EXPECT_NEAR(fac[q].k1xSumq, -0.5 * sum, 1e-3 * std::max(1.0, std::abs(sum)));
      EXPECT_NEAR(fac[q].kBxSumq, -7.5 * sum, 1e-2 * std::max(1.0, std::abs(sum)));
    }
  }
}

} // namespace jasper::test
