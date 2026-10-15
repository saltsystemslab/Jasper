#include <gtest/gtest.h>
#include "helpers.cuh"
#include "jasper/beam_search/api.cuh"
#include "jasper/index/graph.cuh"
#include "jasper/rabitq/edge_rq.cuh"
#include "jasper/rabitq/layout.cuh"
#include "jasper/rabitq/quantize.cuh"
#include "jasper/rabitq/rq_kernels.cuh"
#include "jasper/rotation/rotation.cuh"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <set>
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

// ═══════════════════════════════════════════════════════════════
// Directional (per-edge) RaBitQ
//   storage layout  — rabitq/edge_rq.cuh
//   encode          — rabitq/rq_kernels.cuh  (graph::populate_edge_rq)
//   estimator math  — beam_search/rq_beam_search.cuh
// ═══════════════════════════════════════════════════════════════

using rq_row_t = edge_rq_list<uint32_t, /*N_NEIGHBORS=*/32, /*RQ_DIM=*/32, /*RQ_BITS=*/1>;

class EdgeRqLayoutTest : public ::testing::Test {};

TEST_F(EdgeRqLayoutTest, Constants_1bit_d32) {
  EXPECT_EQ(rq_row_t::n_words, 1u);
  EXPECT_EQ(rq_row_t::dims_per_word, 32u);
  EXPECT_EQ(rq_row_t::level_mask, 1u);
  // Per edge: one 32-bit code word + fp16 rescale + bf16 mag² = 8 bytes.
  EXPECT_EQ(sizeof(rq_row_t), 32u * 8u);
}

TEST_F(EdgeRqLayoutTest, WordsAreIndexedPerEdge) {
  rq_row_t row{};
  for (uint32_t e = 0; e < 32; e++)
    row.set_word(static_cast<uint8_t>(e), 0, e * 2654435761u);
  for (uint32_t e = 0; e < 32; e++)
    EXPECT_EQ(row.get_word(static_cast<uint8_t>(e), 0), e * 2654435761u)
        << "edge " << e;
}

TEST_F(EdgeRqLayoutTest, GetLevelIsOneBitPerDimension) {
  rq_row_t row{};
  row.set_word(3, 0, 0xA5A5A5A5u);
  for (uint32_t d = 0; d < 32; d++)
    EXPECT_EQ(row.get_level(3, d), (0xA5A5A5A5u >> d) & 1u) << "dim " << d;
}

TEST_F(EdgeRqLayoutTest, TwoBitLevelsPackFourPerByte) {
  using row2_t = edge_rq_list<uint32_t, 4, 32, 2>;
  EXPECT_EQ(row2_t::n_words, 2u);
  EXPECT_EQ(row2_t::dims_per_word, 16u);
  EXPECT_EQ(row2_t::level_mask, 3u);

  row2_t row{};
  row.set_word(1, 0, 0xE4u);  // 0b11'10'01'00 → dims 0..3 = 0,1,2,3
  EXPECT_EQ(row.get_level(1, 0), 0u);
  EXPECT_EQ(row.get_level(1, 1), 1u);
  EXPECT_EQ(row.get_level(1, 2), 2u);
  EXPECT_EQ(row.get_level(1, 3), 3u);
}

// The plane stride IS the neighbor count, which changes between the DiskANN
// builder's R_int and R_fin rows. At rq_dim=32 there is only one plane and the
// stride never shows; at 128 there are four, and a byte-wise row copy between
// two degrees would misplace every plane past the first. copy_rq_edge goes
// through get_word/set_word for exactly this reason.
TEST_F(EdgeRqLayoutTest, WordPlanesAreStridedByNeighborCount) {
  using narrow_t = edge_rq_list<uint32_t, 32, 128, 1>;
  using wide_t   = edge_rq_list<uint32_t, 64, 128, 1>;
  static_assert(narrow_t::n_words == 4 && wide_t::n_words == 4,
                "expected 4 planes at rq_dim=128, rq_bits=1");

  narrow_t a{};
  wide_t   b{};
  for (uint32_t w = 0; w < narrow_t::n_words; w++) a.set_word(7, w, 0x1000u + w);
  for (uint32_t w = 0; w < narrow_t::n_words; w++) b.set_word(3, w, a.get_word(7, w));
  for (uint32_t w = 0; w < wide_t::n_words; w++)
    EXPECT_EQ(b.get_word(3, w), 0x1000u + w) << "plane " << w;

  // Plane 1 of edge 0 really does sit at a different offset in the two layouts.
  EXPECT_EQ(&a.code[1 * 32] - &a.code[0], 32);
  EXPECT_EQ(&b.code[1 * 64] - &b.code[0], 64);
}

TEST_F(EdgeRqLayoutTest, CbMatchesBitWidth) {
  EXPECT_FLOAT_EQ((rq_globals<32, 1>::cb()), -0.5f);
  EXPECT_FLOAT_EQ((rq_globals<32, 2>::cb()), -1.5f);
  EXPECT_FLOAT_EQ((rq_globals<32, 4>::cb()), -7.5f);
}

// ── The JL projection P_d ──────────────────────────────────────

class ProjectionMatrixTest : public ::testing::Test {};

TEST_F(ProjectionMatrixTest, RowsAreOrthonormal) {
  constexpr int D = 96, d = 32;
  std::vector<float> P(static_cast<size_t>(D) * d);
  set_projection_matrix<float>(D, d, P.data(), /*seed=*/7);

  // Row j occupies P[j*D .. j*D+D): P·Pᵀ must be the identity.
  for (int a = 0; a < d; a++) {
    for (int b = 0; b < d; b++) {
      double dot = 0.0;
      for (int i = 0; i < D; i++)
        dot += static_cast<double>(P[a * D + i]) * P[b * D + i];
      EXPECT_NEAR(dot, a == b ? 1.0 : 0.0, 1e-4)
          << "row " << a << " · row " << b;
    }
  }
}

TEST_F(ProjectionMatrixTest, DeterministicInSeed) {
  constexpr int D = 64, d = 16;
  std::vector<float> a(static_cast<size_t>(D) * d), b(a.size()), c(a.size());
  set_projection_matrix<float>(D, d, a.data(), 11);
  set_projection_matrix<float>(D, d, b.data(), 11);
  set_projection_matrix<float>(D, d, c.data(), 12);
  EXPECT_EQ(a, b) << "same seed must give the same P_d on both sides";
  EXPECT_NE(a, c);
}

TEST_F(ProjectionMatrixTest, PreservesNormsInExpectation) {
  // E||P_d x||² = (d/D)||x||²; the estimator relies on this only through the
  // unit vectors, but a gross scale error here would break everything.
  constexpr int D = 128, d = 32;
  std::vector<float> P(static_cast<size_t>(D) * d);
  set_projection_matrix<float>(D, d, P.data(), 3);

  std::mt19937 rng(5);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  double ratio_sum = 0.0;
  constexpr int kTrials = 200;
  for (int t = 0; t < kTrials; t++) {
    std::vector<float> x(D);
    double xn = 0.0;
    for (int i = 0; i < D; i++) { x[i] = nd(rng); xn += double(x[i]) * x[i]; }
    double pn = 0.0;
    for (int j = 0; j < d; j++) {
      double acc = 0.0;
      for (int i = 0; i < D; i++) acc += double(P[j * D + i]) * x[i];
      pn += acc * acc;
    }
    ratio_sum += pn / xn;
  }
  EXPECT_NEAR(ratio_sum / kTrials, double(d) / D, 0.02);
}

// ── GPU fixtures ───────────────────────────────────────────────
// dim 64 with k_ranks 4 puts rq_dim at 32, so P_d genuinely reduces the
// dimension (rq_dim == dim would make it a plain rotation).
namespace {

constexpr uint32_t RQ_R        = 32;   // out-degree
constexpr uint32_t RQ_DIM_DATA = 64;   // data dimension

using rq_cfg = jasper::graph_config<uint32_t, RQ_R, __half, float,
                                    jasper::distance_func::L2, true, 4, uint8_t>;
static_assert(rq_cfg::rq_dim == 32, "expected rq_dim = 8 * k_ranks");
static_assert(rq_cfg::rq_bits == 1, "expected 1-bit RaBitQ");

using rq_edge_t = rq_cfg::edge_rq_list_t;

struct RqTestGraph {
  jasper::graph<rq_cfg> g{};
  std::vector<float>    vecs;    // [n * padded_dim] — exactly what the halves hold
  std::vector<uint32_t> nbrs;    // [n * RQ_R]
  uint32_t n = 0, dim = 0, padded_dim = 0;

  const float* vec(uint32_t i) const {
    return vecs.data() + static_cast<size_t>(i) * padded_dim;
  }
  float dist_sq(const float* a, const float* b) const {
    float s = 0.0f;
    for (uint32_t d = 0; d < padded_dim; d++) { const float df = a[d] - b[d]; s += df * df; }
    return s;
  }
};

// Random Gaussian vectors plus each node's RQ_R nearest neighbors by brute
// force. No explicit rotation: an i.i.d. Gaussian vector is already isotropic,
// so its first rq_dim coordinates are as good a random projection as a rotated
// vector's would be — the prerotate exists for real data with structured axes.
RqTestGraph build_rq_test_graph(uint32_t n, uint32_t dim, uint32_t seed) {
  RqTestGraph t;
  t.n = n;
  t.dim = dim;
  t.padded_dim = jasper::vector_view<__half>::pad(dim);

  auto h_view = jasper::vector_view<__half>::allocate(dim, n, /*on_host=*/true);
  std::memset(h_view.data, 0, h_view.size_bytes());  // pad lanes must read zero

  std::mt19937 rng(seed);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  t.vecs.assign(static_cast<size_t>(n) * t.padded_dim, 0.0f);
  for (uint32_t i = 0; i < n; i++) {
    __half* row = h_view[i];
    for (uint32_t d = 0; d < dim; d++) {
      const __half h = __float2half(nd(rng));
      row[d] = h;
      t.vecs[static_cast<size_t>(i) * t.padded_dim + d] = __half2float(h);
    }
  }

  t.nbrs.assign(static_cast<size_t>(n) * RQ_R, 0);
  std::vector<std::pair<float, uint32_t>> cand(n);
  for (uint32_t i = 0; i < n; i++) {
    for (uint32_t j = 0; j < n; j++)
      cand[j] = {i == j ? std::numeric_limits<float>::max()
                        : t.dist_sq(t.vec(i), t.vec(j)), j};
    std::partial_sort(cand.begin(), cand.begin() + RQ_R, cand.end());
    for (uint32_t e = 0; e < RQ_R; e++)
      t.nbrs[static_cast<size_t>(i) * RQ_R + e] = cand[e].second;
  }

  t.g = jasper::graph<rq_cfg>::allocate_and_load(h_view, /*on_host=*/false);
  t.g.medoid = 0;
  h_view.deallocate();

  // Install the edge lists directly (n fits in one segment).
  using segment_t   = jasper::graph_segment<rq_cfg>;
  using edge_list_t = rq_cfg::edge_list_t;
  std::vector<segment_t> h_segs(t.g.segments.begin(), t.g.segments.end());
  std::vector<edge_list_t> h_edges(n);
  std::vector<uint8_t>     h_counts(n, static_cast<uint8_t>(RQ_R));
  for (uint32_t i = 0; i < n; i++) {
    for (uint32_t e = 0; e < RQ_R; e++) {
      const uint32_t v = t.nbrs[static_cast<size_t>(i) * RQ_R + e];
      h_edges[i].edges[e] = v;
      h_edges[i].dist[e]  = __float2bfloat16(t.dist_sq(t.vec(i), t.vec(v)));
    }
  }
  cudaMemcpy(h_segs[0].edges, h_edges.data(),
             sizeof(edge_list_t) * n, cudaMemcpyHostToDevice);
  cudaMemcpy(h_segs[0].edge_counts, h_counts.data(), n, cudaMemcpyHostToDevice);
  return t;
}

std::vector<rq_edge_t> read_edge_rqs(jasper::graph<rq_cfg>& g, uint32_t n) {
  using segment_t = jasper::graph_segment<rq_cfg>;
  std::vector<segment_t> h_segs(g.segments.begin(), g.segments.end());
  std::vector<rq_edge_t> out(n);
  cudaMemcpy(out.data(), h_segs[0].edge_rqs,
             sizeof(rq_edge_t) * n, cudaMemcpyDeviceToHost);
  return out;
}

}  // namespace

// populate_edge_rq must store, for every edge: the sign pattern of the
// projected residual (v-u)[:rq_dim] (1-bit RaBitQ), the EXACT full-dimension
// ||v-u||², and rescale = 1/<x̄,û> = 2·||r_d|| / ||r_d||₁.
TEST(EdgeRqEncodeTest, CodeMagnitudeAndRescaleMatchHostReference) {
  auto t = build_rq_test_graph(64, RQ_DIM_DATA, 7);
  auto globals = t.g.generate_rq_globals();
  t.g.populate_edge_rq(globals);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaGetLastError());
  auto rows = read_edge_rqs(t.g, t.n);

  constexpr uint32_t D = rq_cfg::rq_dim;
  for (uint32_t u = 0; u < t.n; u++) {
    for (uint32_t e = 0; e < RQ_R; e++) {
      const uint32_t v = t.nbrs[static_cast<size_t>(u) * RQ_R + e];

      // Magnitude is a full-dimension quantity...
      double mag_sq = 0.0;
      for (uint32_t d = 0; d < t.padded_dim; d++) {
        const double r = static_cast<double>(t.vec(v)[d]) - t.vec(u)[d];
        mag_sq += r * r;
      }
      // ...while the code sees only the first rq_dim coordinates. t.vecs holds
      // exactly the half values the kernel reads, and the residual is one
      // float subtraction on both sides, so the sign comparison is exact
      // rather than a near-zero coin flip.
      double proj_sq = 0.0, proj_l1 = 0.0;
      for (uint32_t d = 0; d < D; d++) {
        const float rf = t.vec(v)[d] - t.vec(u)[d];
        proj_sq += static_cast<double>(rf) * rf;
        proj_l1 += std::fabs(static_cast<double>(rf));
        ASSERT_EQ(rows[u].get_level(static_cast<uint8_t>(e), d),
                  rf >= 0.0f ? 1u : 0u)
            << "u=" << u << " edge=" << e << " dim=" << d;
      }

      const float want_mag = static_cast<float>(mag_sq);
      EXPECT_NEAR(rows[u].get_mag_sq(static_cast<uint8_t>(e)), want_mag,
                  0.01f * want_mag + 1e-3f)
          << "u=" << u << " edge=" << e;   // bf16 storage → ~0.4% relative

      const float want_rescale =
          static_cast<float>(2.0 * std::sqrt(proj_sq) / proj_l1);
      EXPECT_NEAR(rows[u].get_rescale(static_cast<uint8_t>(e)), want_rescale,
                  0.01f * want_rescale)
          << "u=" << u << " edge=" << e;
    }
  }
  t.g.deallocate();
}

// The estimate itself: A + mag² − (2√A/||r_d||)·mag·rescale·<x̄, r_d>, with
// r_d = P_d·q − P_d·u.
// Sanity bounds, not tight numerics — what they pin down is that the stored
// factors combine into something near-unbiased and clearly better than the
// magnitude-only fallback (A + mag², i.e. assuming the residual is orthogonal
// to the query direction).
TEST(EdgeRqEstimatorTest, NearUnbiasedAndBeatsMagnitudeOnlyBaseline) {
  auto t = build_rq_test_graph(64, RQ_DIM_DATA, 11);
  auto globals = t.g.generate_rq_globals();
  t.g.populate_edge_rq(globals);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaGetLastError());
  auto rows = read_edge_rqs(t.g, t.n);

  constexpr uint32_t D  = rq_cfg::rq_dim;
  constexpr float    CB = rq_globals<rq_cfg::rq_dim, rq_cfg::rq_bits>::cb();

  std::mt19937 rng(99);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<float> q(t.padded_dim, 0.0f);

  double sum_rel = 0.0, sum_abs = 0.0, sum_abs_base = 0.0;
  uint32_t cnt = 0;

  for (uint32_t trial = 0; trial < 32; trial++) {
    for (uint32_t d = 0; d < t.dim; d++) q[d] = __half2float(__float2half(nd(rng)));

    for (uint32_t u = 0; u < t.n; u += 4) {
      // A over the full dimension; the residual prefix is the projected side.
      double A = 0.0, proj_sq = 0.0;
      std::vector<double> rd(D);
      for (uint32_t d = 0; d < t.padded_dim; d++) {
        const double r = static_cast<double>(q[d]) - t.vec(u)[d];
        A += r * r;
        if (d < D) { rd[d] = r; proj_sq += r * r; }
      }
      const double node_scale = 2.0 * std::sqrt(A) / std::sqrt(proj_sq);

      for (uint32_t e = 0; e < RQ_R; e++) {
        const uint32_t v = t.nbrs[static_cast<size_t>(u) * RQ_R + e];
        const uint8_t  e8 = static_cast<uint8_t>(e);

        double dot = 0.0;
        for (uint32_t d = 0; d < D; d++)
          dot += (static_cast<double>(rows[u].get_level(e8, d)) + CB) * rd[d];

        const double mag_sq = rows[u].get_mag_sq(e8);
        const double est    = A + mag_sq
            - node_scale * std::sqrt(mag_sq) * rows[u].get_rescale(e8) * dot;
        const double truth  = t.dist_sq(q.data(), t.vec(v));

        sum_rel      += (est - truth) / truth;
        sum_abs      += std::fabs(est - truth) / truth;
        sum_abs_base += std::fabs((A + mag_sq) - truth) / truth;
        cnt++;
      }
    }
  }
  ASSERT_GT(cnt, 0u);
  const double bias     = sum_rel / cnt;
  const double mae      = sum_abs / cnt;
  const double mae_base = sum_abs_base / cnt;

  EXPECT_LT(std::fabs(bias), 0.05)
      << "mean relative error " << bias << " — estimator should be near-unbiased";
  EXPECT_LT(mae, 0.6 * mae_base)
      << "mae " << mae << " vs magnitude-only baseline " << mae_base;
}

// End-to-end through the real kernel: this is what type-checks rq_estimator
// inside graph_beam_search_kernel. The frontier records EXACT distances for
// visited nodes, so with a navigable graph and a wide enough beam the estimator
// only has to rank sensibly for recall to come out near 1.
TEST(RqBeamSearchTest, RecallAgainstBruteForce) {
  constexpr uint32_t NQ = 32;
  constexpr uint32_t K  = 10;

  auto t = build_rq_test_graph(256, RQ_DIM_DATA, 13);
  auto globals = t.g.generate_rq_globals();
  t.g.populate_edge_rq(globals);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaGetLastError());

  auto h_q = jasper::vector_view<__half>::allocate(t.dim, NQ, /*on_host=*/true);
  std::memset(h_q.data, 0, h_q.size_bytes());
  std::vector<float> qf(static_cast<size_t>(NQ) * t.padded_dim, 0.0f);
  std::mt19937 rng(21);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  for (uint32_t i = 0; i < NQ; i++) {
    for (uint32_t d = 0; d < t.dim; d++) {
      const __half h = __float2half(nd(rng));
      h_q[i][d] = h;
      qf[static_cast<size_t>(i) * t.padded_dim + d] = __half2float(h);
    }
  }
  auto d_q = h_q.to_device();
  h_q.deallocate();

  jasper::search_params params{
    .k          = K,
    .beam_width = 64,
    .limit      = 256,
  };
  auto result = jasper::rq_search(t.g, globals, d_q, params);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<thrust::pair<uint32_t, float>> h_res(static_cast<size_t>(NQ) * K);
  CUDA_CHECK(cudaMemcpy(h_res.data(), result.frontier,
                        sizeof(h_res[0]) * h_res.size(), cudaMemcpyDeviceToHost));

  uint32_t hits = 0;
  std::vector<std::pair<float, uint32_t>> cand(t.n);
  for (uint32_t i = 0; i < NQ; i++) {
    const float* q = qf.data() + static_cast<size_t>(i) * t.padded_dim;
    for (uint32_t j = 0; j < t.n; j++) cand[j] = {t.dist_sq(q, t.vec(j)), j};
    std::partial_sort(cand.begin(), cand.begin() + K, cand.end());

    std::set<uint32_t> truth;
    for (uint32_t e = 0; e < K; e++) truth.insert(cand[e].second);
    for (uint32_t e = 0; e < K; e++)
      if (truth.count(h_res[static_cast<size_t>(i) * K + e].first)) hits++;
  }
  const double recall = static_cast<double>(hits) / (NQ * K);
  EXPECT_GE(recall, 0.9) << "recall@" << K << " = " << recall;

  cudaFree(result.frontier);
  cudaFree(d_q.data);
  t.g.deallocate();
}

// ── rq_dim wider than the data ─────────────────────────────────
// 8*k_ranks can outrun the dimension (k_ranks=16 on DEEP's 96 dims, and here
// rq_dim=32 on 16). There is then nothing to project onto, so the scheme has
// to degenerate to plain 1-bit RaBitQ over every real coordinate with the
// surplus held at zero — NOT read past the end of the row. The same config
// serves both tests, so this costs no extra instantiation.
constexpr uint32_t RQ_DIM_SMALL = 16;   // < rq_dim (32): the clamped case
static_assert(RQ_DIM_SMALL < rq_cfg::rq_dim, "fixture must exercise the clamp");

// The encoder's half: the coded prefix is the whole vector, the tail is pinned
// to the sign of an exact zero, and rescale/mag come out of the real
// coordinates alone — which is the same code a native rq_dim == dim build
// would write, since a zero coordinate adds 0 to both ||r|| and ||r||₁.
TEST(EdgeRqEncodeTest, ClampsAndZeroPadsWhenRqDimExceedsData) {
  auto t = build_rq_test_graph(64, RQ_DIM_SMALL, 7);
  ASSERT_LT(t.padded_dim, rq_cfg::rq_dim) << "fixture did not clamp";

  auto globals = t.g.generate_rq_globals();
  t.g.populate_edge_rq(globals);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaGetLastError());
  auto rows = read_edge_rqs(t.g, t.n);

  constexpr uint32_t D = rq_cfg::rq_dim;
  for (uint32_t u = 0; u < t.n; u++) {
    for (uint32_t e = 0; e < RQ_R; e++) {
      const uint8_t  e8 = static_cast<uint8_t>(e);
      const uint32_t v  = t.nbrs[static_cast<size_t>(u) * RQ_R + e];

      double mag_sq = 0.0, proj_l1 = 0.0;
      for (uint32_t d = 0; d < t.padded_dim; d++) {
        const float rf = t.vec(v)[d] - t.vec(u)[d];
        mag_sq  += static_cast<double>(rf) * rf;
        proj_l1 += std::fabs(static_cast<double>(rf));
        ASSERT_EQ(rows[u].get_level(e8, d), rf >= 0.0f ? 1u : 0u)
            << "u=" << u << " edge=" << e << " dim=" << d;
      }
      // The tail never touched the vector: sign(0) = +, so every surplus bit
      // is a 1. A garbage value here means rq_populate read off the row.
      for (uint32_t d = t.padded_dim; d < D; d++)
        ASSERT_EQ(rows[u].get_level(e8, d), 1u)
            << "u=" << u << " edge=" << e << " pad dim=" << d;

      // mag² is the full (= projected, here) residual, and rescale uses only
      // the real coordinates: 2·||r|| / ||r||₁.
      const float want_mag = static_cast<float>(mag_sq);
      EXPECT_NEAR(rows[u].get_mag_sq(e8), want_mag, 0.01f * want_mag + 1e-3f)
          << "u=" << u << " edge=" << e;
      const float want_rescale =
          static_cast<float>(2.0 * std::sqrt(mag_sq) / proj_l1);
      EXPECT_NEAR(rows[u].get_rescale(e8), want_rescale, 0.01f * want_rescale)
          << "u=" << u << " edge=" << e;
    }
  }
  t.g.deallocate();
}

// The estimator's half, end to end. rq_estimator has to zero the [padded_dim,
// rq_dim) tail of its shared residual ONCE in the constructor and keep it
// zero, because the per-edge dot runs to the compile-time rq_dim. If that tail
// held junk, every estimate would take a random contribution from it and
// recall would collapse — so this is the test that fails if the clamp is only
// half-implemented.
TEST(RqBeamSearchTest, RecallWhenRqDimExceedsData) {
  constexpr uint32_t NQ = 32;
  constexpr uint32_t K  = 10;

  auto t = build_rq_test_graph(256, RQ_DIM_SMALL, 13);
  ASSERT_LT(t.padded_dim, rq_cfg::rq_dim) << "fixture did not clamp";

  auto globals = t.g.generate_rq_globals();
  t.g.populate_edge_rq(globals);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaGetLastError());

  auto h_q = jasper::vector_view<__half>::allocate(t.dim, NQ, /*on_host=*/true);
  std::memset(h_q.data, 0, h_q.size_bytes());
  std::vector<float> qf(static_cast<size_t>(NQ) * t.padded_dim, 0.0f);
  std::mt19937 rng(21);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  for (uint32_t i = 0; i < NQ; i++) {
    for (uint32_t d = 0; d < t.dim; d++) {
      const __half h = __float2half(nd(rng));
      h_q[i][d] = h;
      qf[static_cast<size_t>(i) * t.padded_dim + d] = __half2float(h);
    }
  }
  auto d_q = h_q.to_device();
  h_q.deallocate();

  jasper::search_params params{
    .k          = K,
    .beam_width = 64,
    .limit      = 256,
  };
  auto result = jasper::rq_search(t.g, globals, d_q, params);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<thrust::pair<uint32_t, float>> h_res(static_cast<size_t>(NQ) * K);
  CUDA_CHECK(cudaMemcpy(h_res.data(), result.frontier,
                        sizeof(h_res[0]) * h_res.size(), cudaMemcpyDeviceToHost));

  uint32_t hits = 0;
  std::vector<std::pair<float, uint32_t>> cand(t.n);
  for (uint32_t i = 0; i < NQ; i++) {
    const float* q = qf.data() + static_cast<size_t>(i) * t.padded_dim;
    for (uint32_t j = 0; j < t.n; j++) cand[j] = {t.dist_sq(q, t.vec(j)), j};
    std::partial_sort(cand.begin(), cand.begin() + K, cand.end());

    std::set<uint32_t> truth;
    for (uint32_t e = 0; e < K; e++) truth.insert(cand[e].second);
    for (uint32_t e = 0; e < K; e++)
      if (truth.count(h_res[static_cast<size_t>(i) * K + e].first)) hits++;
  }
  // Looser than the projecting case: 16 coded dimensions is a weaker ranker
  // than 32. The bar is "ranks sensibly", not a numerical target.
  const double recall = static_cast<double>(hits) / (NQ * K);
  EXPECT_GE(recall, 0.85) << "recall@" << K << " = " << recall;

  cudaFree(result.frontier);
  cudaFree(d_q.data);
  t.g.deallocate();
}

} // namespace jasper::test