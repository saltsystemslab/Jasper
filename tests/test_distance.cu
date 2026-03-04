#include <gtest/gtest.h>
#include "helpers.cuh"
#include "jasper/distance/distance.cuh"

namespace jasper::test {

// ── Thin kernel to call device distance functions from host tests ──

template <jasper::distance_func FUNC, typename DATA_T, typename DIST_T, uint32_t TILE_SIZE>
__global__ void distance_kernel(const DATA_T* a, const DATA_T* b,
                                 uint32_t dim, float* out) {
  namespace cg = cooperative_groups;
  auto block = cg::this_thread_block();
  auto tile = cg::tiled_partition<TILE_SIZE>(block);
  float d = jasper::compute_distance<FUNC, DATA_T, DIST_T, TILE_SIZE>(
      a, b, dim, tile);
  if (tile.thread_rank() == 0) {
    *out = d;
  }
}

// ── Test fixture ───────────────────────────────────────────────

class DistanceTest : public ::testing::Test {
protected:
  void SetUp() override { CUDA_CHECK(cudaSetDevice(0)); }
  void TearDown() override { CUDA_CHECK(cudaDeviceSynchronize()); }
};

// ── L2 tests ───────────────────────────────────────────────────

TEST_F(DistanceTest, L2_IdenticalVectors_Zero) {
  std::vector<float> v = {1.0f, 2.0f, 3.0f, 4.0f};
  DeviceBuf<float> da(v), db(v), dout(1);

  distance_kernel<distance_func::L2, float, float, 4>
      <<<1, 32>>>(da.ptr, db.ptr, 4, dout.ptr);
    CUDA_CHECK(cudaGetLastError()); 
  CUDA_CHECK(cudaDeviceSynchronize());

  auto h = dout.to_host();
  EXPECT_FLOAT_EQ(h[0], 0.0f);
}

TEST_F(DistanceTest, L2_KnownDistance) {
  // (0,0,0) vs (3,4,0) → squared L2 = 9+16 = 25
  std::vector<float> a = {0, 0, 0, 0};
  std::vector<float> b = {3, 4, 0, 0};
  DeviceBuf<float> da(a), db(b), dout(1);

  distance_kernel<distance_func::L2, float, float, 4>
      <<<1, 32>>>(da.ptr, db.ptr, 4, dout.ptr);
  CUDA_CHECK(cudaGetLastError()); 
  CUDA_CHECK(cudaDeviceSynchronize());

  auto h = dout.to_host();
  EXPECT_NEAR(h[0], 25.0f, 1e-5f);
}

TEST_F(DistanceTest, L2_LargerDim) {
  // 16-dim, all ones vs all zeros → squared L2 = 16
  constexpr uint32_t dim = 16;
  std::vector<float> a(dim, 1.0f);
  std::vector<float> b(dim, 0.0f);
  DeviceBuf<float> da(a), db(b), dout(1);

  distance_kernel<distance_func::L2, float, float, 4>
      <<<1, 32>>>(da.ptr, db.ptr, dim, dout.ptr);
  CUDA_CHECK(cudaGetLastError()); 
  CUDA_CHECK(cudaDeviceSynchronize());

  auto h = dout.to_host();
  EXPECT_NEAR(h[0], 16.0f, 1e-4f);
}

TEST_F(DistanceTest, L2_Symmetric) {
  std::vector<float> a = {1, 2, 3, 4};
  std::vector<float> b = {5, 6, 7, 8};
  DeviceBuf<float> da(a), db(b), dout1(1), dout2(1);

  distance_kernel<distance_func::L2, float, float, 4>
      <<<1, 32>>>(da.ptr, db.ptr, 4, dout1.ptr);
  distance_kernel<distance_func::L2, float, float, 4>
      <<<1, 32>>>(db.ptr, da.ptr, 4, dout2.ptr);
  CUDA_CHECK(cudaGetLastError()); 
  CUDA_CHECK(cudaDeviceSynchronize());

  auto h1 = dout1.to_host();
  auto h2 = dout2.to_host();
  EXPECT_FLOAT_EQ(h1[0], h2[0]);
}

// ── Inner product tests ────────────────────────────────────────

TEST_F(DistanceTest, IP_Orthogonal_Zero) {
  // (1,0,0,0) · (0,1,0,0) = 0 → negated = 0
  std::vector<float> a = {1, 0, 0, 0};
  std::vector<float> b = {0, 1, 0, 0};
  DeviceBuf<float> da(a), db(b), dout(1);

  distance_kernel<distance_func::INNER_PRODUCT, float, float, 4>
      <<<1, 32>>>(da.ptr, db.ptr, 4, dout.ptr);
  CUDA_CHECK(cudaGetLastError()); 
  CUDA_CHECK(cudaDeviceSynchronize());

  auto h = dout.to_host();
  EXPECT_NEAR(h[0], 0.0f, 1e-6f);
}

TEST_F(DistanceTest, IP_KnownValue) {
  // (1,2,3,4) · (2,3,4,5) = 2+6+12+20 = 40 → negated = -40
  std::vector<float> a = {1, 2, 3, 4};
  std::vector<float> b = {2, 3, 4, 5};
  DeviceBuf<float> da(a), db(b), dout(1);

  distance_kernel<distance_func::INNER_PRODUCT, float, float, 4>
      <<<1, 32>>>(da.ptr, db.ptr, 4, dout.ptr);
  CUDA_CHECK(cudaGetLastError()); 
  CUDA_CHECK(cudaDeviceSynchronize());

  auto h = dout.to_host();
  EXPECT_NEAR(h[0], -40.0f, 1e-4f);
}

TEST_F(DistanceTest, IP_SelfDot_Negative) {
  // (3,4) · (3,4) = 25 → negated = -25
  std::vector<float> a = {3, 4, 0, 0};
  DeviceBuf<float> da(a), dout(1);

  distance_kernel<distance_func::INNER_PRODUCT, float, float, 4>
      <<<1, 32>>>(da.ptr, da.ptr, 2, dout.ptr);
  CUDA_CHECK(cudaGetLastError()); 
  CUDA_CHECK(cudaDeviceSynchronize());

  auto h = dout.to_host();
  EXPECT_NEAR(h[0], -25.0f, 1e-4f);
}

} // namespace jasper::test