#pragma once

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <cublas_v2.h>

#include "jasper/index/graph.cuh"
#include "jasper/rabitq/distance.cuh"
#include "jasper/rabitq/layout.cuh"
#include "jasper/rabitq/quantize.cuh"
#include "jasper/rotation/rotation.cuh"

namespace jasper {

// RaBitQ codes for every vector slot of a graph, plus the random rotation and
// rotated centroid that queries must be prepared with. Code i belongs to graph
// slot i (global index global_offset + i).
//
// Built from a graph by build_rabitq_index() and passed alongside it to
// rabitq_search(), the same way lsh_globals rides along with
// directional_search(). The codes snapshot the graph's slots: rebuild after
// append_batch() or compact(), which add or renumber slots. mark_deleted() and
// consolidate() leave slots in place, so the codes stay valid across them.
struct rabitq_index {
  rabitq_data_store<float> codes;            // device
  float*     d_rotation     = nullptr;       // device, [dim * dim] column-major P
  float*     d_rot_centroid = nullptr;       // device, [dim], P^T * centroid
  MetricType metric         = METRIC_L2;

  uint32_t dim()          const { return codes.dim; }
  uint32_t n_vectors()    const { return codes.n_vectors; }
  uint32_t bits_per_dim() const { return codes.bits_per_dim; }
  bool     empty()        const { return codes.data == nullptr; }

  // Rotate device queries into the codes' space and compute their estimator
  // factors. Caller frees the result with rabitq_queries::deallocate().
  template <typename DATA_T>
  rabitq_queries prepare_queries(vector_view<DATA_T> d_queries,
                                 cudaStream_t stream = 0) const {
    return rabitq_prepare_queries(d_queries, d_rotation, d_rot_centroid,
                                  bits_per_dim(), metric, stream);
  }

  void deallocate() {
    cudaFree(codes.data);
    cudaFree(d_rotation);
    cudaFree(d_rot_centroid);
    codes          = rabitq_data_store<float>{};
    d_rotation     = nullptr;
    d_rot_centroid = nullptr;
  }
};

// Per-dim sums over graph slots [0, n_vectors) into `sums` (pre-zeroed).
// Threads stride over dims and blocks over vectors, so a warp reads
// consecutive dims of one vector.
template <typename GRAPH_CFG>
__global__ void rabitq_centroid_sum_kernel(
    typename graph<GRAPH_CFG>::device_view g,
    double* __restrict__ sums) {
  using index_t = typename GRAPH_CFG::index_t;
  for (uint32_t j = threadIdx.x; j < g.dim; j += blockDim.x) {
    double s = 0;
    for (index_t r = blockIdx.x; r < g.n_vectors; r += gridDim.x) {
      s += static_cast<double>(
          static_cast<float>(g.get_vector(g.global_offset + r)[j]));
    }
    atomicAdd(&sums[j], s);
  }
}

// Copy graph slots [start, start + count) into contiguous float rows
// [count * dim] (no row padding), the layout cuBLAS rotation expects.
template <typename GRAPH_CFG>
__global__ void rabitq_gather_kernel(
    typename graph<GRAPH_CFG>::device_view g,
    typename GRAPH_CFG::index_t start,
    typename GRAPH_CFG::index_t count,
    float* __restrict__ out) {
  using index_t = typename GRAPH_CFG::index_t;
  const uint64_t total = static_cast<uint64_t>(count) * g.dim;
  for (uint64_t t = blockIdx.x * static_cast<uint64_t>(blockDim.x) + threadIdx.x;
       t < total; t += static_cast<uint64_t>(gridDim.x) * blockDim.x) {
    const index_t  r = static_cast<index_t>(t / g.dim);
    const uint32_t j = static_cast<uint32_t>(t % g.dim);
    out[t] = static_cast<float>(g.get_vector(g.global_offset + start + r)[j]);
  }
}

// Quantize every vector slot of `g` with RaBitQ at `bits_per_dim` bits per
// dimension (1, 2, 4 or 8). Vectors are centered on the dataset centroid and
// rotated by a random orthogonal matrix seeded by `seed`, then quantized in
// batches so the float staging buffers stay bounded. The graph itself is not
// modified. Caller frees the result with rabitq_index::deallocate().
template <typename GRAPH_CFG>
__host__ rabitq_index build_rabitq_index(const graph<GRAPH_CFG>& g,
                                         uint32_t bits_per_dim = 4,
                                         uint64_t seed = 0) {
  using index_t = typename GRAPH_CFG::index_t;

  if (bits_per_dim != 1 && bits_per_dim != 2 && bits_per_dim != 4 &&
      bits_per_dim != 8) {
    throw std::invalid_argument(
        "build_rabitq_index: bits_per_dim must be 1, 2, 4 or 8 (got " +
        std::to_string(bits_per_dim) + ")");
  }

  auto lk = g.lock_shared();  // reads vectors only; blocks concurrent mutation
  if (g.n_vectors == 0) {
    throw std::runtime_error("build_rabitq_index: graph is empty");
  }

  auto check_cuda = [](cudaError_t err, const char* name) {
    if (err != cudaSuccess) {
      throw std::runtime_error(std::string("build_rabitq_index: ") + name +
                               " failed: " + cudaGetErrorString(err));
    }
  };

  const uint32_t dim = g.dim;
  const index_t  n   = g.n_vectors;
  auto view = g.view();

  rabitq_index rq;
  rq.metric = (GRAPH_CFG::dist_func == distance_func::L2) ? METRIC_L2
                                                          : METRIC_IP;

  // 1. Random orthogonal rotation P.
  {
    std::vector<float> h_P(static_cast<size_t>(dim) * dim);
    std::vector<float> h_Pt(h_P.size());
    set_rotation_matrix(static_cast<int>(dim), h_P.data(), h_Pt.data(), seed);
    check_cuda(cudaMalloc(&rq.d_rotation, sizeof(float) * h_P.size()),
               "cudaMalloc rotation");
    check_cuda(cudaMemcpy(rq.d_rotation, h_P.data(), sizeof(float) * h_P.size(),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy rotation");
  }

  cublasHandle_t handle;
  cublasCreate(&handle);

  // 2. Dataset centroid, rotated into the same space as the data.
  {
    double* d_sums = nullptr;
    check_cuda(cudaMalloc(&d_sums, sizeof(double) * dim), "cudaMalloc centroid sums");
    cudaMemset(d_sums, 0, sizeof(double) * dim);
    const uint32_t grid = static_cast<uint32_t>(std::min<index_t>(n, 1024));
    rabitq_centroid_sum_kernel<GRAPH_CFG><<<grid, 256>>>(view, d_sums);
    check_cuda(cudaDeviceSynchronize(), "centroid kernel");

    std::vector<double> h_sums(dim);
    cudaMemcpy(h_sums.data(), d_sums, sizeof(double) * dim, cudaMemcpyDeviceToHost);
    cudaFree(d_sums);

    std::vector<float> h_centroid(dim);
    for (uint32_t j = 0; j < dim; j++) {
      h_centroid[j] = static_cast<float>(h_sums[j] / static_cast<double>(n));
    }

    float* d_centroid = nullptr;
    check_cuda(cudaMalloc(&d_centroid, sizeof(float) * dim), "cudaMalloc centroid");
    check_cuda(cudaMalloc(&rq.d_rot_centroid, sizeof(float) * dim),
               "cudaMalloc rotated centroid");
    cudaMemcpy(d_centroid, h_centroid.data(), sizeof(float) * dim,
               cudaMemcpyHostToDevice);
    rotate_single_data_vec(handle, d_centroid, rq.d_rot_centroid, dim,
                           rq.d_rotation);
    cudaFree(d_centroid);
  }

  // 3. Quantize slot batches: gather -> rotate -> quantize into the store.
  const double t_const = rabitq_t_const(dim, bits_per_dim);

  rq.codes.n_vectors    = n;
  rq.codes.dim          = dim;
  rq.codes.bits_per_dim = bits_per_dim;
  check_cuda(cudaMalloc(&rq.codes.data, rq.codes.total_bytes()),
             "cudaMalloc codes");

  // ~256MB per float staging buffer (two of them: gathered + rotated).
  constexpr size_t STAGING_BYTES = size_t{256} << 20;
  const index_t batch = static_cast<index_t>(std::clamp<size_t>(
      STAGING_BYTES / (sizeof(float) * dim), 1, n));

  float* d_batch     = nullptr;
  float* d_batch_rot = nullptr;
  check_cuda(cudaMalloc(&d_batch, sizeof(float) * batch * dim), "cudaMalloc batch");
  check_cuda(cudaMalloc(&d_batch_rot, sizeof(float) * batch * dim),
             "cudaMalloc rotated batch");

  for (index_t start = 0; start < n; start += batch) {
    const index_t count = std::min<index_t>(batch, n - start);

    const uint64_t total = static_cast<uint64_t>(count) * dim;
    const uint32_t grid = static_cast<uint32_t>(
        std::min<uint64_t>((total + 255) / 256, 65535));
    rabitq_gather_kernel<GRAPH_CFG><<<grid, 256>>>(view, start, count, d_batch);
    check_cuda(cudaGetLastError(), "gather kernel");

    rotate_data_vec(handle, d_batch, d_batch_rot, count, dim, rq.d_rotation);

    rabitq_data_store<float> window = rq.codes;
    window.data      += static_cast<size_t>(start) * rq.codes.node_bytes();
    window.n_vectors  = count;
    switch (bits_per_dim) {
      case 1: rabitq_quantize_rotated<1>(d_batch_rot, count, rq.d_rot_centroid, t_const, rq.metric, window); break;
      case 2: rabitq_quantize_rotated<2>(d_batch_rot, count, rq.d_rot_centroid, t_const, rq.metric, window); break;
      case 4: rabitq_quantize_rotated<4>(d_batch_rot, count, rq.d_rot_centroid, t_const, rq.metric, window); break;
      case 8: rabitq_quantize_rotated<8>(d_batch_rot, count, rq.d_rot_centroid, t_const, rq.metric, window); break;
    }
  }

  cudaFree(d_batch);
  cudaFree(d_batch_rot);
  cublasDestroy(handle);
  check_cuda(cudaDeviceSynchronize(), "quantization");

  std::cout << "[build_rabitq_index]: " << static_cast<uint64_t>(n)
            << " vectors, dim=" << dim << ", bits_per_dim=" << bits_per_dim
            << ", t_const=" << t_const << ", "
            << (rq.codes.total_bytes() >> 20) << " MB of codes\n";
  return rq;
}

}  // namespace jasper
