#pragma once

#include <cublas_v2.h>
#include <curand.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cmath>
#include <queue>
#include <vector>
#include <iostream>

#include "jasper/rabitq/encode.cuh"
#include "jasper/rabitq/layout.cuh"
#include "jasper/rabitq/distance.cuh"
#include "jasper/rotation/rotation.cuh"
#include "jasper/index/vector.cuh"

namespace cg = cooperative_groups;

namespace jasper {

// kTightStart / best_rescale_factor / calc_multi_code now live in encode.cuh,
// shared with the directional (per-edge) quantizer in rabitq/rq_kernels.cuh.

static __global__ void abs_kernel(float* data, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) data[idx] = fabsf(data[idx]);
}

static __host__ double get_const_scaling_factors(uint32_t dim, uint32_t bits_per_dim) {
  constexpr int32_t n_samples = 1000;

  float* d_vectors;
  cudaMalloc(&d_vectors, sizeof(float) * n_samples * dim);

  curandGenerator_t gen;
  curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
  curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
  curandGenerateNormal(gen, d_vectors, n_samples * dim, 0.0f, 1.0f);
  curandDestroyGenerator(gen);

  // Normalize each vector
  cublasHandle_t handle;
  cublasCreate(&handle);
  for (int i = 0; i < n_samples; i++) {
    float* row = d_vectors + i * dim;
    float norm = 0;
    cublasSnrm2(handle, dim, row, 1, &norm);
    float inv = 1.0f / norm;
    cublasSscal(handle, dim, &inv, row, 1);
  }
  cublasDestroy(handle);

  // Abs
  abs_kernel<<<(n_samples * dim + 255) / 256, 256>>>(d_vectors, n_samples * dim);
  cudaDeviceSynchronize();

  // Copy to host for best_rescale_factor (host-only function)
  float* h_vectors;
  cudaMallocHost(&h_vectors, sizeof(float) * n_samples * dim);
  cudaMemcpy(h_vectors, d_vectors, sizeof(float) * n_samples * dim,
             cudaMemcpyDeviceToHost);
  cudaFree(d_vectors);

  double sum = 0;
  for (int i = 0; i < n_samples; i++) {
    sum += best_rescale_factor(h_vectors + i * dim, dim, bits_per_dim);
  }
  cudaFreeHost(h_vectors);

  return sum / n_samples;
}

// ═══════════════════════════════════════════════════════════════
// Device kernels — runtime dim via shared memory
// ═══════════════════════════════════════════════════════════════

// Shared memory layout per tile:
//   float    residual_vec[dim]
//   uint8_t  uncompressed_code[dim]

// Main quantize kernel — SIZE_PER_DIM is compile-time, dim is runtime
template <uint32_t SIZE_PER_DIM, uint16_t TILE_SIZE = 4>
__global__ void rabitq_quantize_kernel(
    const float* __restrict__ d_rot_vectors,  // [n_vectors * dim]
    uint64_t n_vectors,
    rabitq_data_store<float> out_store,         // device store to write into
    const float* __restrict__ d_centroid,     // [dim]
    double t_const,
    MetricType metric_type) {

  auto block = cg::this_thread_block();
  auto tile = cg::tiled_partition<TILE_SIZE>(block);

  uint32_t tiles_per_block = blockDim.x / TILE_SIZE;
  uint32_t tile_id_in_block = block.thread_rank() / TILE_SIZE;
  uint64_t global_tile_id = tile_id_in_block + blockIdx.x * tiles_per_block;
  uint64_t total_tiles = gridDim.x * tiles_per_block;

  uint32_t dim = out_store.dim;

  // Dynamic shared memory: per-tile workspace
  // Layout: [tiles_per_block][dim floats + dim uint8s]
  extern __shared__ char rq_smem[];
  uint32_t per_tile_bytes = sizeof(float) * dim + sizeof(uint8_t) * dim;
  float* residual_vec = reinterpret_cast<float*>(
      rq_smem + tile_id_in_block * per_tile_bytes);
  uint8_t* uncompressed_code = reinterpret_cast<uint8_t*>(
      residual_vec + dim);

  for (uint64_t i = global_tile_id; i < n_vectors; i += total_tiles) {
    // Compute residual = rotated_vec - centroid, and L2 norm
    float l2_sqr_tmp = 0;
    for (uint32_t j = tile.thread_rank(); j < dim; j += tile.size()) {
      float r = d_rot_vectors[i * dim + j] - d_centroid[j];
      residual_vec[j] = r;
      l2_sqr_tmp += r * r;
    }
    float l2_sqr = cg::reduce(tile, l2_sqr_tmp, cg::plus<float>());
    float l2_norm = sqrtf(l2_sqr);

    // Quantize
    float ipnorm_inv = calc_multi_code<SIZE_PER_DIM, TILE_SIZE>(
        residual_vec, uncompressed_code, dim, t_const, l2_norm, tile);

    // Compute factors
    float ip_resi_xucb_tmp = 0;
    float ip_cent_xucb_tmp = 0;
    float ip_resi_cent_tmp = 0;
    float cb = -(static_cast<float>(1 << (SIZE_PER_DIM - 1)) - 0.5f);
    for (uint32_t j = tile.thread_rank(); j < dim; j += tile.size()) {
      float xu_cb = static_cast<float>(uncompressed_code[j]) + cb;
      ip_resi_xucb_tmp += residual_vec[j] * xu_cb;
      ip_cent_xucb_tmp += d_centroid[j] * xu_cb;
      ip_resi_cent_tmp += residual_vec[j] * d_centroid[j];
    }
    float ip_resi_xucb = cg::reduce(tile, ip_resi_xucb_tmp, cg::plus<float>());
    float ip_cent_xucb = cg::reduce(tile, ip_cent_xucb_tmp, cg::plus<float>());
    float ip_resi_cent = cg::reduce(tile, ip_resi_cent_tmp, cg::plus<float>());
    if (ip_resi_xucb == 0) {
      ip_resi_xucb = std::numeric_limits<float>::infinity();
    }

    // Write f_add, f_rescale
    auto node = out_store[i];
    if (tile.thread_rank() == 0) {
      if (metric_type == METRIC_L2) {
        *node.f_add = l2_sqr + 2.0f * l2_sqr * ip_cent_xucb / ip_resi_xucb;
        *node.f_rescale = ipnorm_inv * -2.0f * l2_norm;
      } else {
        *node.f_add = 1.0f - ip_resi_cent + l2_sqr * ip_cent_xucb / ip_resi_xucb;
        *node.f_rescale = ipnorm_inv * -l2_norm;
      }
    }

    // Pack uncompressed codes into bit-packed binary codes
    uint32_t code_bytes = out_store.code_bytes();
    for (uint32_t j = tile.thread_rank(); j < code_bytes; j += tile.size()) {
      uint8_t packed = 0;
      // How many dimensions fit in this byte
      uint32_t bits_in_byte = 8;
      uint32_t bit_pos = j * 8;
      while (bits_in_byte >= SIZE_PER_DIM && bit_pos < dim * SIZE_PER_DIM) {
        uint32_t dim_idx = bit_pos / SIZE_PER_DIM;
        uint32_t shift = bit_pos % 8;
        if (dim_idx < dim) {
          packed |= uncompressed_code[dim_idx] << shift;
        }
        bit_pos += SIZE_PER_DIM;
        bits_in_byte -= SIZE_PER_DIM;
      }
      node.bin_code[j] = packed;
    }
    tile.sync();
  }
}

// ── Flatten kernel (replaces data_vector → float*) ─────────────
template <typename T>
__global__ void flatten_to_float_kernel(
    const T* __restrict__ in,  // flat [n_vectors * dim]
    float* __restrict__ out,
    uint32_t dim,
    uint64_t n_vectors) {
  uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t total = n_vectors * dim;
  if (idx < total) {
    out[idx] = static_cast<float>(in[idx]);
  }
}

// ═══════════════════════════════════════════════════════════════
// Host entry point
// ═══════════════════════════════════════════════════════════════

template <uint32_t SIZE_PER_DIM = 1>
__host__ rabitq_data_store<float> rabitq_quantize(
    vector_view<float> original_vectors,   // device, [n_vectors * dim]
    float* d_rotation_matrix,              // device, [dim * dim] column-major
    float* d_centroid,                     // device, [dim]
    MetricType metric_type) {

  uint32_t dim = original_vectors.dim;
  uint64_t n_vectors = original_vectors.n_vectors;

  cublasHandle_t handle;
  cublasCreate(&handle);

  // Rotate data vectors: d_rot = P^T * data^T → stored as [n_vectors * dim]
  float* d_rot_vectors;
  cudaMalloc(&d_rot_vectors, sizeof(float) * n_vectors * dim);
  rotate_data_vec(handle, original_vectors.data, d_rot_vectors,
                  n_vectors, dim, d_rotation_matrix);

  // Rotate centroid
  float* d_rot_centroid;
  cudaMalloc(&d_rot_centroid, sizeof(float) * dim);
  rotate_single_data_vec(handle, d_centroid, d_rot_centroid, dim,
                         d_rotation_matrix);
  cudaDeviceSynchronize();

  // Compute t_const
  double t_const = get_const_scaling_factors(dim, SIZE_PER_DIM);
  std::cout << "t_const=" << t_const << std::endl;

  // Allocate output store on device
  auto out_store = rabitq_data_store<float>::allocate_device(
      n_vectors, dim, SIZE_PER_DIM);

  // Launch quantization kernel
  constexpr uint32_t TILE_SIZE = 4;
  constexpr uint32_t BLOCK_SIZE = 256;
  uint32_t tiles_per_block = BLOCK_SIZE / TILE_SIZE;
  uint32_t grid = (n_vectors + tiles_per_block - 1) / tiles_per_block;
  uint32_t smem_per_tile = sizeof(float) * dim + sizeof(uint8_t) * dim;
  uint32_t smem = smem_per_tile * tiles_per_block;

  rabitq_quantize_kernel<SIZE_PER_DIM, TILE_SIZE>
      <<<grid, BLOCK_SIZE, smem>>>(
          d_rot_vectors, n_vectors, out_store,
          d_rot_centroid, t_const, metric_type);

  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    std::cerr << "Quantize kernel failed: "
              << cudaGetErrorString(err) << std::endl;
  }

  // Cleanup temporaries
  cudaFree(d_rot_vectors);
  cudaFree(d_rot_centroid);
  cublasDestroy(handle);

  return out_store;
}

// ── Overload that converts non-float input ─────────────────────
template <typename DATA_T, uint32_t SIZE_PER_DIM = 1>
__host__ rabitq_data_store<float> rabitq_quantize(
    vector_view<DATA_T> original_vectors,
    float* d_rotation_matrix,
    float* d_centroid,
    MetricType metric_type) {

  uint32_t dim = original_vectors.dim;
  uint64_t n_vectors = original_vectors.n_vectors;
  uint64_t total = n_vectors * dim;

  // Convert to float
  float* d_float;
  cudaMalloc(&d_float, sizeof(float) * total);
  flatten_to_float_kernel<<<(total + 255) / 256, 256>>>(
      original_vectors.data, d_float, dim, n_vectors);
  cudaDeviceSynchronize();

  vector_view<float> float_view(d_float, dim, n_vectors, false /*on_host*/);
  auto result = rabitq_quantize<SIZE_PER_DIM>(
      float_view, d_rotation_matrix, d_centroid, metric_type);

  cudaFree(d_float);
  return result;
}

} // namespace jasper