#pragma once

#include <cublas_v2.h>
#include <curand.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <queue>
#include <vector>
#include <iostream>
#include <stdexcept>
#include <string>

#include "jasper/rabitq/layout.cuh"
#include "jasper/rabitq/distance.cuh"
#include "jasper/rotation/rotation.cuh"
#include "jasper/index/vector.cuh"

namespace cg = cooperative_groups;

namespace jasper {


constexpr std::array<float, 9> kTightStart = {
    0, 0.15, 0.20, 0.52, 0.59, 0.71, 0.75, 0.77, 0.81
};

__host__ inline double best_rescale_factor(
    const float* o_abs, size_t dim, size_t ex_bits) {
  constexpr double kEps = 1e-5;
  constexpr int kNEnum = 10;
  double max_o = *std::max_element(o_abs, o_abs + dim);

  double t_end = static_cast<double>(((1 << ex_bits) - 1) + kNEnum) / max_o;
  double t_start = t_end * kTightStart[ex_bits];

  std::vector<int> cur_o_bar(dim);
  double sqr_denominator = static_cast<double>(dim) * 0.25;
  double numerator = 0;

  for (size_t i = 0; i < dim; ++i) {
    int cur = static_cast<int>((t_start * o_abs[i]) + kEps);
    cur_o_bar[i] = cur;
    sqr_denominator += cur * cur + cur;
    numerator += (cur + 0.5) * o_abs[i];
  }

  std::priority_queue<std::pair<double, size_t>,
                      std::vector<std::pair<double, size_t>>,
                      std::greater<>>
      next_t;

  for (size_t i = 0; i < dim; ++i) {
    next_t.emplace(static_cast<double>(cur_o_bar[i] + 1) / o_abs[i], i);
  }

  double max_ip = 0;
  double t = 0;

  while (!next_t.empty()) {
    double cur_t = next_t.top().first;
    size_t update_id = next_t.top().second;
    next_t.pop();

    cur_o_bar[update_id]++;
    int update_o_bar = cur_o_bar[update_id];
    sqr_denominator += 2 * update_o_bar;
    numerator += o_abs[update_id];

    double cur_ip = numerator / std::sqrt(sqr_denominator);
    if (cur_ip > max_ip) {
      max_ip = cur_ip;
      t = cur_t;
    }

    if (update_o_bar < (1 << ex_bits) - 1) {
      double t_next = static_cast<double>(update_o_bar + 1) / o_abs[update_id];
      if (t_next < t_end) {
        next_t.emplace(t_next, update_id);
      }
    }
  }
  return t;
}

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

// Rescale constant for a `bits_per_dim`-bit code. One bit is the sign, so the
// magnitude calc_multi_code clamps to has bits_per_dim - 1 bits; t must be
// fitted to that range, or the largest coordinates saturate. 1-bit codes are
// sign-only and ignore t.
static __host__ double rabitq_t_const(uint32_t dim, uint32_t bits_per_dim) {
  if (bits_per_dim <= 1) return 0.0;
  return get_const_scaling_factors(dim, bits_per_dim - 1);
}

// ═══════════════════════════════════════════════════════════════
// Device kernels — runtime dim via shared memory
// ═══════════════════════════════════════════════════════════════

// Shared memory layout per tile:
//   float    residual_vec[dim]
//   uint8_t  uncompressed_code[dim]

// Calculate quantized code for one vector.
// residual_vec and uncompressed_code live in shared memory.
template <uint32_t SIZE_PER_DIM, uint16_t TILE_SIZE, typename ParentT>
__device__ float calc_multi_code(
    float* residual_vec,
    uint8_t* uncompressed_code,
    uint32_t dim,
    double t_const,
    float l2_norm,
    cg::thread_block_tile<TILE_SIZE, ParentT>& tile) {

  constexpr double kEps = 1e-5;

  float ip_norm_tmp = 0;
  for (uint32_t i = tile.thread_rank(); i < dim; i += tile.size()) {
    float abs_o = fabsf(residual_vec[i] / l2_norm);
    int val = static_cast<int>((t_const * abs_o) + kEps);
    if (val >= (1 << (SIZE_PER_DIM - 1))) {
      val = (1 << (SIZE_PER_DIM - 1)) - 1;
    }
    uncompressed_code[i] = static_cast<uint8_t>(val);
    ip_norm_tmp += (val + 0.5f) * abs_o;
  }
  float ip_norm = cg::reduce(tile, ip_norm_tmp, cg::plus<float>());
  float ip_norm_inv = 1.0f / ip_norm;
  if (isnan(ip_norm_inv)) ip_norm_inv = 1.0f;

  // Apply sign encoding
  uint32_t const mask = (1 << (SIZE_PER_DIM - 1)) - 1;
  for (uint32_t i = tile.thread_rank(); i < dim; i += tile.size()) {
    if (residual_vec[i] >= 0) {
      uncompressed_code[i] += 1 << (SIZE_PER_DIM - 1);
    } else {
      uncompressed_code[i] = (~uncompressed_code[i]) & mask;
    }
  }
  tile.sync();

  return ip_norm_inv;
}

__host__ __device__ inline uint32_t rabitq_quantize_smem_per_tile(uint32_t dim) {
  return (static_cast<uint32_t>(sizeof(float) + sizeof(uint8_t)) * dim + 3u) & ~3u;
}

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
  // Layout: [tiles_per_block][dim floats + dim uint8s], stride rounded up to 4B
  // so every tile's float array stays aligned for odd dims.
  extern __shared__ char rq_smem[];
  uint32_t per_tile_bytes = rabitq_quantize_smem_per_tile(dim);
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

// ── Convert (padded) rows to contiguous float rows ─────────────
// in:  [n_vectors * in_stride] (e.g. a vector_view's padded rows)
// out: [n_vectors * dim], the unpadded row-major layout cuBLAS rotation expects
template <typename T>
__global__ void rows_to_float_kernel(
    const T* __restrict__ in,
    uint32_t in_stride,
    float* __restrict__ out,
    uint32_t dim,
    uint64_t n_vectors) {
  uint64_t total = n_vectors * dim;
  for (uint64_t idx = blockIdx.x * static_cast<uint64_t>(blockDim.x) + threadIdx.x;
       idx < total; idx += static_cast<uint64_t>(gridDim.x) * blockDim.x) {
    uint64_t row = idx / dim;
    uint32_t col = static_cast<uint32_t>(idx % dim);
    out[idx] = static_cast<float>(in[row * in_stride + col]);
  }
}

template <typename T>
__host__ void rows_to_float(const T* in, uint32_t in_stride, float* out,
                            uint32_t dim, uint64_t n_vectors,
                            cudaStream_t stream = 0) {
  uint64_t total = n_vectors * dim;
  uint32_t grid = static_cast<uint32_t>(
      std::min<uint64_t>((total + 255) / 256, 65535));
  if (grid == 0) return;
  rows_to_float_kernel<<<grid, 256, 0, stream>>>(in, in_stride, out, dim, n_vectors);
}

// Per-query estimator factors, one tile per query. `add` is the query half of
// the additive term: ||q_r - c_r||^2 for L2, -<q_r, c_r> for IP.
template <uint32_t TILE_SIZE = 4>
__global__ void rabitq_query_factor_kernel(
    const float* __restrict__ rot_queries,   // [n_queries * dim]
    uint32_t n_queries,
    uint32_t dim,
    const float* __restrict__ rot_centroid,  // [dim]
    uint32_t bits_per_dim,
    MetricType metric_type,
    RabitqQueryFactor* __restrict__ factors) {

  auto block = cg::this_thread_block();
  auto tile = cg::tiled_partition<TILE_SIZE>(block);

  uint64_t tiles_per_block = blockDim.x / TILE_SIZE;
  uint64_t global_tile_id = block.thread_rank() / TILE_SIZE + blockIdx.x * tiles_per_block;
  uint64_t total_tiles = gridDim.x * tiles_per_block;

  const float c_1 = -0.5f;
  const float c_b = -static_cast<float>((1u << bits_per_dim) - 1u) / 2.0f;

  for (uint64_t i = global_tile_id; i < n_queries; i += total_tiles) {
    const float* q = rot_queries + i * dim;
    float sqr_norm_tmp = 0, ip_cent_tmp = 0, sumq_tmp = 0;
    for (uint32_t j = tile.thread_rank(); j < dim; j += tile.size()) {
      float diff = q[j] - rot_centroid[j];
      sqr_norm_tmp += diff * diff;
      ip_cent_tmp  += q[j] * rot_centroid[j];
      sumq_tmp     += q[j];
    }
    float sqr_norm = cg::reduce(tile, sqr_norm_tmp, cg::plus<float>());
    float ip_cent  = cg::reduce(tile, ip_cent_tmp,  cg::plus<float>());
    float sumq     = cg::reduce(tile, sumq_tmp,     cg::plus<float>());

    if (tile.thread_rank() == 0) {
      RabitqQueryFactor fac;
      fac.add     = (metric_type == METRIC_L2) ? sqr_norm : -ip_cent;
      fac.k1xSumq = c_1 * sumq;
      fac.kBxSumq = c_b * sumq;
      factors[i] = fac;
    }
  }
}

namespace detail {

// One cuBLAS handle per host thread for the per-search query rotation:
// creating a handle costs more than rotating a query batch, and a handle must
// not be shared by threads that search concurrently (search only takes a
// shared lock on the graph).
inline cublasHandle_t thread_local_cublas_handle() {
  struct handle_holder {
    cublasHandle_t handle = nullptr;
    handle_holder()  { cublasCreate(&handle); }
    ~handle_holder() { if (handle) cublasDestroy(handle); }
  };
  thread_local handle_holder holder;
  return holder.handle;
}

}  // namespace detail

// ═══════════════════════════════════════════════════════════════
// Host entry points
// ═══════════════════════════════════════════════════════════════

// Quantize n_vectors already-rotated rows into out_store[0, n_vectors).
// `out_store` may be a window into a larger store (data offset by whole
// nodes), which is how batched builds fill one store incrementally.
template <uint32_t SIZE_PER_DIM>
__host__ void rabitq_quantize_rotated(
    const float* d_rot_vectors,            // device, [n_vectors * dim]
    uint64_t n_vectors,
    const float* d_rot_centroid,           // device, [dim]
    double t_const,
    MetricType metric_type,
    rabitq_data_store<float> out_store,    // device store to write into
    cudaStream_t stream = 0) {
  if (n_vectors == 0) return;

  // Size the block to the per-tile smem workspace so large dims still launch
  // (64 tiles of dim=960 would need ~300KB).
  constexpr uint32_t TILE_SIZE = 4;
  constexpr uint32_t MAX_TILES_PER_BLOCK = 64;
  constexpr size_t   DEFAULT_SMEM_LIMIT = 48 * 1024;
  const size_t smem_per_tile = rabitq_quantize_smem_per_tile(out_store.dim);
  const uint32_t tiles_per_block = static_cast<uint32_t>(std::clamp<size_t>(
      DEFAULT_SMEM_LIMIT / smem_per_tile, 1, MAX_TILES_PER_BLOCK));
  const size_t smem = smem_per_tile * tiles_per_block;

  auto kernel = rabitq_quantize_kernel<SIZE_PER_DIM, TILE_SIZE>;
  if (smem > DEFAULT_SMEM_LIMIT) {
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         static_cast<int>(smem));
  }
  uint32_t grid = static_cast<uint32_t>(std::min<uint64_t>(
      (n_vectors + tiles_per_block - 1) / tiles_per_block, 1u << 20));

  kernel<<<grid, tiles_per_block * TILE_SIZE, smem, stream>>>(
      d_rot_vectors, n_vectors, out_store, d_rot_centroid, t_const, metric_type);

  cudaError_t err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    std::cerr << "Quantize kernel failed: "
              << cudaGetErrorString(err) << std::endl;
  }
}

template <uint32_t SIZE_PER_DIM = 1>
__host__ rabitq_data_store<float> rabitq_quantize(
    vector_view<float> original_vectors,   // device, [n_vectors * padded_dim]
    float* d_rotation_matrix,              // device, [dim * dim] column-major
    float* d_centroid,                     // device, [dim]
    MetricType metric_type) {

  uint32_t dim = original_vectors.dim;
  uint64_t n_vectors = original_vectors.n_vectors;

  cublasHandle_t handle;
  cublasCreate(&handle);

  // cuBLAS rotation reads unpadded rows; compact padded rows first.
  float* d_vectors = original_vectors.data;
  if (original_vectors.padded_dim != dim) {
    cudaMalloc(&d_vectors, sizeof(float) * n_vectors * dim);
    rows_to_float(original_vectors.data, original_vectors.padded_dim,
                  d_vectors, dim, n_vectors);
  }

  // Rotate data vectors: d_rot = P^T * data^T → stored as [n_vectors * dim]
  float* d_rot_vectors;
  cudaMalloc(&d_rot_vectors, sizeof(float) * n_vectors * dim);
  rotate_data_vec(handle, d_vectors, d_rot_vectors,
                  n_vectors, dim, d_rotation_matrix);
  if (d_vectors != original_vectors.data) cudaFree(d_vectors);

  // Rotate centroid
  float* d_rot_centroid;
  cudaMalloc(&d_rot_centroid, sizeof(float) * dim);
  rotate_single_data_vec(handle, d_centroid, d_rot_centroid, dim,
                         d_rotation_matrix);
  cudaDeviceSynchronize();

  // Compute t_const
  double t_const = rabitq_t_const(dim, SIZE_PER_DIM);
  std::cout << "t_const=" << t_const << std::endl;

  // Allocate output store on device
  auto out_store = rabitq_data_store<float>::allocate_device(
      n_vectors, dim, SIZE_PER_DIM);

  rabitq_quantize_rotated<SIZE_PER_DIM>(
      d_rot_vectors, n_vectors, d_rot_centroid, t_const, metric_type,
      out_store);

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

  // Convert to float (dropping row padding)
  float* d_float;
  cudaMalloc(&d_float, sizeof(float) * n_vectors * dim);
  rows_to_float(original_vectors.data, original_vectors.padded_dim,
                d_float, dim, n_vectors);
  cudaDeviceSynchronize();

  vector_view<float> float_view(d_float, dim, dim, n_vectors, false /*on_host*/);
  auto result = rabitq_quantize<SIZE_PER_DIM>(
      float_view, d_rotation_matrix, d_centroid, metric_type);

  cudaFree(d_float);
  return result;
}

// Rotate a batch of queries into the codes' space (P^T q, same as the data)
// and compute their estimator factors. `d_queries` holds padded rows of any
// element type; `bits_per_dim` and `metric_type` must match the data store.
// Caller frees the result with rabitq_queries::deallocate().
template <typename DATA_T>
__host__ rabitq_queries rabitq_prepare_queries(
    vector_view<DATA_T> d_queries,
    float* d_rotation_matrix,              // device, [dim * dim] column-major
    const float* d_rot_centroid,           // device, [dim], already rotated
    uint32_t bits_per_dim,
    MetricType metric_type,
    cudaStream_t stream = 0) {

  rabitq_queries out;
  out.n_queries = d_queries.n_vectors;
  out.dim       = d_queries.dim;
  if (out.n_queries == 0) return out;

  const uint64_t n   = out.n_queries;
  const uint32_t dim = out.dim;

  float* d_flat;
  cudaMalloc(&d_flat, sizeof(float) * n * dim);
  cudaMalloc(&out.rot_queries, sizeof(float) * n * dim);
  cudaMalloc(&out.factors, sizeof(RabitqQueryFactor) * n);

  rows_to_float(d_queries.data, d_queries.padded_dim, d_flat, dim, n, stream);

  cublasHandle_t handle = detail::thread_local_cublas_handle();
  cublasSetStream(handle, stream);
  rotate_data_vec(handle, d_flat, out.rot_queries, n, dim, d_rotation_matrix);
  cudaFree(d_flat);

  constexpr uint32_t TILE_SIZE = 4;
  constexpr uint32_t BLOCK_SIZE = 256;
  constexpr uint32_t tiles_per_block = BLOCK_SIZE / TILE_SIZE;
  uint32_t grid = static_cast<uint32_t>(std::min<uint64_t>(
      (n + tiles_per_block - 1) / tiles_per_block, 65535));
  rabitq_query_factor_kernel<TILE_SIZE><<<grid, BLOCK_SIZE, 0, stream>>>(
      out.rot_queries, out.n_queries, dim, d_rot_centroid,
      bits_per_dim, metric_type, out.factors);

  cudaError_t err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    out.deallocate();
    throw std::runtime_error(std::string("rabitq_prepare_queries failed: ") +
                             cudaGetErrorString(err));
  }
  return out;
}

} // namespace jasper