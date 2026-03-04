#pragma once

#include <cstdint>
#include <cuda_runtime.h>

#include <cassert>

namespace jasper {

template <typename T = float>
struct rabitq_node_view {
  uint8_t* bin_code;
  T* f_add;
  T* f_rescale;

  __host__ __device__ rabitq_node_view() = default;

  // Construct from a pointer to this node's data region
  __host__ __device__ rabitq_node_view(char* data, uint32_t code_bytes)
      : bin_code(reinterpret_cast<uint8_t*>(data))
      , f_add(reinterpret_cast<T*>(data + code_bytes))
      , f_rescale(f_add + 1) {}
};

template <typename T = float>
struct const_rabitq_node_view {
  const uint8_t* bin_code;
  const T* f_add;
  const T* f_rescale;

  __host__ __device__ const_rabitq_node_view() = default;

  __host__ __device__ const_rabitq_node_view(const char* data, uint32_t code_bytes)
      : bin_code(reinterpret_cast<const uint8_t*>(data))
      , f_add(reinterpret_cast<const T*>(data + code_bytes))
      , f_rescale(f_add + 1) {}
};

// ── Collection of all nodes (array-of-structs in a flat buffer) ──
// This is what the graph stores. Indexing into it gives a node view.

template <typename T = float>
struct rabitq_data_store {
  char*    data;          // flat buffer: all nodes packed contiguously
  uint32_t n_vectors;
  uint32_t dim;           // original dimension
  uint32_t bits_per_dim;  // quantization bits (1, 2, 4)

  __host__ __device__ rabitq_data_store()
      : data(nullptr), n_vectors(0), dim(0), bits_per_dim(0) {}

  // Bytes for the binary code portion of one node
  __host__ __device__ uint32_t code_bytes() const {
    uint32_t raw = (bits_per_dim * dim + 7) / 8;
    // align by the size of metadata
    uint32_t align = sizeof(T);
    return (raw + align - 1) / align * align;
  }

  // Total bytes per node
  __host__ __device__ uint32_t node_bytes() const {
    uint32_t n_scalars = 2;
    return code_bytes() + sizeof(T) * n_scalars;
  }

  // Total bytes for the entire store
  __host__ __device__ size_t total_bytes() const {
    return static_cast<size_t>(node_bytes()) * n_vectors;
  }

  // Get a mutable view of node i
  __host__ __device__ rabitq_node_view<T> operator[](uint32_t i) {
    return rabitq_node_view<T>(data + i * node_bytes(), code_bytes());
  }

  // Get a const view of node i
  __host__ __device__ const_rabitq_node_view<T> operator[](uint32_t i) const {
    return const_rabitq_node_view<T>(data + i * node_bytes(), code_bytes());
  }

  __host__ static rabitq_data_store<T> allocate_host(
      uint32_t n_vectors, uint32_t dim, uint32_t bits_per_dim) {
    rabitq_data_store<T> store;
    store.n_vectors = n_vectors;
    store.dim = dim;
    store.bits_per_dim = bits_per_dim;
    cudaMallocHost(&store.data, store.total_bytes());
    return store;
  }

  __host__ static rabitq_data_store<T> allocate_device(
      uint32_t n_vectors, uint32_t dim, uint32_t bits_per_dim) {
    rabitq_data_store<T> store;
    store.n_vectors = n_vectors;
    store.dim = dim;
    store.bits_per_dim = bits_per_dim;
    cudaMalloc(&store.data, store.total_bytes());
    return store;
  }

  // Copy to device, returns a new store pointing to device memory
  __host__ rabitq_data_store<T> to_device() const {
    rabitq_data_store<T> d_store = *this;
    cudaMalloc(&d_store.data, total_bytes());
    cudaMemcpy(d_store.data, data, total_bytes(), cudaMemcpyHostToDevice);
    return d_store;
  }

  // Copy to host, returns a new store pointing to pinned host memory
  __host__ rabitq_data_store<T> to_host() const {
    rabitq_data_store<T> h_store = *this;
    cudaMallocHost(&h_store.data, total_bytes());
    cudaMemcpy(h_store.data, data, total_bytes(), cudaMemcpyDeviceToHost);
    return h_store;
  }
};

// Generic: extract `bits_per_dim` bits starting at dimension i
__host__ __device__ inline uint8_t extract_bits(
    const uint8_t* code, uint32_t i, uint32_t bits_per_dim) {
  uint32_t bit_idx = i * bits_per_dim;
  uint32_t byte_idx = bit_idx / 8;
  uint32_t bit_off = bit_idx % 8;
  uint16_t chunk = code[byte_idx];
  if (bit_off + bits_per_dim > 8) {
    chunk |= uint16_t(code[byte_idx + 1]) << 8;
  }
  return (chunk >> bit_off) & ((1u << bits_per_dim) - 1u);
}

// Fast specializations
__host__ __device__ inline uint8_t extract_1bit(const uint8_t* code, uint32_t i) {
  return (code[i >> 3] >> (i & 7)) & 1u;
}

__host__ __device__ inline uint8_t extract_2bit(const uint8_t* code, uint32_t i) {
  return (code[i >> 2] >> ((i & 3) << 1)) & 0x3u;
}

__host__ __device__ inline uint8_t extract_4bit(const uint8_t* code, uint32_t i) {
  return (code[i >> 1] >> ((i & 1) << 2)) & 0xFu;
}

// struct to store query metadata
// Quantized query vector
struct RabitqQueryFactor {
  float add = 0;
  float k1xSumq = 0;
  float kBxSumq = 0;
};

} // namespace jasper