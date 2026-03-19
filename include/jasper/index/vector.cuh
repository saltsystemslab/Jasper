#pragma once

#include <fstream>

namespace jasper {

template <typename DATA_T>
struct __align__(16) vector_view {
  DATA_T *data;
  uint32_t dim;
  uint32_t padded_dim;     // padded dim, guarantees 16-byte aligned rows
  uint32_t n_vectors;

  static constexpr uint32_t alignment = 16 / sizeof(DATA_T);

  __host__ __device__ static uint32_t pad(uint32_t d) {
    return (d + alignment - 1) / alignment * alignment;
  }

  __host__ __device__ vector_view() : data(nullptr), dim(0), padded_dim(0), n_vectors(0) {}

  __host__ __device__ vector_view(DATA_T* data, uint32_t dim, uint32_t n_vectors)
      : data(data), dim(dim), padded_dim(pad(dim)), n_vectors(n_vectors) {}

  // Private ctor that preserves an existing padded_dim (used by subview)
  __host__ __device__ vector_view(DATA_T* data, uint32_t dim, uint32_t padded_dim, uint32_t n_vectors)
      : data(data), dim(dim), padded_dim(padded_dim), n_vectors(n_vectors) {}

  __host__ size_t size_bytes() const {
    return sizeof(DATA_T) * static_cast<size_t>(n_vectors) * padded_dim;
  }

  // Copy this view's data to device, return a new view pointing to device memory.
  // Caller is responsible for freeing the returned view's data via cudaFree.
  __host__ vector_view<DATA_T> to_device() const {
    DATA_T* d_data = nullptr;
    cudaMalloc(&d_data, size_bytes());
    cudaMemcpy(d_data, data, size_bytes(), cudaMemcpyHostToDevice);
    return {d_data, dim, padded_dim, n_vectors};
  }

  // Copy this view's data to pinned host memory, return a new view.
  // Caller is responsible for freeing the returned view's data via cudaFreeHost.
  __host__ vector_view<DATA_T> to_host() const {
    DATA_T* h_data = nullptr;
    cudaMallocHost(&h_data, size_bytes());
    cudaMemcpy(h_data, data, size_bytes(), cudaMemcpyDeviceToHost);
    return {h_data, dim, padded_dim, n_vectors};
  }

  // Access vector i as a pointer: &data[i * padded_dim]
  __host__ __device__ DATA_T* operator[](uint32_t i) {
    return &data[i * padded_dim];
  }

  __host__ __device__ const DATA_T* operator[](uint32_t i) const {
    return &data[i * padded_dim];
  }

  // Return a read-only, non-owning view over a contiguous subset of vectors.
  // Starts at vector index `offset`, spanning `count` vectors.
  __host__ __device__ vector_view<DATA_T> subview(uint32_t offset, uint32_t count) const {
    assert(offset + count <= n_vectors);
    return {data + static_cast<size_t>(offset) * padded_dim, dim, padded_dim, count};
  }
};
// Returns a vector_view pointing to pinned host memory.
// The caller is responsible for freeing the data via cudaFreeHost.
template <typename DATA_T>
__host__ vector_view<DATA_T> load_vectors_from_file(std::string filename) {
  std::ifstream file(filename, std::ios::in | std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Error opening file: " << filename << std::endl;
    return {nullptr, 0, 0};
  }

  file.seekg(0, std::ios::end);
  std::streamsize file_size = file.tellg();
  file.seekg(0, std::ios::beg);

  int32_t n_data_points;
  int32_t n_dimensions;
  file.read(reinterpret_cast<char*>(&n_data_points), sizeof(int32_t));
  file.read(reinterpret_cast<char*>(&n_dimensions), sizeof(int32_t));

  std::streamsize data_size = file_size - 2 * sizeof(int32_t);

  std::cout << "Read " << data_size << " bytes of data\n";
  std::cout << "n_data_points=" << n_data_points
            << " dim=" << n_dimensions << "\n";

  std::streamsize expected = static_cast<std::streamsize>(sizeof(DATA_T))
                           * n_data_points * n_dimensions;
  if (expected != data_size) {
    std::cerr << "Size mismatch: expected " << expected
              << " bytes but file has " << data_size << " bytes\n";
    return {nullptr, 0, 0};
  }

  uint32_t dim = static_cast<uint32_t>(n_dimensions);
  uint32_t padded_dim = vector_view<DATA_T>::pad(dim);

  // Allocate padded pinned memory, zero-filled so pad elements are 0
  size_t padded_bytes = sizeof(DATA_T) * static_cast<size_t>(n_data_points) * padded_dim;
  DATA_T* host_data;
  cudaMallocHost(&host_data, padded_bytes);
  memset(host_data, 0, padded_bytes);

  if (dim == padded_dim) {
    // No padding needed — read directly
    if (!file.read(reinterpret_cast<char*>(host_data), data_size)) {
      std::cerr << "Error reading file\n";
      cudaFreeHost(host_data);
      return {nullptr, 0, 0};
    }
  } else {
    // Read each vector into its padded row
    size_t row_bytes = sizeof(DATA_T) * dim;
    for (int32_t i = 0; i < n_data_points; i++) {
      if (!file.read(reinterpret_cast<char*>(host_data + static_cast<size_t>(i) * padded_dim), row_bytes)) {
        std::cerr << "Error reading vector " << i << "\n";
        cudaFreeHost(host_data);
        return {nullptr, 0, 0};
      }
    }
  }

  std::cout << "Successfully read " << n_data_points
            << " vectors (dim=" << dim
            << ", padded_dim=" << padded_dim
            << ") from " << filename << "\n";

  return {host_data, dim, n_data_points};
}

}