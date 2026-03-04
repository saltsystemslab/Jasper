#pragma once

#include <fstream>

namespace jasper {

template <typename DATA_T>
struct __align__(16) vector_view {
  DATA_T *data;
  uint32_t dim;
  uint32_t n_vectors;

  __host__ __device__ vector_view() : data(nullptr), dim(0), n_vectors(0) {}

  __host__ __device__ vector_view(DATA_T* data, uint32_t dim, uint32_t n_vectors)
      : data(data), dim(dim), n_vectors(n_vectors) {}

  __host__ size_t size_bytes() const {
    return sizeof(DATA_T) * static_cast<size_t>(n_vectors) * dim;
  }

  // Copy this view's data to device, return a new view pointing to device memory.
  // Caller is responsible for freeing the returned view's data via cudaFree.
  __host__ vector_view<DATA_T> to_device() const {
    DATA_T* d_data = nullptr;
    cudaMalloc(&d_data, size_bytes());
    cudaMemcpy(d_data, data, size_bytes(), cudaMemcpyHostToDevice);
    return {d_data, dim, n_vectors};
  }

  // Copy this view's data to pinned host memory, return a new view.
  // Caller is responsible for freeing the returned view's data via cudaFreeHost.
  __host__ vector_view<DATA_T> to_host() const {
    DATA_T* h_data = nullptr;
    cudaMallocHost(&h_data, size_bytes());
    cudaMemcpy(h_data, data, size_bytes(), cudaMemcpyDeviceToHost);
    return {h_data, dim, n_vectors};
  }

  // Access vector i as a pointer: &data[i * dim]
  __host__ __device__ DATA_T* operator[](uint32_t i) {
    return &data[i * dim];
  }

  __host__ __device__ const DATA_T* operator[](uint32_t i) const {
    return &data[i * dim];
  }
};

// Returns a vector_view pointing to pinned host memory.
// The caller is responsible for freeing the data via cudaFreeHost.
template <typename DATA_T>
__host__ vector_view<DATA_T> load_vectors_from_file(std::string filename) {
  std::ifstream file(filename, std::ios::in | std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Error opening file: " << filename << std::endl;
    return {nullptr, 0};
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

  DATA_T* host_data;
  cudaMallocHost(&host_data, sizeof(DATA_T) * n_data_points * dim);

  if (file.read(reinterpret_cast<char*>(host_data), data_size)) {
    std::cout << "Successfully read " << n_data_points
              << " vectors (dim=" << dim << ") from " << filename << "\n";
  } else {
    std::cerr << "Error reading file\n";
    cudaFreeHost(host_data);
    return {nullptr, 0, 0};
  }

  return {host_data, dim, n_data_points};
}

}