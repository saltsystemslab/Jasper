#pragma once

#include <algorithm>
#include <fstream>
#include <sstream>
#include <vector>

namespace jasper {

template <typename DATA_T>
struct __align__(16) vector_view {
  DATA_T *data;
  uint32_t dim;
  uint32_t padded_dim;     // padded dim, guarantees 16-byte aligned rows
  uint32_t n_vectors;
  bool on_host=false;

  static constexpr uint32_t alignment = 16 / sizeof(DATA_T);

  __host__ __device__ static uint32_t pad(uint32_t d) {
    return (d + alignment - 1) / alignment * alignment;
  }

  __host__ __device__ vector_view() : data(nullptr), dim(0), padded_dim(0), n_vectors(0), on_host(false) {}

  __host__ __device__ vector_view(DATA_T* data, uint32_t dim, uint32_t n_vectors, bool on_host)
      : data(data), dim(dim), padded_dim(pad(dim)), n_vectors(n_vectors), on_host(on_host) {}

  // allocating an empty vector view with dim + n_vectors
  static vector_view allocate(uint32_t dim, uint32_t n_vectors, bool on_host=false){
    vector_view view;
    view.dim = dim;
    view.padded_dim = pad(dim);
    view.n_vectors = n_vectors;
    view.on_host = on_host;
    size_t bytes = view.size_bytes();
    cudaError_t err;
    if (on_host) {
      err = cudaMallocHost(&view.data, bytes);
    } else {
      err = cudaMalloc(&view.data, bytes);
    }
    if (err != cudaSuccess)
      throw std::runtime_error(std::string("vector_view::allocate failed: ") + cudaGetErrorString(err));
    return view;
  }

  // deallocate
  void deallocate() {
    if (on_host) {
      cudaFreeHost(data);
    } else {
      cudaFree(data);
    }
  }

  // Private ctor that preserves an existing padded_dim (used by subview)
  __host__ __device__ vector_view(DATA_T* data, uint32_t dim, uint32_t padded_dim, uint32_t n_vectors, bool on_host)
      : data(data), dim(dim), padded_dim(padded_dim), n_vectors(n_vectors), on_host(on_host) {}

  __host__ size_t size_bytes() const {
    return sizeof(DATA_T) * static_cast<size_t>(n_vectors) * padded_dim;
  }

  // Copy this view's data to device, return a new view pointing to device memory.
  // Caller is responsible for freeing the returned view's data via cudaFree.
  __host__ vector_view<DATA_T> to_device() const {
    DATA_T* d_data = nullptr;
    size_t bytes = size_bytes();

    cudaError_t err = cudaMalloc(&d_data, bytes);
    if (err != cudaSuccess) {
      const char* err_str = cudaGetErrorString(err);
      cudaGetLastError();

      size_t free_mem = 0, total_mem = 0;
      cudaMemGetInfo(&free_mem, &total_mem);

      throw std::runtime_error(
        std::string("vector_view::to_device cudaMalloc failed: ") + err_str +
        " (requested: " + std::to_string(bytes / (1 << 20)) + " MB, "
        "GPU free: " + std::to_string(free_mem / (1 << 20)) + " MB)");
    }

    err = cudaMemcpy(d_data, data, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      const char* err_str = cudaGetErrorString(err);
      cudaGetLastError();
      cudaFree(d_data);
      throw std::runtime_error(
        std::string("vector_view::to_device cudaMemcpy failed: ") + err_str);
    }

    return {d_data, dim, padded_dim, n_vectors, false};
  }

  // Copy this view's data into an already-allocated target view.
  // Target must have matching dim/padded_dim and capacity >= n_vectors.
  // Memcpy direction is inferred from each side's on_host flag.
  __host__ void copy_to(vector_view<DATA_T>& target, cudaStream_t stream = 0) const {
    if (target.dim != dim || target.padded_dim != padded_dim) {
      throw std::runtime_error("vector_view::copy_to: dim/padded_dim mismatch");
    }
    if (target.n_vectors < n_vectors) {
      throw std::runtime_error("vector_view::copy_to: target capacity too small");
    }

    cudaMemcpyKind kind;
    if      ( on_host &&  target.on_host) kind = cudaMemcpyHostToHost;
    else if ( on_host && !target.on_host) kind = cudaMemcpyHostToDevice;
    else if (!on_host &&  target.on_host) kind = cudaMemcpyDeviceToHost;
    else                                  kind = cudaMemcpyDeviceToDevice;

    size_t bytes = sizeof(DATA_T) * static_cast<size_t>(n_vectors) * padded_dim;
    cudaMemcpyAsync(target.data, data, bytes, kind, stream);
  }

  // Copy this view's data to pinned host memory, return a new view.
  // Caller is responsible for freeing the returned view's data via cudaFreeHost.
  __host__ vector_view<DATA_T> to_host() const {
    DATA_T* h_data = nullptr;
    cudaMallocHost(&h_data, size_bytes());
    cudaMemcpy(h_data, data, size_bytes(), cudaMemcpyDeviceToHost);
    return {h_data, dim, padded_dim, n_vectors, true};
  }

  // Access vector i as a pointer: &data[i * padded_dim]
  __host__ __device__ DATA_T* operator[](uint32_t i) {
    return &data[static_cast<size_t>(i) * padded_dim];
  }

  __host__ __device__ const DATA_T* operator[](uint32_t i) const {
    return &data[static_cast<size_t>(i) * padded_dim];
  }

  // Return a read-only, non-owning view over a contiguous subset of vectors.
  // Starts at vector index `offset`, spanning `count` vectors.
  __host__ __device__ vector_view<DATA_T> subview(uint32_t offset, uint32_t count) const {
    assert(offset + count <= n_vectors);
    return {data + static_cast<size_t>(offset) * padded_dim, dim, padded_dim, count, on_host};
  }
};

// Returns a vector_view pointing to pinned host memory.
// The caller is responsible for freeing the data via cudaFreeHost.
template <typename DATA_T>
__host__ vector_view<DATA_T> load_vectors_from_file(std::string filename) {
  std::ifstream file(filename, std::ios::in | std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Error opening file: " << filename << std::endl;
    return {nullptr, 0, 0, false};
  }

  file.seekg(0, std::ios::end);
  std::streamsize file_size = file.tellg();
  file.seekg(0, std::ios::beg);

  int32_t n_data_points;
  int32_t n_dimensions;
  file.read(reinterpret_cast<char*>(&n_data_points), sizeof(int32_t));
  file.read(reinterpret_cast<char*>(&n_dimensions), sizeof(int32_t));

  if (n_dimensions < 0 || n_data_points < 0) {
    throw std::runtime_error("negative dimension or count read from file");
  }

  std::streamsize data_size = file_size - 2 * sizeof(int32_t);

  std::cout << "Read " << data_size << " bytes of data\n";
  std::cout << "n_data_points=" << n_data_points
            << " dim=" << n_dimensions << "\n";

  std::streamsize expected = static_cast<std::streamsize>(sizeof(DATA_T))
                           * n_data_points * n_dimensions;
  if (expected != data_size) {
    std::cerr << "Size mismatch: expected " << expected
              << " bytes but file has " << data_size << " bytes\n";
    return {nullptr, 0, 0, false};
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
      return {nullptr, 0, 0, false};
    }
  } else {
    // Read each vector into its padded row
    size_t row_bytes = sizeof(DATA_T) * dim;
    for (int32_t i = 0; i < n_data_points; i++) {
      if (!file.read(reinterpret_cast<char*>(host_data + static_cast<size_t>(i) * padded_dim), row_bytes)) {
        std::cerr << "Error reading vector " << i << "\n";
        cudaFreeHost(host_data);
        return {nullptr, 0, 0, false};
      }
    }
  }

  std::cout << "Successfully read " << n_data_points
            << " vectors (dim=" << dim
            << ", padded_dim=" << padded_dim
            << ") from " << filename << "\n";

  return {
    host_data,
    static_cast<uint32_t>(dim),
    static_cast<uint32_t>(n_data_points),
    true
  };
}

// Read SrcT-encoded vectors from file and cast each element to DATA_T.
// Output lives in pinned host memory with DATA_T-aligned padded rows.
// Caller is responsible for freeing the returned view's data via cudaFreeHost.
template <typename DATA_T, typename SrcT>
__host__ vector_view<DATA_T> load_vectors_from_file_cast(std::string filename) {
  std::ifstream file(filename, std::ios::in | std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Error opening file: " << filename << std::endl;
    return {nullptr, 0, 0, false};
  }

  file.seekg(0, std::ios::end);
  std::streamsize file_size = file.tellg();
  file.seekg(0, std::ios::beg);

  int32_t n_data_points;
  int32_t n_dimensions;
  file.read(reinterpret_cast<char*>(&n_data_points), sizeof(int32_t));
  file.read(reinterpret_cast<char*>(&n_dimensions), sizeof(int32_t));

  if (n_dimensions < 0 || n_data_points < 0) {
    throw std::runtime_error("negative dimension or count read from file");
  }

  std::streamsize data_size = file_size - 2 * sizeof(int32_t);
  std::streamsize expected = static_cast<std::streamsize>(sizeof(SrcT))
                           * n_data_points * n_dimensions;
  if (expected != data_size) {
    std::cerr << "Size mismatch: expected " << expected
              << " bytes but file has " << data_size << " bytes\n";
    return {nullptr, 0, 0, false};
  }

  uint32_t dim = static_cast<uint32_t>(n_dimensions);
  uint32_t padded_dim = vector_view<DATA_T>::pad(dim);

  size_t padded_bytes = sizeof(DATA_T) * static_cast<size_t>(n_data_points) * padded_dim;
  DATA_T* host_data;
  cudaMallocHost(&host_data, padded_bytes);
  memset(host_data, 0, padded_bytes);

  // Read in large chunks rather than one syscall per vector, then convert.
  // Going through float as intermediate so e.g. uint8_t -> __half works
  // without relying on direct conversion constructors.
  constexpr size_t kTargetChunkBytes = size_t{16} << 20;  // ~16 MiB raw per read
  size_t rows_per_chunk = std::max<size_t>(1, kTargetChunkBytes / (sizeof(SrcT) * dim));
  std::vector<SrcT> buf(rows_per_chunk * dim);

  for (size_t base = 0; base < static_cast<size_t>(n_data_points); base += rows_per_chunk) {
    size_t chunk_rows = std::min(rows_per_chunk, static_cast<size_t>(n_data_points) - base);
    if (!file.read(reinterpret_cast<char*>(buf.data()),
                   sizeof(SrcT) * chunk_rows * dim)) {
      std::cerr << "Error reading vectors at " << base << "\n";
      cudaFreeHost(host_data);
      return {nullptr, 0, 0, false};
    }
    for (size_t r = 0; r < chunk_rows; r++) {
      const SrcT* src = buf.data() + r * dim;
      DATA_T* dst = host_data + (base + r) * padded_dim;
      for (uint32_t j = 0; j < dim; j++) {
        dst[j] = static_cast<DATA_T>(static_cast<float>(src[j]));
      }
    }
    size_t done = base + chunk_rows;
    std::cout << "\rLoading " << filename << ": " << done << "/" << n_data_points
              << " (" << (100 * done / static_cast<size_t>(n_data_points)) << "%)"
              << std::flush;
  }
  std::cout << "\n";

  std::cout << "Loaded " << n_data_points
            << " vectors (dim=" << dim
            << ", padded_dim=" << padded_dim
            << ", src=" << sizeof(SrcT) << "B"
            << ", dst=" << sizeof(DATA_T) << "B)"
            << " from " << filename << "\n";

  return {
    host_data,
    static_cast<uint32_t>(dim),
    static_cast<uint32_t>(n_data_points),
    true
  };
}

}