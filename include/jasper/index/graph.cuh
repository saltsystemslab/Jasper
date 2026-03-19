#pragma once

#include <cstdint>
#include <vector>

#include "jasper/distance/distance.cuh"

#include "assert.h"
#include "stdio.h"

namespace jasper {
  
template <typename INDEX_T, uint32_t N_NEIGHBORS>
struct edge_list {
  INDEX_T edges[N_NEIGHBORS];

  __host__ __device__ INDEX_T& operator[](uint32_t i) { return edges[i]; }
  __host__ __device__ const INDEX_T& operator[](uint32_t i) const { return edges[i]; }
  __host__ __device__ constexpr uint32_t size() const { return N_NEIGHBORS; }

  void print() const {
    std::cout << "Edges: ";
    for (uint32_t i = 0; i < N_NEIGHBORS; ++i) {
      std::cout << edges[i] << " ";
    }
    std::cout << std::endl;
  }
};

template <typename INDEX_T,
          uint8_t N_NEIGHBORS,
          typename DATA_T,
          typename DISTANCE_T,
          distance_func DIST_FUNC>
struct graph_config {
  using index_t = INDEX_T;
  using data_t = DATA_T;
  using distance_t = DISTANCE_T;
  using edge_list_t = edge_list<INDEX_T, N_NEIGHBORS>;
  using vector_view_t = vector_view<DATA_T>;
  
  static constexpr uint8_t n_neighbors = N_NEIGHBORS;
  static constexpr distance_func dist_func = DIST_FUNC;
};

template <typename graph_cfg>
struct graph {
  using index_t     = typename graph_cfg::index_t;
  using edge_list_t = typename graph_cfg::edge_list_t;
  using vector_view_t    = typename graph_cfg::vector_view_t;

  uint32_t dim;
  index_t n_vectors;
  index_t medoid;

  edge_list_t *edges;
  uint8_t *edge_counts;

  // flat vectors DATA_T[n_vectors * dim]
  vector_view_t vectors;
};
template <typename graph_config>
__host__ graph<graph_config> load_graph_from_file(std::string input_fname,
                                                  uint32_t dim,
                                                  bool on_host = false) {
  std::ifstream inputFile(input_fname, std::ios::binary);
  if (!inputFile.is_open()) {
    std::cerr << "Failed to open " << input_fname << " for graph load\n";
    exit(EXIT_FAILURE);
  }

  using data_t      = typename graph_config::data_t;
  using edge_list_t = typename graph_config::edge_list_t;

  graph<graph_config> g;
  g.dim = dim;

  uint64_t total_file_size;
  uint64_t big_n_vectors;
  uint64_t medoid_as_uint64;
  uint64_t bytes_per_node;

  // Read the metadata
  inputFile.read(reinterpret_cast<char*>(&total_file_size), sizeof(uint64_t));
  inputFile.read(reinterpret_cast<char*>(&big_n_vectors),   sizeof(uint64_t));
  inputFile.read(reinterpret_cast<char*>(&medoid_as_uint64), sizeof(uint64_t));
  inputFile.read(reinterpret_cast<char*>(&bytes_per_node),   sizeof(uint64_t));

  g.medoid    = static_cast<typename graph_config::index_t>(medoid_as_uint64);
  g.n_vectors = static_cast<typename graph_config::index_t>(big_n_vectors);

  std::cout << "total_file_size: " << total_file_size << "\n";
  std::cout << "big_n_vectors: "   << big_n_vectors << "\n";
  std::cout << "medoid: "          << g.medoid << "\n";
  std::cout << "bytes_per_node: "  << bytes_per_node << "\n";
  std::cout << "dim: "             << dim << "\n";

  uint32_t padded_dim = vector_view<data_t>::pad(dim);

  // Allocate pinned host memory (padded layout for vectors)
  uint8_t*     host_edge_counts;
  edge_list_t* host_edges;
  data_t*      host_vectors;

  cudaMallocHost(&host_edge_counts, sizeof(uint8_t)     * big_n_vectors);
  cudaMallocHost(&host_edges,       sizeof(edge_list_t)  * big_n_vectors);
  cudaMallocHost(&host_vectors,     sizeof(data_t)       * big_n_vectors * padded_dim);
  memset(host_vectors, 0,           sizeof(data_t)       * big_n_vectors * padded_dim);

  // Read per-node data (disk is packed at dim, memory is strided at padded_dim)
  for (uint64_t i = 0; i < big_n_vectors; i++) {
    inputFile.read(reinterpret_cast<char*>(&host_vectors[i * padded_dim]),
                   sizeof(data_t) * dim);
    uint8_t n_neighbors;
    inputFile.read(reinterpret_cast<char*>(&n_neighbors), sizeof(uint8_t));
    host_edge_counts[i] = n_neighbors;
    inputFile.read(reinterpret_cast<char*>(&host_edges[i]), sizeof(edge_list_t));
  }

  if (on_host) {
    g.edge_counts = host_edge_counts;
    g.edges       = host_edges;
    g.vectors     = {host_vectors, dim, static_cast<uint32_t>(big_n_vectors)};
  } else {
    // Allocate device memory (padded layout)
    data_t* d_vectors;
    cudaMalloc(&g.edge_counts, sizeof(uint8_t)     * big_n_vectors);
    cudaMalloc(&g.edges,       sizeof(edge_list_t)  * big_n_vectors);
    cudaMalloc(&d_vectors,     sizeof(data_t)       * big_n_vectors * padded_dim);

    cudaMemcpy(g.edge_counts, host_edge_counts,
               sizeof(uint8_t) * big_n_vectors, cudaMemcpyHostToDevice);
    cudaMemcpy(g.edges, host_edges,
               sizeof(edge_list_t) * big_n_vectors, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vectors, host_vectors,
               sizeof(data_t) * big_n_vectors * padded_dim, cudaMemcpyHostToDevice);

    g.vectors = {d_vectors, dim, static_cast<uint32_t>(big_n_vectors)};

    cudaFreeHost(host_edge_counts);
    cudaFreeHost(host_edges);
    cudaFreeHost(host_vectors);
  }

  cudaDeviceSynchronize();
  return g;
}

template <typename graph_config>
__host__ void save_graph_to_file(const graph<graph_config>& g,
                                 std::string output_fname,
                                 bool on_host = false) {
  using data_t      = typename graph_config::data_t;
  using edge_list_t = typename graph_config::edge_list_t;

  std::ofstream outFile(output_fname, std::ios::binary);
  if (!outFile.is_open()) {
    std::cerr << "Failed to open " << output_fname << " for graph save\n";
    exit(EXIT_FAILURE);
  }

  uint64_t big_n_vectors    = static_cast<uint64_t>(g.n_vectors);
  uint64_t medoid_as_uint64 = static_cast<uint64_t>(g.medoid);
  uint32_t dim              = g.vectors.dim;
  uint32_t padded_dim       = g.vectors.padded_dim;
  // File format uses packed dim, not padded
  uint64_t bytes_per_node   = sizeof(data_t) * dim + sizeof(uint8_t) + sizeof(edge_list_t);
  uint64_t total_file_size  = 4 * sizeof(uint64_t) + big_n_vectors * bytes_per_node;

  // Write metadata
  outFile.write(reinterpret_cast<const char*>(&total_file_size),  sizeof(uint64_t));
  outFile.write(reinterpret_cast<const char*>(&big_n_vectors),    sizeof(uint64_t));
  outFile.write(reinterpret_cast<const char*>(&medoid_as_uint64), sizeof(uint64_t));
  outFile.write(reinterpret_cast<const char*>(&bytes_per_node),   sizeof(uint64_t));

  // If data is on device, copy to host first
  uint8_t*     host_edge_counts = nullptr;
  edge_list_t* host_edges       = nullptr;
  data_t*      host_vectors     = nullptr;

  if (on_host) {
    host_edge_counts = g.edge_counts;
    host_edges       = g.edges;
    host_vectors     = g.vectors.data;
  } else {
    cudaMallocHost(&host_edge_counts, sizeof(uint8_t)     * big_n_vectors);
    cudaMallocHost(&host_edges,       sizeof(edge_list_t)  * big_n_vectors);
    cudaMallocHost(&host_vectors,     sizeof(data_t)       * big_n_vectors * padded_dim);

    cudaMemcpy(host_edge_counts, g.edge_counts,
               sizeof(uint8_t) * big_n_vectors, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_edges, g.edges,
               sizeof(edge_list_t) * big_n_vectors, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_vectors, g.vectors.data,
               sizeof(data_t) * big_n_vectors * padded_dim, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
  }

  // Write per-node data (strip padding, write only dim elements per vector)
  for (uint64_t i = 0; i < big_n_vectors; i++) {
    outFile.write(reinterpret_cast<const char*>(&host_vectors[i * padded_dim]),
                  sizeof(data_t) * dim);
    outFile.write(reinterpret_cast<const char*>(&host_edge_counts[i]),
                  sizeof(uint8_t));
    outFile.write(reinterpret_cast<const char*>(&host_edges[i]),
                  sizeof(edge_list_t));
  }

  outFile.close();

  if (!on_host) {
    cudaFreeHost(host_edge_counts);
    cudaFreeHost(host_edges);
    cudaFreeHost(host_vectors);
  }

  std::cout << "Saved graph to " << output_fname << "\n";
  std::cout << "  n_vectors: " << big_n_vectors << "\n";
  std::cout << "  medoid: " << g.medoid << "\n";
  std::cout << "  dim: " << dim << "\n";
  std::cout << "  total_file_size: " << total_file_size << "\n";
}

}