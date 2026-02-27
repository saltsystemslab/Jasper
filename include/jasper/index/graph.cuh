#pragma once

#include <cstdint>
#include <vector>

#include "assert.h"
#include "stdio.h"

namespace jasper {
  
template <typename INDEX_T, uint32_t N_NEIGHBORS>
struct edge_list {
  INDEX_T edges[R];

  __host__ __device__ INDEX_T& operator[](uint32_t i) { return data[i]; }
  __host__ __device__ const INDEX_T& operator[](uint32_t i) const { return data[i]; }
  __host__ __device__ constexpr uint32_t size() const { return N_NEIGHBORS; }

  void print() const {
    std::cout << "Edges: ";
    for (uint32_t i = 0; i < R; ++i) {
      std::cout << edges[i] << " ";
    }
    std::cout << std::endl;
  }
};

template <typename INDEX_T,
          uint8_t N_NEIGHBORS,
          typename DATA_T,
          uint16_t DATA_DIM,
          typename DISTANCE_T,
          template <typename, typename, uint, uint> class DISTANCE_FUNC_T>
struct graph_config {
  using index_t = INDEX_T;
  using data_t = DATA_T;
  using distance_t = DISTANCE_T;
  using edge_list_t = edge_list<INDEX_T, N_NEIGHBORS>;
  using vector_t = vector<DATA_T, DATA_DIM>;
  using distance_func_t = DISTANCE_FUNC_T<DATA_T, DATA_T, DATA_DIM, 4>;
  
  static constexpr uint8_t n_neighbors = N_NEIGHBORS;
  static constexpr uint16_t data_dim = DATA_DIM;
};

template <typename graph_cfg>
struct graph {
  using index_t     = typename graph_cfg::index_t;
  using edge_list_t = typename graph_cfg::edge_list_t;
  using vector_t    = typename graph_cfg::vector_t;

  index_t n_vectors;
  index_t medoid;

  edge_list_t *edges;
  uint8_t *edge_counts;

  // TODO: i will remove vectors from graph once i have the chance
  //       or it should be managed more carefully.
  vector_t *vectors;
};

template <typename graph_config>
__host__ graph<graph_config> load_graph_from_file(std::string input_fname, bool on_host=false) {
  std::string vector_filename = input_fname;

  std::ifstream inputFile(vector_filename, std::ios::binary);
  if (!inputFile.is_open()) {
    std::cerr << "Failed to open " << vector_filename << " for graph load\n";
    exit(EXIT_FAILURE);
  }

  // graph object
  graph<graph_config> g;
  using edge_list_t = typename graph<graph_config>::edge_list_t;
  using vector_t = typename graph<graph_config>::vector_t;

  uint64_t total_file_size;
  uint64_t big_n_vectors;
  uint64_t medoid_as_uint64;
  uint64_t bytes_per_node;

  // Read the metadata
  inputFile.read(reinterpret_cast<char *>(&total_file_size), sizeof(uint64_t));
  inputFile.read(reinterpret_cast<char *>(&big_n_vectors), sizeof(uint64_t));
  inputFile.read(reinterpret_cast<char *>(&medoid_as_uint64), sizeof(uint64_t));
  inputFile.read(reinterpret_cast<char *>(&bytes_per_node), sizeof(uint64_t));

  g.medoid = static_cast<uint32_t>(medoid_as_uint64);
  g.n_vectors = big_n_vectors;

  std::cout << "total_file_size: " << total_file_size << "\n";
  std::cout << "big_n_vectors: " << big_n_vectors << "\n";
  std::cout << "medoid: " << g.medoid << "\n";
  std::cout << "bytes_per_node: " << bytes_per_node << "\n";

  uint8_t *host_edge_counts;
  edge_list_t *host_edges;
  vector_t *host_vectors;

  cudaMallocHost(&host_edge_counts, sizeof(uint8_t) * big_n_vectors);
  cudaMallocHost(&host_edges, sizeof(edge_list_t) * big_n_vectors);
  cudaMallocHost(&host_vectors, sizeof(vector_t) * big_n_vectors);


  for (uint i = 0; i < big_n_vectors; i++) {
    // Read vector
    inputFile.read(reinterpret_cast<char *>(&host_vectors[i]), sizeof(vector_t));
    // Read neighbor count
    uint8_t n_neighbors;
    inputFile.read(reinterpret_cast<char *>(&n_neighbors), sizeof(uint8_t));
    host_edge_counts[i] = n_neighbors;
    // Read neighbor list
    inputFile.read(reinterpret_cast<char *>(&host_edges[i]), sizeof(edge_list_t));
  }

  if (on_host) {
    g.edge_counts = host_edge_counts;
    g.edges = host_edges;
    g.vectors = host_vectors;
  } else {
    cudaMalloc(&g.edge_counts, sizeof(uint8_t) * big_n_vectors);
    cudaMalloc(&g.edges, sizeof(edge_list_t) * big_n_vectors);
    cudaMalloc(&g.vectors, sizeof(vector_t) * big_n_vectors);

    cudaMemcpy(g.edge_counts, host_edge_counts, sizeof(uint8_t) * big_n_vectors,
               cudaMemcpyHostToDevice);
    cudaMemcpy(g.edges, host_edges, sizeof(edge_list_t) * big_n_vectors,
               cudaMemcpyHostToDevice);
    cudaMemcpy(g.vectors, host_vectors, sizeof(vector_t) * big_n_vectors,
               cudaMemcpyHostToDevice);
    cudaError_t err = cudaFreeHost(host_edge_counts);
    if (err != cudaSuccess) {
      printf("cudaFree error: %s\n", cudaGetErrorString(err));
    }
    err = cudaFreeHost(host_edges);
    if (err != cudaSuccess) {
      printf("cudaFree error: %s\n", cudaGetErrorString(err));
    }
    err = cudaFreeHost(host_vectors);
    if (err != cudaSuccess) {
      printf("cudaFree error: %s\n", cudaGetErrorString(err));
    }
  }

  cudaDeviceSynchronize();
  return g;
}

template <typename graph_config>
__host__ void save_graph_to_file(const graph<graph_config>& g,
                                 std::string output_fname,
                                 bool on_host = false) {
  using edge_list_t = typename graph<graph_config>::edge_list_t;
  using vector_t    = typename graph<graph_config>::vector_t;

  std::ofstream outFile(output_fname, std::ios::binary);
  if (!outFile.is_open()) {
    std::cerr << "Failed to open " << output_fname << " for graph save\n";
    exit(EXIT_FAILURE);
  }

  uint64_t big_n_vectors    = static_cast<uint64_t>(g.n_vectors);
  uint64_t medoid_as_uint64 = static_cast<uint64_t>(g.medoid);
  uint64_t bytes_per_node   = sizeof(vector_t) + sizeof(uint8_t) + sizeof(edge_list_t);
  uint64_t total_file_size  = 4 * sizeof(uint64_t) + big_n_vectors * bytes_per_node;

  // Write metadata
  outFile.write(reinterpret_cast<const char*>(&total_file_size),  sizeof(uint64_t));
  outFile.write(reinterpret_cast<const char*>(&big_n_vectors),    sizeof(uint64_t));
  outFile.write(reinterpret_cast<const char*>(&medoid_as_uint64), sizeof(uint64_t));
  outFile.write(reinterpret_cast<const char*>(&bytes_per_node),   sizeof(uint64_t));

  // If data is on device, copy to host first
  uint8_t*     host_edge_counts = nullptr;
  edge_list_t* host_edges       = nullptr;
  vector_t*    host_vectors     = nullptr;

  if (on_host) {
    host_edge_counts = g.edge_counts;
    host_edges       = g.edges;
    host_vectors     = g.vectors;
  } else {
    cudaMallocHost(&host_edge_counts, sizeof(uint8_t)     * big_n_vectors);
    cudaMallocHost(&host_edges,       sizeof(edge_list_t)  * big_n_vectors);
    cudaMallocHost(&host_vectors,     sizeof(vector_t)     * big_n_vectors);

    cudaMemcpy(host_edge_counts, g.edge_counts,
               sizeof(uint8_t) * big_n_vectors, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_edges, g.edges,
               sizeof(edge_list_t) * big_n_vectors, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_vectors, g.vectors,
               sizeof(vector_t) * big_n_vectors, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
  }

  // Write per-node data (same order as load: vector, edge_count, edge_list)
  for (uint64_t i = 0; i < big_n_vectors; i++) {
    outFile.write(reinterpret_cast<const char*>(&host_vectors[i]),     sizeof(vector_t));
    outFile.write(reinterpret_cast<const char*>(&host_edge_counts[i]), sizeof(uint8_t));
    outFile.write(reinterpret_cast<const char*>(&host_edges[i]),       sizeof(edge_list_t));
  }

  outFile.close();

  // Free temp host memory if we allocated it
  if (!on_host) {
    cudaFreeHost(host_edge_counts);
    cudaFreeHost(host_edges);
    cudaFreeHost(host_vectors);
  }

  std::cout << "Saved graph to " << output_fname << "\n";
  std::cout << "  n_vectors: " << big_n_vectors << "\n";
  std::cout << "  medoid: " << g.medoid << "\n";
  std::cout << "  bytes_per_node: " << bytes_per_node << "\n";
  std::cout << "  total_file_size: " << total_file_size << "\n";
}

}