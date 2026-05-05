#pragma once

#include <cstdint>
#include <vector>

#include <thrust/device_vector.h>

#include "jasper/distance/distance.cuh"

#include "assert.h"
#include "stdio.h"

namespace jasper {
  
template <typename INDEX_T, uint32_t N_NEIGHBORS>
struct edge_list {
  INDEX_T edges[N_NEIGHBORS];
  float dist[N_NEIGHBORS];

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
  static constexpr index_t vectors_per_segment = 1u << 20;
};

template <typename graph_cfg>
struct graph_segment {
  using index_t     = typename graph_cfg::index_t;
  using edge_list_t = typename graph_cfg::edge_list_t;
  using vector_view_t    = typename graph_cfg::vector_view_t;

  static constexpr uint32_t max_vectors = graph_cfg::vectors_per_segment;
  static constexpr uint32_t n_neighbors = graph_cfg::n_neighbors;

  index_t n_vectors;

  // storage (preallocated to graph_cfg::vectors_per_segement)
  edge_list_t *edges;
  uint8_t *edge_counts;
  vector_view_t vectors;

  // if on_host is true, the data is allocated on cpu pinned memory
  bool on_host = false;

  static graph_segment allocate(uint32_t dim, bool on_host=false) {
    graph_segment seg{};
    seg.n_vectors = 0;
    seg.on_host = on_host;

    if (on_host) {
        cudaMallocHost(&seg.edges, max_vectors * sizeof(edge_list_t));
        cudaMallocHost(&seg.edge_counts, max_vectors * sizeof(uint8_t));
        std::memset(seg.edge_counts, 0, max_vectors * sizeof(uint8_t));
    } else {
        cudaMalloc(&seg.edges, max_vectors * sizeof(edge_list_t));
        cudaMalloc(&seg.edge_counts, max_vectors * sizeof(uint8_t));
        cudaMemset(seg.edge_counts, 0, max_vectors * sizeof(uint8_t));
    }

    seg.vectors = vector_view_t::allocate(dim, max_vectors, on_host);

    return seg;
  }

  void deallocate() {
    if (on_host) {
        cudaFreeHost(edges);
        cudaFreeHost(edge_counts);
    } else {
        cudaFree(edges);
        cudaFree(edge_counts);
    }
    vectors.deallocate();
    edges = nullptr;
    edge_counts = nullptr;
    n_vectors = 0;
  }

  // Migrate this segment's storage between host pinned memory and device memory.
  // No-op if already on the target side.
  void move_to(bool target_on_host, cudaStream_t stream = 0) {
    if (on_host == target_on_host) return;

    using data_t = typename graph_cfg::data_t;

    const cudaMemcpyKind copy_kind =
        on_host ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost;
    const uint32_t padded_dim = vector_view_t::pad(vectors.dim);

    // Allocate new buffers on the target side (same capacity as allocate()).
    edge_list_t* new_edges       = nullptr;
    uint8_t*     new_edge_counts = nullptr;

    if (target_on_host) {
      cudaMallocHost(&new_edges,       max_vectors * sizeof(edge_list_t));
      cudaMallocHost(&new_edge_counts, max_vectors * sizeof(uint8_t));
      std::memset(new_edge_counts, 0,  max_vectors * sizeof(uint8_t));
    } else {
      cudaMalloc(&new_edges,       max_vectors * sizeof(edge_list_t));
      cudaMalloc(&new_edge_counts, max_vectors * sizeof(uint8_t));
      cudaMemsetAsync(new_edge_counts, 0,
                      max_vectors * sizeof(uint8_t), stream);
    }
    vector_view_t new_vectors =
        vector_view_t::allocate(vectors.dim, max_vectors, target_on_host);

    // Copy only the valid prefix.
    if (n_vectors > 0) {
      cudaMemcpyAsync(new_edges, edges,
                      static_cast<size_t>(n_vectors) * sizeof(edge_list_t),
                      copy_kind, stream);
      cudaMemcpyAsync(new_edge_counts, edge_counts,
                      static_cast<size_t>(n_vectors) * sizeof(uint8_t),
                      copy_kind, stream);
      cudaMemcpyAsync(new_vectors.data, vectors.data,
                      static_cast<size_t>(n_vectors) * padded_dim * sizeof(data_t),
                      copy_kind, stream);
    }
    cudaStreamSynchronize(stream);  // must complete before freeing old buffers

    // Free the old buffers using the *previous* on_host flag.
    if (on_host) {
      cudaFreeHost(edges);
      cudaFreeHost(edge_counts);
    } else {
      cudaFree(edges);
      cudaFree(edge_counts);
    }
    vectors.deallocate();

    // Install new buffers.
    edges       = new_edges;
    edge_counts = new_edge_counts;
    vectors     = new_vectors;
    on_host     = target_on_host;
  }

  // Device side read.
  // Note: not concurrent safe.
  //       our construction algorithm handles accesses externally.

  // TODO: Currently, to write the neighbor list and vectors, we need to get their 
  //       mutable references. I don't really like that. --Zikun

  __device__ __forceinline__
  bool is_valid_local_idx(uint32_t local_idx) const {
    return local_idx < static_cast<uint32_t>(n_vectors);
  }

  __device__ __forceinline__
  uint8_t get_edge_count(uint32_t local_idx) const {
    assert(local_idx < static_cast<uint32_t>(n_vectors));
    return edge_counts[local_idx];
  }

  // mutable reference
  __device__ __forceinline__
  const edge_list_t& get_neighbor_list(uint32_t local_idx) {
    assert(local_idx < static_cast<uint32_t>(n_vectors));
    return edges[local_idx];
  }

  __device__ __forceinline__
  index_t get_neighbor(uint32_t local_idx, uint8_t neighbor_idx) const {
    assert(local_idx < static_cast<uint32_t>(n_vectors));
    assert(neighbor_idx < edge_counts[local_idx]);
    return edges[local_idx][neighbor_idx];
  }

  __device__ __forceinline__
  float get_neighbor_dist(uint32_t local_idx, uint8_t neighbor_idx) const {
    assert(local_idx < static_cast<uint32_t>(n_vectors));
    assert(neighbor_idx < edge_counts[local_idx]);
    return edges[local_idx].dist[neighbor_idx];
  }

  // mutable reference
  __device__ __forceinline__
  auto get_vector(uint32_t local_idx) {
    assert(local_idx < static_cast<uint32_t>(n_vectors));
    return vectors[local_idx];
  }

  // Device side write
  // Note: not concurrent safe.
  //       our construction algorithm handles accesses externally.

  __device__ __forceinline__
  void set_edge_count(uint32_t local_idx, uint8_t count) {
    assert(local_idx < static_cast<uint32_t>(n_vectors));
    assert(count <= n_neighbors);
    edge_counts[local_idx] = count;
  }

  __device__ __forceinline__
  void set_neighbor(uint32_t local_idx, uint8_t neighbor_idx, index_t neighbor) {
    assert(local_idx < static_cast<uint32_t>(n_vectors));
    assert(neighbor_idx < n_neighbors);
    edges[local_idx][neighbor_idx] = neighbor;
  }

  __device__ __forceinline__
  void set_neighbor_dist(uint32_t local_idx, uint8_t neighbor_idx, float dist_val) {
    assert(local_idx < static_cast<uint32_t>(n_vectors));
    assert(neighbor_idx < n_neighbors);
    edges[local_idx].dist[neighbor_idx] = dist_val;
  }
};

template <typename graph_cfg>
struct graph {
  using index_t     = typename graph_cfg::index_t;
  using data_t = typename graph_cfg::data_t;
  using segment_t   = graph_segment<graph_cfg>;
  using edge_list_t = typename graph_cfg::edge_list_t;
  using vector_view_t    = typename graph_cfg::vector_view_t;

  static constexpr uint32_t vectors_per_segment = graph_cfg::vectors_per_segment;

  uint32_t dim;
  index_t n_vectors;
  uint32_t n_segments;
  index_t medoid;
  index_t global_offset = 0;
  bool on_host;

  thrust::device_vector<segment_t> segments;

  static graph allocate(uint32_t dim, index_t n_vector_slots, bool on_host) {
    graph g{};
    g.dim = dim;
    g.n_vectors = 0;
    g.medoid = 0;
    g.n_segments = (n_vector_slots+vectors_per_segment-1)/vectors_per_segment;
    g.on_host = on_host;

    std::vector<segment_t> h_segments(g.n_segments);
    for (uint32_t i = 0; i < g.n_segments; i++) {
      h_segments[i] = segment_t::allocate(dim, on_host);
    }
    g.segments = thrust::device_vector<segment_t>(h_segments.begin(), h_segments.end());

    return g;
  }

  static graph allocate_and_load(vector_view_t data, bool on_host) {
    uint32_t dim       = data.dim;
    uint32_t padded_dim = vector_view_t::pad(dim);
    index_t  n_vectors = static_cast<index_t>(data.n_vectors);

    // Allocate the graph with enough segment slots for all vectors
    graph g = allocate(dim, n_vectors, on_host);
    g.n_vectors = n_vectors;

    // Copy segments back to host so we can set per-segment metadata and issue memcpys
    std::vector<segment_t> h_segments(g.segments.begin(), g.segments.end());

    const cudaMemcpyKind copy_kind = [&]{
      if (data.on_host && on_host)   return cudaMemcpyHostToHost;
      if (data.on_host && !on_host)  return cudaMemcpyHostToDevice;
      if (!data.on_host && on_host)  return cudaMemcpyDeviceToHost;
      return cudaMemcpyDeviceToDevice;
    }();

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    for (uint32_t s = 0; s < g.n_segments; s++) {
      uint64_t seg_begin = static_cast<uint64_t>(s) * vectors_per_segment;
      uint32_t seg_count = static_cast<uint32_t>(
          std::min(static_cast<uint64_t>(vectors_per_segment),
                   static_cast<uint64_t>(n_vectors) - seg_begin));

      h_segments[s].n_vectors = static_cast<index_t>(seg_count);

      // Copy vector data for this segment (edge counts are already zeroed by allocate)
      cudaMemcpyAsync(h_segments[s].vectors.data,
                      data.data + seg_begin * padded_dim,
                      static_cast<size_t>(seg_count) * padded_dim * sizeof(data_t),
                      copy_kind, stream);
    }

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    // Upload updated segment structs (with correct n_vectors) back to device
    g.segments = thrust::device_vector<segment_t>(h_segments.begin(), h_segments.end());

    cudaDeviceSynchronize();
    return g;
  }

  // load the range of vectors with index [start, end) to a vector view
  // useful for graph construction
  __host__ void load_vectors_to_view(vector_view_t target, index_t start, index_t end) {
    if (target.n_vectors < end - start) {
      throw std::runtime_error("load_vectors_to_view: target.n_vectors < end - start");
    }

    uint32_t padded_dim = vector_view_t::pad(dim);

    std::vector<segment_t> h_segments(segments.begin(), segments.end());

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    index_t copied = 0;
    for (index_t idx = start; idx < end; ) {
      uint32_t seg_id    = segment_of(idx);
      uint32_t local_idx = local_of(idx);

      uint32_t avail_in_seg = static_cast<uint32_t>(h_segments[seg_id].n_vectors) - local_idx;
      uint32_t remaining    = static_cast<uint32_t>(end - idx);
      uint32_t count        = std::min(avail_in_seg, remaining);

      cudaMemcpyAsync(target.data + copied * padded_dim,
                      h_segments[seg_id].vectors.data + local_idx * padded_dim,
                      count * padded_dim * sizeof(data_t),
                      cudaMemcpyDeviceToDevice, stream);

      idx    += count;
      copied += count;
    }

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
  }

  void deallocate() {
    std::vector<segment_t> h_segments(segments.begin(), segments.end());
    for (auto &seg : h_segments) {
      seg.deallocate();
    }
    segments.clear();
    segments.shrink_to_fit();
    n_vectors  = 0;
    n_segments = 0;
  }

  __host__ __device__ __forceinline__
  static uint32_t segment_of(index_t global_idx) {
    return static_cast<uint32_t>(global_idx / vectors_per_segment);
  }

  __host__ __device__ __forceinline__
  static uint32_t local_of(index_t global_idx) {
    return static_cast<uint32_t>(global_idx % vectors_per_segment);
  }

  __host__ index_t get_padded_dim() const {
    return vector_view<data_t>::pad(dim);
  }

  // Migrate the entire graph (all segments + medoid metadata) between
  // host pinned memory and device memory. No-op if already on the target side.
  void move_to(bool target_on_host) {
    if (on_host == target_on_host) return;

    // Pull segment structs back so we can mutate their pointers.
    std::vector<segment_t> h_segments(segments.begin(), segments.end());

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    for (auto& seg : h_segments) {
      seg.move_to(target_on_host, stream);
    }
    cudaStreamDestroy(stream);

    // Re-upload segments with their new pointers.
    segments = thrust::device_vector<segment_t>(h_segments.begin(), h_segments.end());
    on_host  = target_on_host;

    cudaDeviceSynchronize();
  }

  // Rewrite every stored neighbor index from partition-local to global
  // (global = local + offset), then record global_offset on this graph.
  // Must be called after move_to(true) since it walks host-side edge arrays.
  __host__ void apply_global_offset(index_t offset) {
    global_offset = offset;
    if (offset == 0) return;

    if (!on_host) {
      throw std::runtime_error(
          "apply_global_offset requires graph on host — call move_to(true) first");
    }

    constexpr index_t INVALID = std::numeric_limits<index_t>::max();

    std::vector<segment_t> h_segs(segments.begin(), segments.end());
    for (auto& seg : h_segs) {
      for (uint32_t i = 0; i < static_cast<uint32_t>(seg.n_vectors); i++) {
        uint8_t cnt = seg.edge_counts[i];
        for (uint8_t j = 0; j < cnt; j++) {
          index_t& nb = seg.edges[i][j];
          if (nb != INVALID) nb += offset;
        }
      }
    }

    medoid += offset;
  }

  struct device_view {
    segment_t *segments;
    uint32_t   dim;
    index_t    n_vectors;
    uint32_t   n_segments;
    index_t    medoid;
    index_t    global_offset;

    __device__ __forceinline__
    index_t to_local(index_t global_idx) const {
      return global_idx - global_offset;
    }

    __device__ __forceinline__
    bool is_valid(index_t global_idx) const {
      if (global_idx < global_offset || global_idx >= global_offset + n_vectors) return false;
      index_t local_idx = to_local(global_idx);
      uint32_t seg_id = graph::segment_of(local_idx);
      uint32_t loc    = graph::local_of(local_idx);
      return seg_id < n_segments && segments[seg_id].is_valid_local_idx(loc);
    }

    __device__ __forceinline__
    index_t get_padded_dim() const {
      return vector_view<data_t>::pad(dim);
    }

    __device__ __forceinline__
    uint8_t get_edge_count(index_t global_idx) const {
      index_t local_idx = to_local(global_idx);
      assert(local_idx < n_vectors && "global_idx out of bounds");
      return segments[segment_of(local_idx)].get_edge_count(local_of(local_idx));
    }

    // returns a mutable neighbor reference
    __device__ __forceinline__
    auto& get_neighbor_list(index_t global_idx) {
      index_t local_idx = to_local(global_idx);
      assert(local_idx < n_vectors && "global_idx out of bounds");
      return segments[segment_of(local_idx)].get_neighbor_list(local_of(local_idx));
    }

    __device__ __forceinline__
    index_t get_neighbor(index_t global_idx, uint8_t neighbor_idx) const {
      index_t local_idx = to_local(global_idx);
      assert(local_idx < n_vectors && "global_idx out of bounds");
      return segments[segment_of(local_idx)].get_neighbor(local_of(local_idx), neighbor_idx);
    }

    __device__ __forceinline__
    float get_neighbor_dist(index_t global_idx, uint8_t neighbor_idx) const {
      index_t local_idx = to_local(global_idx);
      assert(local_idx < n_vectors && "global_idx out of bounds");
      return segments[segment_of(local_idx)].get_neighbor_dist(local_of(local_idx), neighbor_idx);
    }

    // returns a mutable vector reference
    __device__ __forceinline__
    auto get_vector(index_t global_idx) {
      index_t local_idx = to_local(global_idx);
      assert(local_idx < n_vectors && "global_idx out of bounds");
      return segments[segment_of(local_idx)].get_vector(local_of(local_idx));
    }

    __device__ __forceinline__
    void set_edge_count(index_t global_idx, uint8_t count) {
      index_t local_idx = to_local(global_idx);
      assert(local_idx < n_vectors && "global_idx out of bounds");
      segments[segment_of(local_idx)].set_edge_count(local_of(local_idx), count);
    }

    __device__ __forceinline__
    void set_neighbor(index_t global_idx, uint8_t neighbor_idx, index_t neighbor) {
      index_t local_idx = to_local(global_idx);
      assert(local_idx < n_vectors && "global_idx out of bounds");
      segments[segment_of(local_idx)].set_neighbor(local_of(local_idx), neighbor_idx, neighbor);
    }

    __device__ __forceinline__
    void set_neighbor_dist(index_t global_idx, uint8_t neighbor_idx, float dist_val) {
      index_t local_idx = to_local(global_idx);
      assert(local_idx < n_vectors && "global_idx out of bounds");
      segments[segment_of(local_idx)].set_neighbor_dist(local_of(local_idx), neighbor_idx, dist_val);
    }
  };

  __host__ double avg_degree() const {
    if (n_vectors == 0) return 0.0;

    std::vector<segment_t> h_segs(segments.begin(), segments.end());
    uint64_t total = 0;

    if (on_host) {
      for (const auto& seg : h_segs) {
        for (uint32_t i = 0; i < static_cast<uint32_t>(seg.n_vectors); i++)
          total += seg.edge_counts[i];
      }
    } else {
      for (const auto& seg : h_segs) {
        uint32_t count = static_cast<uint32_t>(seg.n_vectors);
        std::vector<uint8_t> h_counts(count);
        cudaMemcpy(h_counts.data(), seg.edge_counts,
                   count * sizeof(uint8_t), cudaMemcpyDeviceToHost);
        for (uint8_t c : h_counts) total += c;
      }
    }

    return static_cast<double>(total) / static_cast<double>(n_vectors);
  }

  // Dump neighbor lists to out. Iterates local indices [start, start+count).
  // count=0 means all vectors. Prints global index = local + global_offset.
  __host__ void dump_neighborhood(index_t start = 0, index_t count = 10,
                                  std::ostream& out = std::cout) const {
    index_t end = (count == 0) ? n_vectors
                               : std::min(start + count, n_vectors);

    std::vector<segment_t> h_segs(segments.begin(), segments.end());

    for (index_t idx = start; idx < end; idx++) {
      uint32_t    seg_id = segment_of(idx);
      uint32_t    loc    = local_of(idx);
      uint8_t     cnt;
      edge_list_t el;

      if (on_host) {
        cnt = h_segs[seg_id].edge_counts[loc];
        el  = h_segs[seg_id].edges[loc];
      } else {
        cudaMemcpy(&cnt, h_segs[seg_id].edge_counts + loc,
                   sizeof(uint8_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(&el,  h_segs[seg_id].edges + loc,
                   sizeof(edge_list_t), cudaMemcpyDeviceToHost);
      }

      out << "[" << (idx + global_offset) << "] ("
          << static_cast<int>(cnt) << " neighbors):";
      for (uint8_t j = 0; j < cnt; j++) {
        out << "  " << el.edges[j];
      }
      out << "\n";
    }
  }

  // return a view that can be pass to kernel
  device_view view() const {
    return device_view{
      const_cast<segment_t*>(thrust::raw_pointer_cast(segments.data())),
      dim,
      n_vectors,
      n_segments,
      medoid,
      global_offset
    };
  }
};

template <typename graph_cfg>
__host__ graph<graph_cfg> load_graph_from_file(std::string input_fname,
                                                  uint32_t dim,
                                                  bool on_host = false) {
  std::ifstream inputFile(input_fname, std::ios::binary);
  if (!inputFile.is_open()) {
    std::cerr << "Failed to open " << input_fname << " for graph load\n";
    exit(EXIT_FAILURE);
  }

  using index_t        = typename graph_cfg::index_t;
  using data_t         = typename graph_cfg::data_t;
  using edge_list_t    = typename graph_cfg::edge_list_t;
  using segment_t      = graph_segment<graph_cfg>;
  using vector_view_t  = typename graph_cfg::vector_view_t;

  static constexpr uint32_t vector_per_segment = graph_cfg::vectors_per_segment;

  uint64_t total_file_size;
  uint64_t big_n_vectors;
  uint64_t medoid_as_uint64;
  uint64_t bytes_per_node;

  // Read the metadata
  inputFile.read(reinterpret_cast<char*>(&total_file_size), sizeof(uint64_t));
  inputFile.read(reinterpret_cast<char*>(&big_n_vectors),   sizeof(uint64_t));
  inputFile.read(reinterpret_cast<char*>(&medoid_as_uint64), sizeof(uint64_t));
  inputFile.read(reinterpret_cast<char*>(&bytes_per_node),   sizeof(uint64_t));

  std::cout << "total_file_size: " << total_file_size << "\n";
  std::cout << "big_n_vectors: "   << big_n_vectors << "\n";
  std::cout << "medoid: "          << medoid_as_uint64 << "\n";
  std::cout << "bytes_per_node: "  << bytes_per_node << "\n";
  std::cout << "dim: "             << dim << "\n";

  uint32_t padded_dim = vector_view<data_t>::pad(dim);
  uint32_t n_segments = static_cast<uint32_t>((big_n_vectors + vector_per_segment - 1) / vector_per_segment);

  // Pinned host staging buffers (always pinned, regardless of on_host, for fast file I/O)
  uint8_t*     h_edge_counts;
  edge_list_t* h_edges;
  data_t*      h_vectors;

  cudaMallocHost(&h_edge_counts, sizeof(uint8_t)     * big_n_vectors);
  cudaMallocHost(&h_edges,       sizeof(edge_list_t) * big_n_vectors);
  cudaMallocHost(&h_vectors,     sizeof(data_t)      * big_n_vectors * padded_dim);
  memset(h_vectors, 0,           sizeof(data_t)      * big_n_vectors * padded_dim);

  for (uint64_t i = 0; i < big_n_vectors; i++) {
    inputFile.read(reinterpret_cast<char*>(&h_vectors[i * padded_dim]),
                   sizeof(data_t) * dim);
    uint8_t n_neighbors;
    inputFile.read(reinterpret_cast<char*>(&n_neighbors), sizeof(uint8_t));
    h_edge_counts[i] = n_neighbors;
    inputFile.read(reinterpret_cast<char*>(&h_edges[i]), sizeof(edge_list_t));
  }
  inputFile.close();

  // Source is pinned host; destination depends on where the segment lives.
  const cudaMemcpyKind copy_kind =
      on_host ? cudaMemcpyHostToHost : cudaMemcpyHostToDevice;

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  std::vector<segment_t> h_segments(n_segments);

  for (uint32_t s = 0; s < n_segments; s++) {
    uint64_t seg_begin = static_cast<uint64_t>(s) * vector_per_segment;
    uint32_t seg_count = static_cast<uint32_t>(
        std::min(static_cast<uint64_t>(vector_per_segment), big_n_vectors - seg_begin));

    segment_t& seg = h_segments[s];
    seg.n_vectors = static_cast<index_t>(seg_count);

    // Allocate edges / edge_counts on host or device to match on_host
    if (on_host) {
      cudaMallocHost(&seg.edges,       vector_per_segment * sizeof(edge_list_t));
      cudaMallocHost(&seg.edge_counts, vector_per_segment * sizeof(uint8_t));
      std::memset(seg.edge_counts, 0,  vector_per_segment * sizeof(uint8_t));
    } else {
      cudaMalloc(&seg.edges,       vector_per_segment * sizeof(edge_list_t));
      cudaMalloc(&seg.edge_counts, vector_per_segment * sizeof(uint8_t));
      cudaMemsetAsync(seg.edge_counts, 0,
                      vector_per_segment * sizeof(uint8_t), stream);
    }

    seg.vectors = vector_view_t::allocate(dim, vector_per_segment, on_host);
    seg.on_host = on_host;

    // Copy this segment's slice (kind picked above)
    cudaMemcpyAsync(seg.edge_counts,
                    &h_edge_counts[seg_begin],
                    seg_count * sizeof(uint8_t),
                    copy_kind, stream);

    cudaMemcpyAsync(seg.edges,
                    &h_edges[seg_begin],
                    seg_count * sizeof(edge_list_t),
                    copy_kind, stream);

    cudaMemcpyAsync(seg.vectors.data,
                    &h_vectors[seg_begin * padded_dim],
                    seg_count * padded_dim * sizeof(data_t),
                    copy_kind, stream);
  }

  cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);

  cudaFreeHost(h_edge_counts);
  cudaFreeHost(h_edges);
  cudaFreeHost(h_vectors);

  // create graph
  graph<graph_cfg> g{};
  g.dim        = dim;
  g.n_vectors  = static_cast<index_t>(big_n_vectors);
  g.n_segments = n_segments;
  g.medoid     = static_cast<index_t>(medoid_as_uint64);
  g.segments   = thrust::device_vector<segment_t>(h_segments.begin(), h_segments.end());
  g.on_host    = on_host;

  cudaDeviceSynchronize();
  return g;
}

template <typename graph_cfg>
__host__ void save_graph_to_file(const graph<graph_cfg>& g,
                                 std::string output_fname) {
  using data_t         = typename graph_cfg::data_t;
  using edge_list_t    = typename graph_cfg::edge_list_t;
  using segment_t      = graph_segment<graph_cfg>;
  using vector_view_t  = typename graph_cfg::vector_view_t;

  static constexpr uint32_t vectors_per_segment = graph_cfg::vectors_per_segment;

  std::ofstream outFile(output_fname, std::ios::binary);
  if (!outFile.is_open()) {
    std::cerr << "Failed to open " << output_fname << " for graph save\n";
    exit(EXIT_FAILURE);
  }

  uint64_t big_n_vectors    = static_cast<uint64_t>(g.n_vectors);
  uint64_t medoid_as_uint64 = static_cast<uint64_t>(g.medoid);
  uint32_t dim              = g.dim;
  uint32_t padded_dim       = vector_view_t::pad(g.dim);
  // File format uses packed dim, not padded
  uint64_t bytes_per_node   = sizeof(data_t) * dim + sizeof(uint8_t) + sizeof(edge_list_t);
  uint64_t total_file_size  = 4 * sizeof(uint64_t) + big_n_vectors * bytes_per_node;

  // Write metadata
  outFile.write(reinterpret_cast<const char*>(&total_file_size),  sizeof(uint64_t));
  outFile.write(reinterpret_cast<const char*>(&big_n_vectors),    sizeof(uint64_t));
  outFile.write(reinterpret_cast<const char*>(&medoid_as_uint64), sizeof(uint64_t));
  outFile.write(reinterpret_cast<const char*>(&bytes_per_node),   sizeof(uint64_t));

  // Copy segment structs (pointers + n_vectors) back to host
  std::vector<segment_t> h_segments(g.segments.begin(), g.segments.end());

  if (g.on_host) {
    // Buffers already on host — write straight from the segment pointers.
    for (uint32_t s = 0; s < g.n_segments; s++) {
      const segment_t& seg = h_segments[s];
      uint32_t seg_count   = static_cast<uint32_t>(seg.n_vectors);

      for (uint32_t i = 0; i < seg_count; i++) {
        outFile.write(reinterpret_cast<const char*>(seg.vectors.data + i * padded_dim),
                      sizeof(data_t) * dim);
        outFile.write(reinterpret_cast<const char*>(&seg.edge_counts[i]),
                      sizeof(uint8_t));
        outFile.write(reinterpret_cast<const char*>(&seg.edges[i]),
                      sizeof(edge_list_t));
      }
    }
  } else {
    // Buffers on device — stage one segment at a time through pinned host memory.
    uint8_t*     h_edge_counts;
    edge_list_t* h_edges;
    data_t*      h_vectors;

    cudaMallocHost(&h_edge_counts, sizeof(uint8_t)     * vectors_per_segment);
    cudaMallocHost(&h_edges,       sizeof(edge_list_t) * vectors_per_segment);
    cudaMallocHost(&h_vectors,     sizeof(data_t)      * vectors_per_segment * padded_dim);

    for (uint32_t s = 0; s < g.n_segments; s++) {
      const segment_t& seg = h_segments[s];
      uint32_t seg_count   = static_cast<uint32_t>(seg.n_vectors);

      cudaMemcpy(h_edge_counts, seg.edge_counts,
                 seg_count * sizeof(uint8_t),     cudaMemcpyDeviceToHost);
      cudaMemcpy(h_edges,       seg.edges,
                 seg_count * sizeof(edge_list_t), cudaMemcpyDeviceToHost);
      cudaMemcpy(h_vectors,     seg.vectors.data,
                 seg_count * padded_dim * sizeof(data_t), cudaMemcpyDeviceToHost);
      cudaDeviceSynchronize();

      for (uint32_t i = 0; i < seg_count; i++) {
        outFile.write(reinterpret_cast<const char*>(&h_vectors[i * padded_dim]),
                      sizeof(data_t) * dim);
        outFile.write(reinterpret_cast<const char*>(&h_edge_counts[i]),
                      sizeof(uint8_t));
        outFile.write(reinterpret_cast<const char*>(&h_edges[i]),
                      sizeof(edge_list_t));
      }
    }

    cudaFreeHost(h_edge_counts);
    cudaFreeHost(h_edges);
    cudaFreeHost(h_vectors);
  }

  outFile.close();

  std::cout << "Saved graph to " << output_fname << "\n"
            << "  n_vectors       : " << big_n_vectors   << "\n"
            << "  medoid          : " << g.medoid        << "\n"
            << "  dim             : " << dim             << "\n"
            << "  total_file_size : " << total_file_size << "\n";
}

}