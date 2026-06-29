#pragma once

#include <cstdint>
#include <vector>

#include <cuda_bf16.h>
#include <thrust/device_vector.h>

#include "jasper/distance/distance.cuh"
#include "jasper/lsh/edge_lsh.cuh"
#include "jasper/lsh/lsh_globals.cuh"
#include "jasper/lsh/lsh_kernels.cuh"

#include "assert.h"
#include "stdio.h"

namespace jasper {
  
template <typename INDEX_T, uint32_t N_NEIGHBORS>
struct edge_list {
  INDEX_T edges[N_NEIGHBORS];
  __nv_bfloat16 dist[N_NEIGHBORS];

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
          distance_func DIST_FUNC,
          bool USE_LSH = false,
          uint8_t K_RANKS = 0,
          typename PACKED_T = uint8_t>
struct graph_config {
  using index_t = INDEX_T;
  using data_t = DATA_T;
  using distance_t = DISTANCE_T;
  using edge_list_t = edge_list<INDEX_T, N_NEIGHBORS>;
  using edge_lsh_list_t = edge_lsh_list<INDEX_T, N_NEIGHBORS, K_RANKS, PACKED_T>;
  using packed_t = PACKED_T;
  using vector_view_t = vector_view<DATA_T>;
  
  static constexpr uint8_t n_neighbors = N_NEIGHBORS;
  static constexpr distance_func dist_func = DIST_FUNC;
  static constexpr index_t vectors_per_segment = 1u << 20;
  static constexpr bool use_lsh = USE_LSH;
  static constexpr uint8_t k_ranks = K_RANKS;

  static_assert(!USE_LSH || K_RANKS > 0, "USE_LSH requires K_RANKS > 0");
  static_assert(K_RANKS <= N_NEIGHBORS, "K_RANKS must fit in a neighbor slot");
};

template <typename graph_cfg>
struct graph_segment {
  using index_t     = typename graph_cfg::index_t;
  using edge_list_t = typename graph_cfg::edge_list_t;
  using edge_lsh_list_t = typename graph_cfg::edge_lsh_list_t;
  using vector_view_t    = typename graph_cfg::vector_view_t;

  static constexpr uint32_t max_vectors = graph_cfg::vectors_per_segment;
  static constexpr uint32_t n_neighbors = graph_cfg::n_neighbors;

  index_t n_vectors;

  // storage (preallocated to graph_cfg::vectors_per_segement)
  edge_list_t *edges;
  uint8_t *edge_counts;
  vector_view_t vectors;
  edge_lsh_list_t* edge_lshs; // only enabled if USE_LSH

  // if on_host is true, the data is allocated on cpu pinned memory
  bool on_host = false;

  static graph_segment allocate(uint32_t dim, bool on_host=false) {
    graph_segment seg{};
    seg.n_vectors = 0;
    seg.on_host = on_host;

    auto check = [](cudaError_t err, const char* name) {
      if (err != cudaSuccess)
        throw std::runtime_error(std::string(name) + " failed: " + cudaGetErrorString(err));
    };

    if (on_host) {
        check(cudaMallocHost(&seg.edges, max_vectors * sizeof(edge_list_t)), "cudaMallocHost(edges)");
        check(cudaMallocHost(&seg.edge_counts, max_vectors * sizeof(uint8_t)), "cudaMallocHost(edge_counts)");
        std::memset(seg.edge_counts, 0, max_vectors * sizeof(uint8_t));
    } else {
        check(cudaMalloc(&seg.edges, max_vectors * sizeof(edge_list_t)), "cudaMalloc(edges)");
        check(cudaMalloc(&seg.edge_counts, max_vectors * sizeof(uint8_t)), "cudaMalloc(edge_counts)");
        check(cudaMemset(seg.edge_counts, 0, max_vectors * sizeof(uint8_t)), "cudaMemset(edge_counts)");
    }

    seg.vectors = vector_view_t::allocate(dim, max_vectors, on_host);

    // edge_lshs is allocated lazily on the first populate_edge_lsh() call.
    seg.edge_lshs = nullptr;

    return seg;
  }

  // Lazily allocate the edge_lsh storage. No-op if already allocated or if
  // use_lsh is disabled. Allocated on host pinned memory or device to match
  // this segment's on_host flag.
  void allocate_edge_lsh() {
    if constexpr (graph_cfg::use_lsh) {
      if (edge_lshs != nullptr) return;

      auto check = [](cudaError_t err, const char* name) {
        if (err != cudaSuccess)
          throw std::runtime_error(std::string(name) + " failed: " + cudaGetErrorString(err));
      };

      if (on_host) {
        check(cudaMallocHost(&edge_lshs, max_vectors * sizeof(edge_lsh_list_t)),
              "cudaMallocHost(edge_lshs)");
      } else {
        check(cudaMalloc(&edge_lshs, max_vectors * sizeof(edge_lsh_list_t)),
              "cudaMalloc(edge_lshs)");
      }
    }
  }

  void deallocate() {
    if (on_host) {
        cudaFreeHost(edges);
        cudaFreeHost(edge_counts);
    } else {
        cudaFree(edges);
        cudaFree(edge_counts);
    }
    if constexpr (graph_cfg::use_lsh) {
      if (on_host) cudaFreeHost(edge_lshs); else cudaFree(edge_lshs);
      edge_lshs = nullptr;
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
    edge_lsh_list_t* new_edge_lshs = nullptr;

    if (target_on_host) {
      cudaMallocHost(&new_edges,       max_vectors * sizeof(edge_list_t));
      cudaMallocHost(&new_edge_counts, max_vectors * sizeof(uint8_t));
      std::memset(new_edge_counts, 0,  max_vectors * sizeof(uint8_t));
      if constexpr (graph_cfg::use_lsh) {
        if (edge_lshs != nullptr)
          cudaMallocHost(&new_edge_lshs, max_vectors * sizeof(edge_lsh_list_t));
      }
    } else {
      cudaMalloc(&new_edges,       max_vectors * sizeof(edge_list_t));
      cudaMalloc(&new_edge_counts, max_vectors * sizeof(uint8_t));
      cudaMemsetAsync(new_edge_counts, 0,
                      max_vectors * sizeof(uint8_t), stream);
      if constexpr (graph_cfg::use_lsh) {
        if (edge_lshs != nullptr)
          cudaMalloc(&new_edge_lshs, max_vectors * sizeof(edge_lsh_list_t));
      }
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
      if constexpr (graph_cfg::use_lsh) {
        if (edge_lshs != nullptr)
          cudaMemcpyAsync(new_edge_lshs, edge_lshs,
                          static_cast<size_t>(n_vectors) * sizeof(edge_lsh_list_t),
                          copy_kind, stream);
      }
      cudaMemcpyAsync(new_vectors.data, vectors.data,
                      static_cast<size_t>(n_vectors) * padded_dim * sizeof(data_t),
                      copy_kind, stream);
    }
    cudaStreamSynchronize(stream);  // must complete before freeing old buffers

    // Free the old buffers using the *previous* on_host flag.
    if (on_host) {
      cudaFreeHost(edges);
      cudaFreeHost(edge_counts);
      if constexpr (graph_cfg::use_lsh) cudaFreeHost(edge_lshs);
    } else {
      cudaFree(edges);
      cudaFree(edge_counts);
      if constexpr (graph_cfg::use_lsh) cudaFree(edge_lshs);
    }
    vectors.deallocate();

    // Install new buffers.
    edges       = new_edges;
    edge_counts = new_edge_counts;
    if constexpr (graph_cfg::use_lsh) edge_lshs = new_edge_lshs;
    vectors     = new_vectors;
    on_host     = target_on_host;
  }

  // Copy this segment's live data (edges, edge_counts, vectors) into an
  // already-allocated target segment. Memcpy direction is inferred from
  // each side's on_host flag. Only the valid [0, n_vectors) prefix is copied.
  // Caller is responsible for synchronizing the stream before reading target.
  void copy_to(graph_segment& target, cudaStream_t stream = 0) const {
    using data_t = typename graph_cfg::data_t;

    if (target.vectors.dim != vectors.dim) {
      throw std::runtime_error("graph_segment::copy_to: dim mismatch");
    }

    const cudaMemcpyKind kind = [&]{
      if      ( on_host &&  target.on_host) return cudaMemcpyHostToHost;
      else if ( on_host && !target.on_host) return cudaMemcpyHostToDevice;
      else if (!on_host &&  target.on_host) return cudaMemcpyDeviceToHost;
      else                                  return cudaMemcpyDeviceToDevice;
    }();

    target.n_vectors = n_vectors;

    if (n_vectors > 0) {
      const uint32_t padded_dim = vector_view_t::pad(vectors.dim);
      cudaMemcpyAsync(target.edges, edges,
                      static_cast<size_t>(n_vectors) * sizeof(edge_list_t),
                      kind, stream);
      cudaMemcpyAsync(target.edge_counts, edge_counts,
                      static_cast<size_t>(n_vectors) * sizeof(uint8_t),
                      kind, stream);
      if constexpr (graph_cfg::use_lsh) {
        if (edge_lshs != nullptr) {
          target.allocate_edge_lsh();  // no-op if target already has it
          cudaMemcpyAsync(target.edge_lshs, edge_lshs,
                          static_cast<size_t>(n_vectors) * sizeof(edge_lsh_list_t),
                          kind, stream);
        }
      }
      cudaMemcpyAsync(target.vectors.data, vectors.data,
                      static_cast<size_t>(n_vectors) * padded_dim * sizeof(data_t),
                      kind, stream);
    }
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
    return __bfloat162float(edges[local_idx].dist[neighbor_idx]);
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
    edges[local_idx].dist[neighbor_idx] = (__nv_bfloat16)dist_val;
  }
};

template <typename graph_cfg>
struct graph {
  using index_t     = typename graph_cfg::index_t;
  using data_t = typename graph_cfg::data_t;
  using segment_t   = graph_segment<graph_cfg>;
  using edge_list_t = typename graph_cfg::edge_list_t;
  using edge_lsh_list_t = typename graph_cfg::edge_lsh_list_t;
  using packed_t = typename graph_cfg::packed_t;
  using vector_view_t    = typename graph_cfg::vector_view_t;

  static constexpr uint32_t n_neighbors = graph_cfg::n_neighbors;
  static constexpr uint32_t vectors_per_segment = graph_cfg::vectors_per_segment;
  static constexpr bool use_lsh = graph_cfg::use_lsh;
  static constexpr uint8_t k_ranks = graph_cfg::k_ranks;

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
    g.n_segments = (n_vector_slots + vectors_per_segment - 1) / vectors_per_segment;
    g.on_host = on_host;

    const size_t padded_dim = vector_view_t::pad(dim);
    size_t bytes_per_segment =
        static_cast<size_t>(vectors_per_segment) * sizeof(edge_list_t)
      + static_cast<size_t>(vectors_per_segment) * sizeof(uint8_t)
      + static_cast<size_t>(vectors_per_segment) * padded_dim * sizeof(data_t);
    if constexpr (use_lsh) {
      bytes_per_segment +=
          static_cast<size_t>(vectors_per_segment) * sizeof(edge_lsh_list_t);
    }
    const size_t total_bytes = bytes_per_segment * g.n_segments;

    std::cout << "[graph::allocate]: " << g.n_segments << " segments x "
              << (bytes_per_segment / (1ull << 20)) << " MB = "
              << (total_bytes / (1ull << 20)) << " MB ("
              << (total_bytes / (1ull << 30)) << " GB) on "
              << (on_host ? "host" : "device");
    if constexpr (use_lsh) {
      std::cout << " (LSH enabled, +"
                << (static_cast<size_t>(vectors_per_segment)
                    * sizeof(edge_lsh_list_t) / (1ull << 20))
                << " MB/segment for edge_lsh_list)";
    }
    std::cout << "\n";

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

  // Append data.n_vectors new vectors to the end of the graph, growing the
  // segment array if the currently allocated segments cannot hold them.
  __host__ index_t insert(vector_view_t data) {
    if (data.dim != dim) {
      throw std::runtime_error("graph::insert: dim mismatch");
    }

    const uint32_t padded_dim = vector_view_t::pad(dim);
    const index_t  start = n_vectors;
    const index_t  count = static_cast<index_t>(data.n_vectors);
    if (count == 0) return start;
    const index_t  end = start + count;

    // Pull segment structs back to host so we can mutate pointers / counts,
    // and grow the segment array if the new vectors don't fit.
    std::vector<segment_t> h_segments(segments.begin(), segments.end());

    const uint32_t required_segments = static_cast<uint32_t>(
        (end + vectors_per_segment - 1) / vectors_per_segment);
    // Only ever grow the segment array — never shrink. When the graph was
    // pre-allocated with enough capacity (required_segments <= n_segments),
    // this allocates nothing. Unconditionally assigning n_segments here would
    // wrongly drop already-allocated segments and leak them on the next grow.
    if (required_segments > n_segments) {
      for (uint32_t s = n_segments; s < required_segments; s++) {
        h_segments.push_back(segment_t::allocate(dim, on_host));
      }
      n_segments = required_segments;
    }

    // Memcpy direction inferred from where the source data and segments live.
    const cudaMemcpyKind copy_kind = [&]{
      if (data.on_host &&  on_host) return cudaMemcpyHostToHost;
      if (data.on_host && !on_host) return cudaMemcpyHostToDevice;
      if (!data.on_host && on_host) return cudaMemcpyDeviceToHost;
      return cudaMemcpyDeviceToDevice;
    }();

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Copy the new vectors into the segments, spanning segment boundaries.
    index_t copied = 0;
    for (index_t idx = start; idx < end; ) {
      uint32_t seg_id    = segment_of(idx);
      uint32_t local_idx = local_of(idx);
      uint32_t space     = vectors_per_segment - local_idx;
      uint32_t remaining = static_cast<uint32_t>(end - idx);
      uint32_t n         = std::min(space, remaining);

      cudaMemcpyAsync(
          h_segments[seg_id].vectors.data + static_cast<size_t>(local_idx) * padded_dim,
          data.data + static_cast<size_t>(copied) * padded_dim,
          static_cast<size_t>(n) * padded_dim * sizeof(data_t),
          copy_kind, stream);

      h_segments[seg_id].n_vectors = static_cast<index_t>(local_idx + n);

      idx    += n;
      copied += n;
    }

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    n_vectors = end;

    // Re-upload segment structs so device-side view() sees the new pointers
    // and updated per-segment n_vectors.
    segments = thrust::device_vector<segment_t>(h_segments.begin(), h_segments.end());

    cudaDeviceSynchronize();
    return start;
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

  // Copy this graph's contents (segments + medoid metadata) into an
  // already-allocated target graph. The target may live on host or device,
  // independent of this graph. Target must have matching dim and at least
  // n_segments segment slots.
  void copy_to(graph& target) const {
    if (target.dim != dim) {
      throw std::runtime_error("graph::copy_to: dim mismatch");
    }
    if (target.n_segments < n_segments) {
      throw std::runtime_error("graph::copy_to: target has too few segments");
    }

    std::vector<segment_t> h_src(segments.begin(), segments.end());
    std::vector<segment_t> h_tgt(target.segments.begin(), target.segments.end());

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    for (uint32_t i = 0; i < n_segments; i++) {
      h_src[i].copy_to(h_tgt[i], stream);
    }
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    target.n_vectors     = n_vectors;
    target.medoid        = medoid;
    target.global_offset = global_offset;

    // Re-upload target segments so device-side structs see the updated n_vectors.
    target.segments = thrust::device_vector<segment_t>(h_tgt.begin(), h_tgt.end());

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

    __device__ __forceinline__
    uint32_t get_lsh_coord(index_t global_idx, uint8_t edge_idx, uint8_t rank) const {
      if constexpr (use_lsh) {
        index_t local_idx = to_local(global_idx);
        assert(local_idx < n_vectors && "global_idx out of bounds");
        return segments[segment_of(local_idx)]
            .edge_lshs[local_of(local_idx)]
            .get_coord(edge_idx, rank);
      } else {
        assert(false && "get_lsh_coord called with use_lsh=false");
        return 0;
      }
    }

    __device__ __forceinline__
    void set_lsh_coord(index_t global_idx, uint8_t edge_idx, uint8_t rank, uint32_t coord) {
      if constexpr (use_lsh) {
        // slot width (uint8_t / uint16_t) is fixed at compile time by packed_t.
        constexpr packed_t SIGN_MASK  = static_cast<packed_t>(packed_t{1} << (sizeof(packed_t) * 8 - 1));
        constexpr packed_t COORD_MASK = static_cast<packed_t>(~SIGN_MASK);
        index_t local_idx = to_local(global_idx);
        assert(local_idx < n_vectors && "global_idx out of bounds");
        assert(coord <= COORD_MASK && "coord does not fit in packed_t coord bits");
        packed_t& slot = segments[segment_of(local_idx)]
            .edge_lshs[local_of(local_idx)]
            .rows[edge_idx].packed[rank];
        slot = static_cast<packed_t>((slot & SIGN_MASK) | (static_cast<packed_t>(coord) & COORD_MASK));
      } else {
        assert(false && "set_lsh_coord called with use_lsh=false");
      }
    }

    // Store a pre-packed coord|sign word (sign in the MSB, coord in the low
    // bits) in one write — used when the caller already holds the packed form.
    __device__ __forceinline__
    void set_lsh_packed(index_t global_idx, uint8_t edge_idx, uint8_t rank, packed_t word) {
      if constexpr (use_lsh) {
        index_t local_idx = to_local(global_idx);
        assert(local_idx < n_vectors && "global_idx out of bounds");
        segments[segment_of(local_idx)]
            .edge_lshs[local_of(local_idx)]
            .rows[edge_idx].packed[rank] = word;
      } else {
        assert(false && "set_lsh_packed called with use_lsh=false");
      }
    }

    __device__ __forceinline__
    float get_lsh_mag_sq(index_t global_idx, uint8_t edge_idx) const {
      if constexpr (use_lsh) {
        index_t local_idx = to_local(global_idx);
        assert(local_idx < n_vectors && "global_idx out of bounds");
        return segments[segment_of(local_idx)]
            .edge_lshs[local_of(local_idx)]
            .get_mag_sq(edge_idx);
      } else {
        assert(false && "get_lsh_mag_sq called with use_lsh=false");
        return 0.0f;
      }
    }

    __device__ __forceinline__
    void set_lsh_sign(index_t global_idx, uint8_t edge_idx, uint8_t rank, bool is_positive) {
      if constexpr (use_lsh) {
        constexpr packed_t SIGN_MASK  = static_cast<packed_t>(packed_t{1} << (sizeof(packed_t) * 8 - 1));
        constexpr packed_t COORD_MASK = static_cast<packed_t>(~SIGN_MASK);
        index_t local_idx = to_local(global_idx);
        assert(local_idx < n_vectors && "global_idx out of bounds");
        packed_t& slot = segments[segment_of(local_idx)]
            .edge_lshs[local_of(local_idx)]
            .rows[edge_idx].packed[rank];
        if (is_positive) slot &= COORD_MASK;
        else             slot |= SIGN_MASK;
      } else {
        assert(false && "set_lsh_sign called with use_lsh=false");
      }
    }

    __device__ __forceinline__
    void set_lsh_mag_sq(index_t global_idx, uint8_t edge_idx, __nv_bfloat16 mag_sq) {
      if constexpr (use_lsh) {
        index_t local_idx = to_local(global_idx);
        assert(local_idx < n_vectors && "global_idx out of bounds");
        segments[segment_of(local_idx)]
            .edge_lshs[local_of(local_idx)]
            .rows[edge_idx].mag_sq = mag_sq;
      } else {
        assert(false && "set_lsh_mag_sq called with use_lsh=false");
      }
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

  // Given a rotated vector, populate the edge lsh.
  void populate_edge_lsh() {
    static_assert(graph_cfg::use_lsh,
                  "populate_edge_lsh requires graph_cfg::use_lsh");

    // Lazily allocate edge_lsh storage on the first call. Pull the segment
    // structs back to host, allocate any missing edge_lsh buffers, then
    // re-upload so the device-side structs (and view()) see the new pointers.
    {
      std::vector<segment_t> h_segs(segments.begin(), segments.end());
      bool any_allocated = false;
      for (auto& seg : h_segs) {
        if (seg.edge_lshs == nullptr) {
          seg.allocate_edge_lsh();
          any_allocated = true;
        }
      }
      if (any_allocated) {
        segments = thrust::device_vector<segment_t>(h_segs.begin(), h_segs.end());
      }
    }

    // Pick a block size. 128 or 256 is usually fine for this kernel since the
    // work per thread is light. Put it in graph_cfg if you want it tunable.
    constexpr uint32_t block_threads = 128;
    static_assert(block_threads % 32 == 0);

    const uint32_t padded_dim = get_padded_dim();
    const uint32_t nwarps     = block_threads / 32;

    const size_t smem_bytes =
          ((padded_dim * sizeof(data_t)   + 15) & ~15)   // relative_vec
        + ((padded_dim * sizeof(uint32_t) + 15) & ~15)   // packed keys
        +  nwarps * sizeof(uint32_t);                    // warp scratch

    // Opt in to >48KB shared mem if needed. Harmless if smem_bytes is small.
    // static bool attr_set = false;
    // if (!attr_set) {
    //   cudaFuncSetAttribute(populate_lsh<graph_cfg>,
    //                       cudaFuncAttributeMaxDynamicSharedMemorySize,
    //                       96 * 1024);   // raise if your padded_dim is large
    //   attr_set = true;
    // }

    const dim3 grid(static_cast<uint32_t>(n_vectors));
    populate_lsh<graph_cfg><<<grid, block_threads, smem_bytes>>>(view());
  }

  lsh_globals<graph_cfg::k_ranks> generate_lsh_globals(
    uint32_t n_samples = 16384,
    uint64_t seed      = 42
  ) const {
    static_assert(graph_cfg::use_lsh,
                  "generate_lsh_globals requires graph_cfg::use_lsh");

    if (on_host) {
      throw std::runtime_error(
          "generate_lsh_globals requires graph on device — "
          "call move_to(false) first");
    }
    if (n_vectors == 0) {
      throw std::runtime_error("generate_lsh_globals: graph is empty");
    }

    constexpr uint8_t  k_ranks       = graph_cfg::k_ranks;
    constexpr uint32_t block_threads = 128;
    static_assert(block_threads % 32 == 0);

    using globals_t = lsh_globals<k_ranks>;

    const uint32_t padded_dim = get_padded_dim();
    const uint32_t nwarps     = block_threads / 32;
    const size_t   smem_bytes =
          ((padded_dim * sizeof(uint32_t) + 15) & ~15)   // packed keys
        +  nwarps * sizeof(uint32_t);                    // warp scratch

    float*     d_rank_sum = nullptr;
    uint32_t*  d_n_valid  = nullptr;
    globals_t* d_globals  = nullptr;
    cudaMalloc(&d_rank_sum, k_ranks * sizeof(float));
    cudaMalloc(&d_n_valid,  sizeof(uint32_t));
    cudaMalloc(&d_globals,  sizeof(globals_t));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaMemsetAsync(d_rank_sum, 0, k_ranks * sizeof(float), stream);
    cudaMemsetAsync(d_n_valid,  0, sizeof(uint32_t),        stream);

    accumulate_rank_sums<graph_cfg>
        <<<n_samples, block_threads, smem_bytes, stream>>>(
            view(), d_rank_sum, d_n_valid, seed);

    finalize_lsh_globals<k_ranks>
        <<<1, 1, 0, stream>>>(d_rank_sum, d_n_valid, d_globals);

    globals_t h_globals{};
    cudaMemcpyAsync(&h_globals, d_globals, sizeof(globals_t),
                    cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    cudaFree(d_globals);
    cudaFree(d_n_valid);
    cudaFree(d_rank_sum);

    return h_globals;
  }

  __host__ void dump_edge_lsh(index_t start = 0, index_t count = 10,
                          std::ostream& out = std::cout) const {
    if constexpr (!graph_cfg::use_lsh) {
      out << "[dump_edge_lsh] use_lsh disabled for this config\n";
      return;
    } else {
      index_t end = (count == 0) ? n_vectors
                                : std::min(start + count, n_vectors);

      std::vector<segment_t> h_segs(segments.begin(), segments.end());

      for (index_t idx = start; idx < end; ++idx) {
        uint32_t        seg_id = segment_of(idx);
        uint32_t        loc    = local_of(idx);
        uint8_t         cnt;
        edge_list_t     edges;
        edge_lsh_list_t el;

        if (h_segs[seg_id].edge_lshs == nullptr) {
          out << "[" << (idx + global_offset)
              << "] LSH not populated (call populate_edge_lsh() first)\n";
          continue;
        }

        if (on_host) {
          cnt   = h_segs[seg_id].edge_counts[loc];
          edges = h_segs[seg_id].edges[loc];
          el    = h_segs[seg_id].edge_lshs[loc];
        } else {
          cudaMemcpy(&cnt,   h_segs[seg_id].edge_counts + loc,
                    sizeof(uint8_t),         cudaMemcpyDeviceToHost);
          cudaMemcpy(&edges, h_segs[seg_id].edges       + loc,
                    sizeof(edge_list_t),     cudaMemcpyDeviceToHost);
          cudaMemcpy(&el,    h_segs[seg_id].edge_lshs   + loc,
                    sizeof(edge_lsh_list_t), cudaMemcpyDeviceToHost);
        }

        out << "[" << (idx + global_offset) << "] LSH ("
            << static_cast<int>(cnt) << " edges):\n";

        for (uint8_t e = 0; e < cnt; ++e) {
          out << "  edge[" << static_cast<int>(e)
              << "] -> " << edges.edges[e] << ":";
          for (uint32_t r = 0; r < graph_cfg::k_ranks; ++r) {
            const uint32_t coord = el.get_coord(e, r);
            const char     sgn   = (el.get_sign(e, r) < 0.0f) ? '-' : '+';
            out << ' ' << sgn << coord;
          }
          out << '\n';
        }
      }
    }
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