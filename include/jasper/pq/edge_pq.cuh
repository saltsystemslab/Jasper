#pragma once

#include <cstdint>
#include <iostream>

namespace jasper {

// Per-node Product-Quantization codes for its out-edges.
//
// Product Quantization splits the D-dim edge residual e = nb - u into M
// contiguous subspaces of size D/M. Each subspace has its own codebook of
// K = 256 centroids, so a residual is encoded as M uint8 centroid ids — M
// bytes per edge. Unlike the cross-polytope edge_lsh_list there is NO separate
// magnitude field: the PQ codes capture only the residual direction, and the
// L2 estimator recovers ||q-v||² using each vector's exact stored ||v||²
// (see graph::d_vector_norms).
//
// Subspace-major SoA layout: all edges' codes for a given subspace are laid
// out contiguously, so code[sub * N_NEIGHBORS + edge]. This lets the ADC
// scoring loop read one subspace across a warp of edges as a single coalesced
// transaction (thread e reads consecutive byte e), rather than the strided
// (edge * M + sub) access an edge-major layout would force.
template <typename INDEX_T, uint32_t N_NEIGHBORS, uint32_t M>
struct alignas(4) edge_pq_list {
  static_assert(M > 0, "PQ requires M > 0 subquantizers");

  // code[sub * N_NEIGHBORS + edge] — centroid id (0..255) for subspace `sub`
  // of `edge`. Subspace-major so consecutive edges are adjacent in memory.
  uint8_t code[N_NEIGHBORS * M];

  __host__ __device__ __forceinline__
  uint8_t get_code(uint8_t edge_idx, uint32_t sub) const {
    return code[sub * N_NEIGHBORS + static_cast<uint32_t>(edge_idx)];
  }

  __host__ __device__ __forceinline__
  void set_code(uint8_t edge_idx, uint32_t sub, uint8_t c) {
    code[sub * N_NEIGHBORS + static_cast<uint32_t>(edge_idx)] = c;
  }

  __host__ void print(std::ostream& out = std::cout, uint32_t len = 0) const {
    if (len == 0) len = N_NEIGHBORS;
    for (uint32_t e = 0; e < len; ++e) {
      out << "  edge[" << e << "] code:";
      for (uint32_t j = 0; j < M; ++j)
        out << ' ' << static_cast<uint32_t>(get_code(static_cast<uint8_t>(e), j));
      out << '\n';
    }
  }
};

}  // namespace jasper
