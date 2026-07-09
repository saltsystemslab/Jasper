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
// magnitude field: ||e||² is reconstructed at search time as the sum of the
// chosen centroids' squared norms (see pq_codebooks::cnorm).
//
// SoA layout mirrors edge_lsh_list: all edges' M-byte codes are laid out
// contiguously, edge e's code starting at code[e * M].
template <typename INDEX_T, uint32_t N_NEIGHBORS, uint32_t M>
struct alignas(4) edge_pq_list {
  static_assert(M > 0, "PQ requires M > 0 subquantizers");

  // code[edge * M + sub] — centroid id (0..255) for subspace `sub` of `edge`.
  uint8_t code[N_NEIGHBORS * M];

  __host__ __device__ __forceinline__
  uint8_t get_code(uint8_t edge_idx, uint32_t sub) const {
    return code[static_cast<uint32_t>(edge_idx) * M + sub];
  }

  __host__ __device__ __forceinline__
  void set_code(uint8_t edge_idx, uint32_t sub, uint8_t c) {
    code[static_cast<uint32_t>(edge_idx) * M + sub] = c;
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
