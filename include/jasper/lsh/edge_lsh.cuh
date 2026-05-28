#pragma once

namespace jasper {

template <typename INDEX_T, uint32_t N_NEIGHBORS, uint32_t K_RANKS>
struct edge_lsh_list {
  // packed[i][r]:
  //   bit 15      = sign of the r-th-rank coord (1 = negative)
  //   bits 0..14  = absolute coord in [0, D), D ≤ 32768
  uint16_t packed[N_NEIGHBORS][K_RANKS];

  __device__ __forceinline__
  uint16_t get_packed(uint8_t edge_idx, uint8_t rank) const {
    return packed[edge_idx][rank];
  }

  __device__ __forceinline__
  uint16_t get_coord(uint8_t edge_idx, uint8_t rank) const {
    return packed[edge_idx][rank] & uint16_t{0x7FFF};
  }

  __device__ __forceinline__
  float get_sign(uint8_t edge_idx, uint8_t rank) const {
    // bit 15 == 1 → negative
    return (packed[edge_idx][rank] & uint16_t{0x8000}) ? -1.0f : +1.0f;
  }

  __host__ void print(std::ostream& out = std::cout, uint32_t len = 0) const {

    if (len==0) len = N_NEIGHBORS;

    for (uint32_t e = 0; e < len; ++e) {
      out << "  edge[" << e << "]:";
      for (uint32_t r = 0; r < K_RANKS; ++r) {
        const uint16_t p     = packed[e][r];
        const uint16_t coord = p & uint16_t{0x7FFF};
        const char     sign  = (p & uint16_t{0x8000}) ? '-' : '+';
        out << ' ' << sign << coord;
      }
      out << '\n';
    }
  }
};

}