#pragma once

#include <cuda_bf16.h>
#include <cstdint>
#include <type_traits>

namespace jasper {

template <typename INDEX_T, uint32_t N_NEIGHBORS, uint32_t K_RANKS, typename PACKED_T>
struct edge_lsh_list {

  static_assert(
    std::is_same_v<PACKED_T, uint8_t> || std::is_same_v<PACKED_T, uint16_t>,
    "PACKED_T must be uint8_t or uint16_t"
  );

  // Per-edge row, packed:
  //   packed[K_RANKS]   — K uint8/uint16 LSH words (sign bit in MSB, coord in low bits)
  //   mag_sq            — ||v-u||² as bf16
  // __align__(4) guarantees row size is a multiple of 4 bytes.
  struct __align__(4) row_t {
    PACKED_T      packed[K_RANKS];
    __nv_bfloat16 mag_sq;
  };
  row_t rows[N_NEIGHBORS];

  __device__ __forceinline__
  PACKED_T get_packed(uint8_t edge_idx, uint8_t rank) const {
    return rows[edge_idx].packed[rank];
  }
  __host__ __device__ __forceinline__
  uint32_t get_coord(uint8_t edge_idx, uint8_t rank) const {
    if constexpr (std::is_same_v<PACKED_T, uint8_t>) {
      return static_cast<uint32_t>(rows[edge_idx].packed[rank] & uint8_t{0x7F});
    } else {
      return static_cast<uint32_t>(rows[edge_idx].packed[rank] & uint16_t{0x7FFF});
    }
  }
  __host__ __device__ __forceinline__
  float get_sign(uint8_t edge_idx, uint8_t rank) const {
    if constexpr (std::is_same_v<PACKED_T, uint8_t>) {
      return (rows[edge_idx].packed[rank] & uint8_t{0x80}) ? -1.0f : +1.0f;
    } else {
      return (rows[edge_idx].packed[rank] & uint16_t{0x8000}) ? -1.0f : +1.0f;
    }
  }
  __device__ __forceinline__
  __nv_bfloat16 get_mag_sq_bf16(uint8_t edge_idx) const {
    return rows[edge_idx].mag_sq;
  }
  __device__ __forceinline__
  float get_mag_sq(uint8_t edge_idx) const {
    return __bfloat162float(rows[edge_idx].mag_sq);
  }

  __host__ void print(std::ostream& out = std::cout, uint32_t len = 0) const {
    if (len == 0) len = N_NEIGHBORS;
    for (uint32_t e = 0; e < len; ++e) {
      const float mag_sq = __bfloat162float(rows[e].mag_sq);
      out << "  edge[" << e << "] mag²=" << mag_sq << ":";
      for (uint32_t r = 0; r < K_RANKS; ++r) {
        const uint16_t p     = rows[e].packed[r];
        if constexpr (std::is_same_v<PACKED_T, uint8_t>) {
          const uint8_t coord = p & uint8_t{0x7F};
          const char    sign  = (p & uint8_t{0x80}) ? '-' : '+';
          out << ' ' << sign << static_cast<uint32_t>(coord);
        } else {
          const uint16_t coord = p & uint16_t{0x7FFF};
          const char     sign  = (p & uint16_t{0x8000}) ? '-' : '+';
          out << ' ' << sign << coord;
        }
      }
      out << '\n';
    }
  }
};
}