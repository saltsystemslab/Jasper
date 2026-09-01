#pragma once

#include <cuda_bf16.h>
#include <cstdint>
#include <type_traits>

namespace jasper {

template <typename INDEX_T, uint32_t N_NEIGHBORS, uint32_t K_RANKS, typename PACKED_T>
struct alignas(4) edge_lsh_list {

  static_assert(
    std::is_same_v<PACKED_T, uint8_t> || std::is_same_v<PACKED_T, uint16_t>,
    "PACKED_T must be uint8_t or uint16_t"
  );

  // SoA layout:
  //   packed[edge * K_RANKS + rank] — LSH words (sign bit in MSB, coord in low bits)
  //   mag_sq[edge]                  — ||v-u||² as bf16
  // Splitting mag_sq out of the packed words removes the per-row alignment
  // padding the old interleaved row_t needed (the 2-byte bf16 forced each row up
  // to a 4-byte multiple). Each edge's K_RANKS packed words still start at
  // packed[edge*K_RANKS], a 4-byte-aligned offset because K_RANKS*sizeof(PACKED_T)
  // is a multiple of 4 for every supported config, and alignas(4) keeps the base
  // aligned — so the estimator's vectorized 32-bit loads remain valid.
  PACKED_T      packed[N_NEIGHBORS * K_RANKS];
  __nv_bfloat16 mag_sq[N_NEIGHBORS];

  __host__ __device__ __forceinline__
  PACKED_T get_packed(uint8_t edge_idx, uint8_t rank) const {
    return packed[static_cast<uint32_t>(edge_idx) * K_RANKS + rank];
  }
  __host__ __device__ __forceinline__
  uint32_t get_coord(uint8_t edge_idx, uint8_t rank) const {
    const PACKED_T w = get_packed(edge_idx, rank);
    if constexpr (std::is_same_v<PACKED_T, uint8_t>) {
      return static_cast<uint32_t>(w & uint8_t{0x7F});
    } else {
      return static_cast<uint32_t>(w & uint16_t{0x7FFF});
    }
  }
  __host__ __device__ __forceinline__
  float get_sign(uint8_t edge_idx, uint8_t rank) const {
    const PACKED_T w = get_packed(edge_idx, rank);
    if constexpr (std::is_same_v<PACKED_T, uint8_t>) {
      return (w & uint8_t{0x80}) ? -1.0f : +1.0f;
    } else {
      return (w & uint16_t{0x8000}) ? -1.0f : +1.0f;
    }
  }
  __device__ __forceinline__
  __nv_bfloat16 get_mag_sq_bf16(uint8_t edge_idx) const {
    return mag_sq[edge_idx];
  }
  __device__ __forceinline__
  float get_mag_sq(uint8_t edge_idx) const {
    return __bfloat162float(mag_sq[edge_idx]);
  }

  __host__ void print(std::ostream& out = std::cout, uint32_t len = 0) const {
    if (len == 0) len = N_NEIGHBORS;
    for (uint32_t e = 0; e < len; ++e) {
      const float m = __bfloat162float(mag_sq[e]);
      out << "  edge[" << e << "] mag²=" << m << ":";
      for (uint32_t r = 0; r < K_RANKS; ++r) {
        const PACKED_T p = get_packed(static_cast<uint8_t>(e), static_cast<uint8_t>(r));
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
