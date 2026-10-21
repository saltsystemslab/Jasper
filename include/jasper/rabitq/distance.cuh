#pragma once

#include <cstdint>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include "jasper/rabitq/layout.cuh"

namespace cg = cooperative_groups;

namespace jasper {

enum MetricType : uint32_t {
  METRIC_L2 = 0,
  METRIC_IP = 1,
};

// <code, q> between one node's packed code and a rotated query that is
// zero-padded to rabitq_data_store::query_padded_dim() and 16B aligned.
// Each thread takes one 32-bit code word (32 / BITS dims) per step, so a tile
// reads consecutive words. Codes are packed LSB-first, as written by
// rabitq_quantize_kernel; pad dims carry code 0 and pad query lanes are 0.
template <uint32_t BITS, typename TileT>
__device__ __forceinline__ float rabitq_code_dot(
    const uint32_t* __restrict__ code_words,
    uint32_t n_words,
    const float* __restrict__ query,
    TileT& tile) {
  static_assert(BITS == 1 || BITS == 2 || BITS == 4 || BITS == 8,
                "RaBitQ codes support 1, 2, 4 or 8 bits per dim");
  constexpr uint32_t DIMS_PER_WORD = 32 / BITS;
  constexpr uint32_t MASK = (1u << BITS) - 1u;

  float acc = 0.0f;
  for (uint32_t w = tile.thread_rank(); w < n_words; w += tile.size()) {
    const uint32_t word = code_words[w];
    const float4* q4 = reinterpret_cast<const float4*>(query + w * DIMS_PER_WORD);
#pragma unroll
    for (uint32_t j = 0; j < DIMS_PER_WORD / 4; j++) {
      const float4 q = q4[j];
      const uint32_t s = 4 * j * BITS;
      acc += static_cast<float>((word >>  s)             & MASK) * q.x;
      acc += static_cast<float>((word >> (s + BITS))     & MASK) * q.y;
      acc += static_cast<float>((word >> (s + 2 * BITS)) & MASK) * q.z;
      acc += static_cast<float>((word >> (s + 3 * BITS)) & MASK) * q.w;
    }
  }
  return cg::reduce(tile, acc, cg::plus<float>());
}

// RaBitQ estimate of the distance between data vector `local_idx` and a query
// (L2: ||o - q||^2, IP: 1 - <o, q>), returned to every thread of the tile:
//   f_add + g_add + f_rescale * (<code, q_r> + c_B * sum(q_r))
// where <code, q_r> + c_B * sum(q_r) = <x_u + c_B, q_r>, the inner product
// with the de-quantized code.
template <uint32_t BITS, typename TileT>
__device__ __forceinline__ float rabitq_estimate_distance(
    const rabitq_data_store<float>& codes,
    uint32_t local_idx,
    const float* __restrict__ rot_query,
    const RabitqQueryFactor& query_factor,
    TileT& tile) {
  const auto node = codes[local_idx];
  const float dot = rabitq_code_dot<BITS>(
      reinterpret_cast<const uint32_t*>(node.bin_code),
      codes.code_bytes() / 4, rot_query, tile);
  return *node.f_add + query_factor.add +
         *node.f_rescale * (dot + query_factor.kBxSumq);
}

}
