// jasper/rabitq/encode.cuh
//
// The RaBitQ encoding primitives, independent of what is being encoded.
//
// Split out of quantize.cuh so both quantizers can share one implementation of
// the actual code construction:
//   - quantize.cuh          — whole vectors against a global centroid (IVF-style)
//   - rabitq/rq_kernels.cuh — per-edge residuals against the source node u
//                             (directional storage; see rq_beam_search.cuh)
// Nothing here touches cuBLAS/cuRAND, so the directional path can include it
// without pulling those into graph.cuh.
#pragma once

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <queue>
#include <vector>

namespace cg = cooperative_groups;

namespace jasper {

constexpr std::array<float, 9> kTightStart = {
    0, 0.15, 0.20, 0.52, 0.59, 0.71, 0.75, 0.77, 0.81
};

__host__ inline double best_rescale_factor(
    const float* o_abs, size_t dim, size_t ex_bits) {
  constexpr double kEps = 1e-5;
  constexpr int kNEnum = 10;
  double max_o = *std::max_element(o_abs, o_abs + dim);

  double t_end = static_cast<double>(((1 << ex_bits) - 1) + kNEnum) / max_o;
  double t_start = t_end * kTightStart[ex_bits];

  std::vector<int> cur_o_bar(dim);
  double sqr_denominator = static_cast<double>(dim) * 0.25;
  double numerator = 0;

  for (size_t i = 0; i < dim; ++i) {
    int cur = static_cast<int>((t_start * o_abs[i]) + kEps);
    cur_o_bar[i] = cur;
    sqr_denominator += cur * cur + cur;
    numerator += (cur + 0.5) * o_abs[i];
  }

  std::priority_queue<std::pair<double, size_t>,
                      std::vector<std::pair<double, size_t>>,
                      std::greater<>>
      next_t;

  for (size_t i = 0; i < dim; ++i) {
    next_t.emplace(static_cast<double>(cur_o_bar[i] + 1) / o_abs[i], i);
  }

  double max_ip = 0;
  double t = 0;

  while (!next_t.empty()) {
    double cur_t = next_t.top().first;
    size_t update_id = next_t.top().second;
    next_t.pop();

    cur_o_bar[update_id]++;
    int update_o_bar = cur_o_bar[update_id];
    sqr_denominator += 2 * update_o_bar;
    numerator += o_abs[update_id];

    double cur_ip = numerator / std::sqrt(sqr_denominator);
    if (cur_ip > max_ip) {
      max_ip = cur_ip;
      t = cur_t;
    }

    if (update_o_bar < (1 << ex_bits) - 1) {
      double t_next = static_cast<double>(update_o_bar + 1) / o_abs[update_id];
      if (t_next < t_end) {
        next_t.emplace(t_next, update_id);
      }
    }
  }
  return t;
}

// ═══════════════════════════════════════════════════════════════
// Device: quantize one (unnormalized) vector into a RaBitQ code
// ═══════════════════════════════════════════════════════════════

// residual_vec and uncompressed_code live in shared memory, both `dim` long.
// `l2_norm` is ||residual_vec||, used to normalize on the fly, so the code is
// built for the UNIT vector ô = residual_vec / l2_norm.
//
// On return uncompressed_code[j] holds the sign-encoded level of dimension j,
// and the returned scalar is 1 / <x̄, ô> where x̄[j] = code[j] + cb is the
// unnormalized reconstruction (cb = -(2^(bits-1) - 0.5)). That reciprocal is
// what makes the RaBitQ estimator <ô, y> ≈ <x̄, y> / <x̄, ô> unbiased, so every
// caller stores it alongside the code.
//
// At SIZE_PER_DIM == 1 every level clamps to 0 and the code degenerates to the
// sign bits of the residual — i.e. classic 1-bit RaBitQ, where `t_const` has no
// effect at all.
template <uint32_t SIZE_PER_DIM, uint16_t TILE_SIZE, typename ParentT>
__device__ float calc_multi_code(
    float* residual_vec,
    uint8_t* uncompressed_code,
    uint32_t dim,
    double t_const,
    float l2_norm,
    cg::thread_block_tile<TILE_SIZE, ParentT>& tile) {

  constexpr double kEps = 1e-5;

  float ip_norm_tmp = 0;
  for (uint32_t i = tile.thread_rank(); i < dim; i += tile.size()) {
    float abs_o = fabsf(residual_vec[i] / l2_norm);
    int val = static_cast<int>((t_const * abs_o) + kEps);
    if (val >= (1 << (SIZE_PER_DIM - 1))) {
      val = (1 << (SIZE_PER_DIM - 1)) - 1;
    }
    uncompressed_code[i] = static_cast<uint8_t>(val);
    ip_norm_tmp += (val + 0.5f) * abs_o;
  }
  float ip_norm = cg::reduce(tile, ip_norm_tmp, cg::plus<float>());
  float ip_norm_inv = 1.0f / ip_norm;
  if (isnan(ip_norm_inv)) ip_norm_inv = 1.0f;

  // Apply sign encoding
  uint32_t const mask = (1 << (SIZE_PER_DIM - 1)) - 1;
  for (uint32_t i = tile.thread_rank(); i < dim; i += tile.size()) {
    if (residual_vec[i] >= 0) {
      uncompressed_code[i] += 1 << (SIZE_PER_DIM - 1);
    } else {
      uncompressed_code[i] = (~uncompressed_code[i]) & mask;
    }
  }
  tile.sync();

  return ip_norm_inv;
}

}  // namespace jasper
