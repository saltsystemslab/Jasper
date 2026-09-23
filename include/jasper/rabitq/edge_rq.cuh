// jasper/rabitq/edge_rq.cuh
//
// Per-node RaBitQ codes for its out-edges, plus the (tiny) query-invariant
// metadata the estimator needs. See rq_beam_search.cuh for the estimator this
// feeds and the derivation it implements.
//
// Each directed edge (u -> v) stores a RaBitQ code of the UNIT direction of the
// residual e = v - u, taken in the RQ_DIM-dimensional projected space:
//
//   û = unit( (v-u)[:RQ_DIM] )     <- quantized, RQ_BITS bits per dimension
//   rescale = 1 / <x̄, û>           <- RaBitQ's unbiasedness factor (see
//                                     calc_multi_code in rabitq/encode.cuh)
//   mag_sq  = ||v-u||²             <- EXACT, over the full dimension
//
// Truncating to the first RQ_DIM coordinates IS the random orthogonal
// projection P_d of the RaBitQ ">32x compression" construction, because the
// stored vectors live in a Haar-rotated space (run_query's --rotate /
// create_lsh_index's prerotate). The first d rows of a Haar-random R are
// distributed exactly as a uniform d-frame — right-invariance of Haar plus
// transitivity of O(D) on d-frames — so T_d·R and P_d are the same random
// object, not an approximation of one another. The rotation is what makes the
// axes exchangeable; truncating UNROTATED data would keep d specific semantic
// features and is not a projection at all.
//
// This is also why the projection costs nothing at run time: the rotated
// vectors ARE the stored vectors, and the beam search already has the residual
// (q-u) in shared memory. RQ_DIM must be <= dim.
//
// mag_sq is kept at full dimension rather than in the projected space on
// purpose: the estimator uses the projected space ONLY for the cosine, and
// pairs it with norms it knows exactly (eq. 65 of the RaBitQ appendix).
#pragma once

#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "jasper/rabitq/encode.cuh"

namespace jasper {

template <typename INDEX_T, uint32_t N_NEIGHBORS, uint32_t RQ_DIM, uint32_t RQ_BITS>
struct alignas(4) edge_rq_list {
  static_assert(RQ_BITS == 1 || RQ_BITS == 2 || RQ_BITS == 4,
                "RQ_BITS must be 1, 2 or 4 (so levels never straddle a word)");
  static_assert(RQ_DIM > 0, "RaBitQ edge codes require RQ_DIM > 0");
  static_assert((RQ_DIM * RQ_BITS) % 32u == 0u,
                "RQ_DIM * RQ_BITS must be a whole number of 32-bit words");

  // Dimensions packed per 32-bit word, and words per edge.
  static constexpr uint32_t dims_per_word = 32u / RQ_BITS;
  static constexpr uint32_t n_words       = (RQ_DIM * RQ_BITS) / 32u;
  static constexpr uint32_t level_mask    = (1u << RQ_BITS) - 1u;

  // Word-plane-major SoA layout: code[w * N_NEIGHBORS + edge]. For a fixed
  // plane w a warp of consecutive edges reads consecutive 32-bit words — one
  // coalesced transaction — which an edge-major layout would break up into
  // n_words-strided accesses. Same reasoning as edge_pq_list's subspace-major
  // codes.
  uint32_t      code[N_NEIGHBORS * n_words];
  __half        rescale[N_NEIGHBORS];  // 1 / <x̄, û>; O(1), so fp16 is plenty
  __nv_bfloat16 mag_sq[N_NEIGHBORS];   // ||v-u||²; wide dynamic range -> bf16

  __host__ __device__ __forceinline__
  uint32_t get_word(uint8_t edge_idx, uint32_t w) const {
    return code[w * N_NEIGHBORS + static_cast<uint32_t>(edge_idx)];
  }

  __host__ __device__ __forceinline__
  void set_word(uint8_t edge_idx, uint32_t w, uint32_t word) {
    code[w * N_NEIGHBORS + static_cast<uint32_t>(edge_idx)] = word;
  }

  // Sign-encoded level of dimension `d` (0 <= d < RQ_DIM). At RQ_BITS == 1 this
  // is the sign bit of the residual coordinate (1 == non-negative).
  __host__ __device__ __forceinline__
  uint32_t get_level(uint8_t edge_idx, uint32_t d) const {
    const uint32_t bit = d * RQ_BITS;
    return (get_word(edge_idx, bit >> 5) >> (bit & 31u)) & level_mask;
  }

  __host__ __device__ __forceinline__
  float get_rescale(uint8_t edge_idx) const {
    return __half2float(rescale[edge_idx]);
  }

  __host__ __device__ __forceinline__
  float get_mag_sq(uint8_t edge_idx) const {
    return __bfloat162float(mag_sq[edge_idx]);
  }

  __host__ void print(std::ostream& out = std::cout, uint32_t len = 0) const {
    if (len == 0) len = N_NEIGHBORS;
    for (uint32_t e = 0; e < len; ++e) {
      out << "  edge[" << e << "] mag²=" << get_mag_sq(static_cast<uint8_t>(e))
          << " rescale=" << get_rescale(static_cast<uint8_t>(e)) << " code:";
      for (uint32_t w = 0; w < n_words; ++w)
        out << ' ' << std::hex << get_word(static_cast<uint8_t>(e), w) << std::dec;
      out << '\n';
    }
  }
};

// ── Query-invariant RaBitQ metadata ────────────────────────────────────────
// The analogue of lsh_globals / pq_codebooks for this path, and by far the
// smallest of the three: the estimator's only real constant (cb) is fixed by
// RQ_BITS at compile time, so this carries just the build-side scaling factor
// plus enough description to sanity-check a graph against a search.
template <uint32_t RQ_DIM, uint32_t RQ_BITS>
struct rq_globals {
  float    t_const = 0.0f;      // level-scaling factor; inert at RQ_BITS == 1
  uint32_t rq_dim  = RQ_DIM;
  uint32_t rq_bits = RQ_BITS;

  // Reconstruction offset: x̄[j] = level[j] + cb. Derived, not stored.
  __host__ __device__ static constexpr float cb() {
    return -(static_cast<float>(1u << (RQ_BITS - 1)) - 0.5f);
  }
};

// Mean best-rescale factor over random unit directions — the same quantity
// rabitq_quantize's get_const_scaling_factors computes, but on the host with
// std::mt19937 instead of 1000 blocking cuBLAS/cuRAND calls, so the directional
// build path stays free of both.
//
// At RQ_BITS == 1 every level clamps to 0 regardless of t_const (classic 1-bit
// RaBitQ), so the sampling is skipped entirely.
template <uint32_t RQ_DIM, uint32_t RQ_BITS>
__host__ inline rq_globals<RQ_DIM, RQ_BITS> make_rq_globals(uint64_t seed = 1234) {
  rq_globals<RQ_DIM, RQ_BITS> g{};
  if constexpr (RQ_BITS == 1) return g;

  constexpr int kNSamples = 1000;
  std::mt19937_64 rng(seed);
  std::normal_distribution<float> N(0.0f, 1.0f);

  std::vector<float> v(RQ_DIM);
  double sum = 0.0;
  for (int s = 0; s < kNSamples; ++s) {
    double norm_sq = 0.0;
    for (uint32_t i = 0; i < RQ_DIM; ++i) {
      v[i] = N(rng);
      norm_sq += static_cast<double>(v[i]) * v[i];
    }
    const float inv = static_cast<float>(1.0 / std::sqrt(norm_sq));
    for (uint32_t i = 0; i < RQ_DIM; ++i) v[i] = std::fabs(v[i] * inv);
    sum += best_rescale_factor(v.data(), RQ_DIM, RQ_BITS);
  }
  g.t_const = static_cast<float>(sum / kNSamples);
  return g;
}

}  // namespace jasper
