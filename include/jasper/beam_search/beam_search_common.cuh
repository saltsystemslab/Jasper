// jasper/beam_search/beam_search_common.cuh
//
// Low-level device helpers shared by every directional-storage beam-search
// variant (cross-polytope LSH, PQ/ADC, ...): the query/candidate exact-scalar
// helper, the single-threaded frontier insertion, and clock-based phase
// profiling. These do not depend on which edge-scoring scheme (estimator) is
// in use — see estimator.cuh / graph_beam_search.cuh for that.
#pragma once

#include <cuda_fp16.h>

#include "jasper/distance/distance.cuh"
#include "jasper/beam_search/entry.cuh"

namespace jasper {

// ── Clock-based phase profiling ───────────────────────────────────────────────
#if defined(JASPER_PROFILE_CLOCKS)

enum PhaseIdx : int {
  PHASE_POP      = 0,
  PHASE_EXACT    = 1,
  PHASE_FRONTIER = 2,
  PHASE_EXPAND   = 3,
  PHASE_ESTIMATE = 4,
  PHASE_SORT     = 5,
  PHASE_DEDUP    = 6,
  PHASE_CLIP     = 7,
  PHASE_COUNT    = 8,
};

extern __device__ uint64_t g_phase_clocks[PHASE_COUNT];

#define CLOCK_START(var) \
  uint64_t var = 0; \
  if (blockIdx.x == 0 && threadIdx.x == 0) { var = clock64(); }

#define CLOCK_ACCUM(var, phase) \
  if (blockIdx.x == 0 && threadIdx.x == 0) { \
    uint64_t _end = clock64(); \
    atomicAdd(reinterpret_cast<unsigned long long*>(&g_phase_clocks[phase]), \
              static_cast<unsigned long long>(_end - (var))); \
  }

inline void print_phase_clocks(double sm_clock_ghz = 1.98) {
  uint64_t h_clocks[PHASE_COUNT];
  cudaMemcpyFromSymbol(h_clocks, g_phase_clocks, sizeof(uint64_t) * PHASE_COUNT);
  const char* names[PHASE_COUNT] = {
    "pop_candidate", "exact_distance", "frontier_insert",
    "expand_neighbors", "estimate_distances",
    "sort_and_merge", "dedup_results", "clip_k"
  };
  uint64_t total = 0;
  for (int i = 0; i < PHASE_COUNT; ++i) total += h_clocks[i];
  printf("\n=== Phase clock breakdown (block 0) ===\n");
  for (int i = 0; i < PHASE_COUNT; ++i) {
    double pct = total > 0 ? 100.0 * h_clocks[i] / total : 0.0;
    double ms  = h_clocks[i] / (sm_clock_ghz * 1e6);
    printf("  %-22s %12llu cycles  %6.1f%%  %.3f ms\n",
           names[i], (unsigned long long)h_clocks[i], pct, ms);
  }
  printf("  %-22s %12llu cycles  100.0%%\n",
         "TOTAL", (unsigned long long)total);
}

inline void reset_phase_clocks() {
  void* ptr = nullptr;
  cudaGetSymbolAddress(&ptr, g_phase_clocks);
  cudaMemset(ptr, 0, sizeof(uint64_t) * PHASE_COUNT);
}

#else
#define CLOCK_START(var)        ((void)0)
#define CLOCK_ACCUM(var, phase) ((void)0)
inline void print_phase_clocks(double = 1.98) {}
inline void reset_phase_clocks() {}
#endif  // JASPER_PROFILE_CLOCKS

// ===== Helper: exact base scalar for u + (for L2) store (q - u) in smem =====
// Single pass over u's vector. Returns, depending on DISTANCE_FUNC:
//   L2            → ||q - u||²          and stores (q - u) into qu_diff
//   INNER_PRODUCT → ⟨q, u⟩ (raw dot)    and leaves qu_diff untouched
// (the IP estimator dots against q directly, so no residual is needed).
// Requires DATA_T == __half and padded_dim divisible by 8.
// vector_view::pad already rounds padded_dim up to ≥16, so the divisibility
// requirement is satisfied for typical configurations.
template <typename DATA_T, uint32_t BLOCK_SIZE, distance_func DISTANCE_FUNC>
__device__ __forceinline__ float compute_qu_diff_and_exact(
    const DATA_T* __restrict__ query_vec,
    const DATA_T* __restrict__ u_vec,
    DATA_T*       __restrict__ qu_diff,
    uint32_t                   padded_dim)
{
  static_assert(std::is_same_v<DATA_T, __half>,
                "compute_qu_diff_and_exact is specialized for __half");
  const uint4* __restrict__ u_v4 = reinterpret_cast<const uint4*>(u_vec);
  const uint4* __restrict__ q_v4 = reinterpret_cast<const uint4*>(query_vec);
  uint4*       __restrict__ d_v4 = reinterpret_cast<uint4*>(qu_diff);
  const uint32_t n_v4 = padded_dim / 8;

  float local = 0.0f;
  for (uint32_t i = threadIdx.x; i < n_v4; i += BLOCK_SIZE) {
    const uint4 u = __ldg(u_v4 + i);
    const uint4 q = q_v4[i];
    uint4 d;

    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      const uint32_t qw = reinterpret_cast<const uint32_t*>(&q)[j];
      const uint32_t uw = reinterpret_cast<const uint32_t*>(&u)[j];
      const __half2  q2 = *reinterpret_cast<const __half2*>(&qw);
      const __half2  u2 = *reinterpret_cast<const __half2*>(&uw);

      if constexpr (DISTANCE_FUNC == distance_func::L2) {
        const __half2  dh = __hsub2(q2, u2);
        reinterpret_cast<uint32_t*>(&d)[j] =
            *reinterpret_cast<const uint32_t*>(&dh);

        const float f0 = __half2float(__low2half (dh));
        const float f1 = __half2float(__high2half(dh));
        local += f0 * f0 + f1 * f1;
      } else {  // INNER_PRODUCT: accumulate ⟨q, u⟩
        const __half2 p2 = __hmul2(q2, u2);
        local += __half2float(__low2half (p2))
               + __half2float(__high2half(p2));
      }
    }
    if constexpr (DISTANCE_FUNC == distance_func::L2) {
      d_v4[i] = d;
    }
  }

  // Intra-warp reduction.
  #pragma unroll
  for (int d = 16; d > 0; d >>= 1)
    local += __shfl_xor_sync(0xFFFFFFFFu, local, d);

  // Inter-warp reduction via one slot per warp.
  constexpr uint32_t N_WARPS = BLOCK_SIZE / 32;
  __shared__ float warp_sum[N_WARPS > 0 ? N_WARPS : 1];
  const uint32_t lane = threadIdx.x & 31u;
  const uint32_t wid  = threadIdx.x >> 5;
  if (lane == 0) warp_sum[wid] = local;
  __syncthreads();
  if (wid == 0) {
    float v = (lane < N_WARPS) ? warp_sum[lane] : 0.0f;
    #pragma unroll
    for (int d = 16; d > 0; d >>= 1)
      v += __shfl_xor_sync(0xFFFFFFFFu, v, d);
    if (lane == 0) warp_sum[0] = v;
  }
  __syncthreads();
  return warp_sum[0];
}

// ===== Helper: insertion-sort a new (u, exact_dist) into the top-k frontier =====
// Single-threaded. Maintains frontier_buffer[0..count) sorted ascending by dist.
__device__ __forceinline__ void frontier_insert_sorted(
    ENTRY_T   new_entry,
    float     new_dist,
    ENTRY_T*  frontier_buffer,
    uint32_t* frontier_count,
    uint32_t  k)
{
  uint32_t fc = *frontier_count;
  if (fc < k) {
    // Append, then shift left until in order.
    int j = static_cast<int>(fc);
    while (j > 0 && get_distance(frontier_buffer[j - 1]) > new_dist) {
      frontier_buffer[j] = frontier_buffer[j - 1];
      --j;
    }
    frontier_buffer[j] = new_entry;
    *frontier_count = fc + 1;
    return;
  }
  // Full: only insert if better than current worst (at position k-1).
  if (new_dist >= get_distance(frontier_buffer[k - 1])) return;
  int j = static_cast<int>(k) - 1;
  while (j > 0 && get_distance(frontier_buffer[j - 1]) > new_dist) {
    frontier_buffer[j] = frontier_buffer[j - 1];
    --j;
  }
  frontier_buffer[j] = new_entry;
}

}  // namespace jasper
