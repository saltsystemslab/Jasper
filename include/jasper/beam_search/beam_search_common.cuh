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
// With JASPER_PROFILE_CLOCKS, thread 0 of every JASPER_PROFILE_STRIDE-th block
// (one query in 64 by default) adds each phase's clock64() cycles to
// g_phase_clocks, and counts its pops and queries, so the report is cycles
// per pop per phase, averaged over a sample of queries rather than one.
// Cycles are a block's elapsed time, stalls included: with the graph in host
// memory, the PCIe round trip lands in the phase that issues the reads.
// CLOCK_ACCUM ends with a barrier first (in every block, sampled or not), so
// a phase is charged until the whole block is done with it, not just thread
// 0; it must sit where the whole block reaches it. That perturbs the timing a
// little, so profile builds are for the breakdown, not for QPS.
#if defined(JASPER_PROFILE_CLOCKS)

#ifndef JASPER_PROFILE_STRIDE
#define JASPER_PROFILE_STRIDE 64
#endif

enum PhaseIdx : int {
  PHASE_POP      = 0,   // pick the next candidate (+ adaptive test, visited mark)
  PHASE_EXACT    = 1,   // directional: read u's edges, codes and vector; exact d(q,u)
  PHASE_FRONTIER = 2,   // insert u into the top-k (+ visited log)
  PHASE_EXPAND   = 3,   // append u's neighbors to the list
  PHASE_ESTIMATE = 4,   // neighbor distances: estimates, or exact for conventional
  PHASE_UNIQUE   = 5,   // directional step 6b: drop copies of nodes already listed
  PHASE_SORT     = 6,   // merge (directional) / sort (conventional) the list
  PHASE_DEDUP    = 7,   // dedup_results
  PHASE_CLIP     = 8,   // clip_k
  PHASE_RERANK   = 9,   // after the loop, once per query: the re-rank
  PHASE_COUNT    = 10,
  PROFILE_POPS    = PHASE_COUNT,      // pops in sampled blocks
  PROFILE_QUERIES = PHASE_COUNT + 1,  // sampled blocks
  PROFILE_SLOTS   = PHASE_COUNT + 2,
};

extern __device__ uint64_t g_phase_clocks[PROFILE_SLOTS];

#define PROFILE_SAMPLED() (blockIdx.x % JASPER_PROFILE_STRIDE == 0 && threadIdx.x == 0)

#define CLOCK_START(var) \
  uint64_t var = 0; \
  if (PROFILE_SAMPLED()) { var = clock64(); }

#define CLOCK_ACCUM(var, phase) \
  __syncthreads(); \
  if (PROFILE_SAMPLED()) { \
    uint64_t _end = clock64(); \
    atomicAdd(reinterpret_cast<unsigned long long*>(&g_phase_clocks[phase]), \
              static_cast<unsigned long long>(_end - (var))); \
  }

#define PROFILE_COUNT(slot) \
  if (PROFILE_SAMPLED()) { \
    atomicAdd(reinterpret_cast<unsigned long long*>(&g_phase_clocks[slot]), 1ull); \
  }

// One machine-readable line, then a table: cycles per pop for each phase of
// the loop, and the re-rank in cycles per query.
inline void print_phase_clocks() {
  uint64_t h[PROFILE_SLOTS];
  cudaMemcpyFromSymbol(h, g_phase_clocks, sizeof(h));
  const char* names[PHASE_COUNT] = {
    "pop", "exact", "frontier", "expand", "estimate",
    "unique", "sort", "dedup", "clip", "rerank"
  };
  const double pops    = h[PROFILE_POPS] ? double(h[PROFILE_POPS]) : 1.0;
  const double queries = h[PROFILE_QUERIES] ? double(h[PROFILE_QUERIES]) : 1.0;
  uint64_t loop = 0;
  for (int i = 0; i < PHASE_COUNT; ++i) if (i != PHASE_RERANK) loop += h[i];
  printf("phase_profile: queries=%llu pops=%llu cycles_per_pop=%.1f",
         (unsigned long long)h[PROFILE_QUERIES], (unsigned long long)h[PROFILE_POPS],
         loop / pops);
  for (int i = 0; i < PHASE_COUNT; ++i)
    printf(" %s=%.1f", names[i], i == PHASE_RERANK ? h[i] / queries : h[i] / pops);
  printf("\n");
  printf("  %-10s %12s %7s\n", "phase", "cycles/pop", "share");
  for (int i = 0; i < PHASE_COUNT; ++i) {
    if (i == PHASE_RERANK) continue;
    printf("  %-10s %12.1f %6.1f%%\n", names[i], h[i] / pops,
           loop ? 100.0 * h[i] / loop : 0.0);
  }
  printf("  %-10s %12.1f\n", "total", loop / pops);
  if (h[PHASE_RERANK])
    printf("  %-10s %12.1f cycles/query\n", "rerank", h[PHASE_RERANK] / queries);
}

inline void reset_phase_clocks() {
  void* ptr = nullptr;
  cudaGetSymbolAddress(&ptr, g_phase_clocks);
  cudaMemset(ptr, 0, sizeof(uint64_t) * PROFILE_SLOTS);
}

#else
#define CLOCK_START(var)        ((void)0)
#define CLOCK_ACCUM(var, phase) ((void)0)
#define PROFILE_COUNT(slot)     ((void)0)
inline void print_phase_clocks() {}
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
//
// STORE_DIMS caps how much of the residual is written: 0 (the default) stores
// all padded_dim coordinates, any other value stores only the first STORE_DIMS
// of them and qu_diff need only be that long. rq_estimator uses this — it reads
// the residual's first rq_dim coordinates and nothing else, so a full-length
// buffer would cost O(D) of shared memory for an O(d) need. Must be a multiple
// of 8 (the uint4 store granularity).
template <typename DATA_T, uint32_t BLOCK_SIZE, distance_func DISTANCE_FUNC,
          uint32_t STORE_DIMS = 0>
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
    // Each uint4 covers 8 halves, so chunk i lands entirely inside a
    // STORE_DIMS-long buffer exactly when 8i < STORE_DIMS.
    if constexpr (DISTANCE_FUNC == distance_func::L2) {
      if (STORE_DIMS == 0 || i * 8u < STORE_DIMS) d_v4[i] = d;
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
static __device__ __forceinline__ void frontier_insert_sorted(
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
