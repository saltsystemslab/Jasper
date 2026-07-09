// jasper/beam_search/directional_kernels.cuh
#pragma once

#include <cub/cub.cuh>
#include <cooperative_groups.h>

#include "jasper/index/vector.cuh"
#include "jasper/index/graph.cuh"
#include "jasper/beam_search/config.cuh"
#include "jasper/beam_search/entry.cuh"
#include "jasper/beam_search/device_kernels.cuh"
#include "jasper/lsh/lsh_globals.cuh"

namespace cg = cooperative_groups;

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

// ===== Helper: ||q - u||^2 (exact) + store (q - u) in smem, single pass =====
// Requires DATA_T == __half and padded_dim divisible by 8.
// vector_view::pad already rounds padded_dim up to ≥16, so the divisibility
// requirement is satisfied for typical configurations.
template <typename DATA_T, uint32_t BLOCK_SIZE>
__device__ __forceinline__ float compute_qu_diff_and_l2sq(
    const DATA_T* __restrict__ query_vec,
    const DATA_T* __restrict__ u_vec,
    DATA_T*       __restrict__ qu_diff,
    uint32_t                   padded_dim)
{
  static_assert(std::is_same_v<DATA_T, __half>,
                "compute_qu_diff_and_l2sq is specialized for __half");
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
      const __half2  dh = __hsub2(q2, u2);
      reinterpret_cast<uint32_t*>(&d)[j] =
          *reinterpret_cast<const uint32_t*>(&dh);

      const float f0 = __half2float(__low2half (dh));
      const float f1 = __half2float(__high2half(dh));
      local += f0 * f0 + f1 * f1;
    }
    d_v4[i] = d;
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

// Key:   uint32_t sortable distance (IEEE→unsigned ordering)
// Value: uint32_t (visited bit | 31-bit index) — top half of original ENTRY_T
template <typename INDEX_T, typename DISTANCE_T, uint32_t BLOCK_SIZE,
          uint32_t MAX_SEARCH_WIDTH, typename BlockRadixSortT>
__device__ void radix_sort(ENTRY_T *result_buffer,
                           uint32_t *result_buffer_count,
                           typename BlockRadixSortT::TempStorage &temp_storage) {
  uint32_t count = result_buffer_count[0];
  constexpr uint32_t ELEMENTS_PER_THREAD = (MAX_SEARCH_WIDTH - 1) / BLOCK_SIZE + 1;

  uint32_t keys[ELEMENTS_PER_THREAD];
  uint32_t vals[ELEMENTS_PER_THREAD];

  // ─── Load + IEEE→sortable transform ───────────────────────────
  // mask = sign==1 ? 0xFFFFFFFF : 0x80000000  → uint sort == float sort
  #pragma unroll
  for (uint32_t i = 0; i < ELEMENTS_PER_THREAD; ++i) {
    uint32_t element_id = threadIdx.x * ELEMENTS_PER_THREAD + i;
    if (element_id < count) {
      ENTRY_T  e      = result_buffer[element_id];
      uint32_t d_bits = static_cast<uint32_t>(e & 0xFFFFFFFFu);
      uint32_t hi32   = static_cast<uint32_t>(e >> 32);
      uint32_t mask   = static_cast<uint32_t>(static_cast<int32_t>(d_bits) >> 31)
                      | 0x80000000u;
      keys[i] = d_bits ^ mask;
      vals[i] = hi32;
    } else {
      keys[i] = 0xFFFFFFFFu;  // sentinel: sorts to end
      vals[i] = static_cast<uint32_t>((empty_entry() >> 32) & 0xFFFFFFFFu);
    }
  }

  BlockRadixSortT(temp_storage).Sort(keys, vals);

  // ─── Inverse transform + reconstruct ENTRY_T ──────────────────
  #pragma unroll
  for (uint32_t i = 0; i < ELEMENTS_PER_THREAD; ++i) {
    uint32_t element_id = threadIdx.x * ELEMENTS_PER_THREAD + i;
    if (element_id < count) {
      uint32_t s_bits = keys[i];
      uint32_t mask   = (s_bits & 0x80000000u) ? 0x80000000u : 0xFFFFFFFFu;
      uint32_t d_bits = s_bits ^ mask;
      result_buffer[element_id] =
          static_cast<ENTRY_T>(d_bits) |
          (static_cast<ENTRY_T>(vals[i]) << 32);
    }
  }
  __syncthreads();
}

// ===== Helper: estimated ||q - v||^2 for each new neighbor v of u =====
// Implements the estimator from sec. "Query Phase":
//   <q-u, r̂> ≈ ( Σ_r c[r] · sign[r] · (q-u)[coord[r]] ) / norm_denom
//   ||q-v||² ≈ ||q-u||² + mag² − 2 · mag · <q-u, r̂>     where mag = ||v-u||
template <typename GRAPH_CFG>
__device__ __forceinline__ void populate_estimated_distances(
    const typename GRAPH_CFG::data_t* __restrict__ smem_qu_diff,
    float                                          exact_dist_u_sq,
    typename graph<GRAPH_CFG>::device_view&        graph,
    typename GRAPH_CFG::index_t                    u_gid,
    const lsh_globals<GRAPH_CFG::k_ranks>&         globals,
    float                                          inv_norm_denom,
    ENTRY_T*  __restrict__                         result_buffer,
    uint32_t                                       offset,
    uint8_t                                        n_edges)
{
  using INDEX_T  = typename GRAPH_CFG::index_t;
  using row_t    = typename GRAPH_CFG::edge_lsh_list_t::row_t;
  using PACKED_T = typename GRAPH_CFG::packed_t;
  constexpr uint8_t  K_RANKS       = GRAPH_CFG::k_ranks;
  constexpr INDEX_T  INVALID_INDEX = static_cast<INDEX_T>(0x7FFFFFFFu);

  // Packed-word layout (compile-time): sign in the MSB, coord in the low bits.
  // packed_t is uint8_t (7-bit coord) or uint16_t (15-bit coord).
  constexpr PACKED_T SIGN_MASK    = static_cast<PACKED_T>(PACKED_T{1} << (sizeof(PACKED_T) * 8 - 1));
  constexpr PACKED_T COORD_MASK   = static_cast<PACKED_T>(~SIGN_MASK);
  constexpr uint32_t PACKED_BYTES = static_cast<uint32_t>(K_RANKS) * sizeof(PACKED_T);
  // mag_sq follows the packed array, aligned up to alignof(bf16) == 2.
  constexpr uint32_t MAG_OFFSET   = (PACKED_BYTES + 1u) & ~1u;

  // Hoist u-side segment lookup.
  const INDEX_T  u_lid    = u_gid - graph.global_offset;
  const uint32_t u_segid  = static_cast<uint32_t>(u_lid / GRAPH_CFG::vectors_per_segment);
  const uint32_t u_locidx = static_cast<uint32_t>(u_lid % GRAPH_CFG::vectors_per_segment);
  const auto&    u_segment = graph.segments[u_segid];

  // Row stride in bytes. row_t = packed_t packed[K_RANKS] + bf16 mag_sq, with
  // __align__(4) padding up to a 4-byte multiple.
  //   u8/K=4: 4 + 2 → 8 bytes.   u16/K=8: 16 + 2 → 20 bytes.
  const uint8_t* __restrict__ row_base =
      reinterpret_cast<const uint8_t*>(&u_segment.edge_lshs[u_locidx].rows[0]);
  constexpr uint32_t ROW_BYTES = sizeof(row_t);

  // c_per_rank in registers.
  float c_reg[K_RANKS];
  #pragma unroll
  for (uint8_t r = 0; r < K_RANKS; ++r) c_reg[r] = globals.c_per_rank[r];

  for (uint32_t e = threadIdx.x; e < n_edges; e += blockDim.x) {
    ENTRY_T entry = result_buffer[offset + e];
    INDEX_T v     = static_cast<INDEX_T>(get_index(entry));
    if (v == INVALID_INDEX) continue;

    const uint8_t* row_ptr = row_base + e * ROW_BYTES;

    // ─── Load K_RANKS packed words + mag_sq from the same row ──
    PACKED_T p[K_RANKS];
    if constexpr (PACKED_BYTES % 4u == 0u) {
      // Packed array is a whole number of 32-bit words → vectorize the loads
      // (row_ptr is 4-byte aligned via row_t's __align__(4)).
      constexpr uint32_t N_WORDS = PACKED_BYTES / 4u;
      const uint32_t* __restrict__ p32 = reinterpret_cast<const uint32_t*>(row_ptr);
      uint32_t words[N_WORDS];
      #pragma unroll
      for (uint32_t w = 0; w < N_WORDS; ++w) words[w] = __ldg(p32 + w);
      #pragma unroll
      for (uint32_t r = 0; r < K_RANKS; ++r) {
        if constexpr (sizeof(PACKED_T) == 1u) {
          p[r] = static_cast<PACKED_T>((words[r >> 2] >> ((r & 3u) * 8u)) & 0xFFu);
        } else {
          p[r] = static_cast<PACKED_T>((words[r >> 1] >> ((r & 1u) * 16u)) & 0xFFFFu);
        }
      }
    } else {
      // Odd packed size (e.g. uint8_t with K_RANKS not a multiple of 4): load
      // each packed word directly.
      const PACKED_T* __restrict__ pp = reinterpret_cast<const PACKED_T*>(row_ptr);
      #pragma unroll
      for (uint8_t r = 0; r < K_RANKS; ++r) p[r] = __ldg(pp + r);
    }

    // mag_sq sits after the packed array (MAG_OFFSET accounts for sizeof(packed_t)
    // and bf16 alignment padding).
    const __nv_bfloat16 mag_sq_bf = __ldg(
        reinterpret_cast<const __nv_bfloat16*>(row_ptr + MAG_OFFSET));
    const float mag_sq = __bfloat162float(mag_sq_bf);

    // ─── Estimator ───
    float dot_acc = 0.0f;
    #pragma unroll
    for (uint8_t r = 0; r < K_RANKS; ++r) {
      const PACKED_T word  = p[r];
      const uint32_t coord = static_cast<uint32_t>(word & COORD_MASK);
      const float    sgn   = (word & SIGN_MASK) ? -1.0f : +1.0f;
      const float    diff  = static_cast<float>(smem_qu_diff[coord]);
      dot_acc += c_reg[r] * sgn * diff;
    }
    const float dot_est = dot_acc * inv_norm_denom;
    const float mag     = sqrtf(mag_sq);
    const float est_sq  = exact_dist_u_sq + mag_sq - 2.0f * mag * dot_est;

    result_buffer[offset + e] = set_distance(entry, est_sq);
  }
  __syncthreads();
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

// ===== The kernel =====
template <typename GRAPH_CFG, uint32_t BLOCK_SIZE,
          uint32_t MAX_SEARCH_WIDTH, bool GET_VISITED,
          uint32_t TILE_SIZE = 4, uint32_t MAX_RESULT_SIZE = 1024>
__global__ void directional_beam_search_kernel(
    typename graph<GRAPH_CFG>::device_view graph,
    lsh_globals<GRAPH_CFG::k_ranks>        globals,
    thrust::pair<typename GRAPH_CFG::index_t, typename GRAPH_CFG::distance_t>* frontier_results,
    thrust::pair<typename GRAPH_CFG::index_t, typename GRAPH_CFG::distance_t>* visited_results,
    uint32_t*                              visited_counts,
    vector_view<typename GRAPH_CFG::data_t> query_vectors,
    bool                                   use_range,
    uint32_t                               query_start,
    uint32_t                               query_end,
    typename GRAPH_CFG::index_t            medoid,
    uint32_t                               k,
    uint32_t                               beam_width,
    uint32_t                               limit)
{
  using INDEX_T    = typename GRAPH_CFG::index_t;
  using DATA_T     = typename GRAPH_CFG::data_t;
  using DISTANCE_T = typename GRAPH_CFG::distance_t;
  static_assert(GRAPH_CFG::use_lsh,
                "directional_beam_search_kernel requires graph_cfg::use_lsh");

  const auto      query_id   = blockIdx.x;
  const uint32_t  padded_dim = graph.get_padded_dim();
  constexpr INDEX_T INVALID_INDEX = static_cast<INDEX_T>(0x7FFFFFFFu);

  assert(beam_width + GRAPH_CFG::n_neighbors <= MAX_SEARCH_WIDTH);

  // ---- Shared memory layout ----
  //   result_buffer[beam_width+n_neighbors]    ENTRY_T   (candidate buffer, estimated dists)
  //   result_buffer_count             uint32_t
  //   frontier_buffer[k]              ENTRY_T   (top-k by EXACT dist, sorted asc)
  //   frontier_buffer_count           uint32_t
  //   smem_query_vec[padded_dim]      DATA_T
  //   smem_qu_diff[padded_dim]        DATA_T    (q - u for current explored u)
  const uint32_t result_buffer_size = beam_width + GRAPH_CFG::n_neighbors;
  extern __shared__ __align__(16) unsigned char smem_raw[];
  unsigned char* p = smem_raw;

  auto* __restrict__ result_buffer       = reinterpret_cast<ENTRY_T*>(p);
  p += result_buffer_size * sizeof(ENTRY_T);
  auto* __restrict__ result_buffer_count = reinterpret_cast<uint32_t*>(p);
  p += sizeof(uint32_t);

  p = reinterpret_cast<unsigned char*>(
        (reinterpret_cast<uintptr_t>(p) + 7) & ~uintptr_t(7));
  auto* __restrict__ frontier_buffer       = reinterpret_cast<ENTRY_T*>(p);
  p += k * sizeof(ENTRY_T);
  auto* __restrict__ frontier_buffer_count = reinterpret_cast<uint32_t*>(p);
  p += sizeof(uint32_t);

  p = reinterpret_cast<unsigned char*>(
      (reinterpret_cast<uintptr_t>(p) + 7) & ~uintptr_t(7));
  ENTRY_T* __restrict__ merge_scratch = reinterpret_cast<ENTRY_T*>(p);
  p += GRAPH_CFG::n_neighbors * sizeof(ENTRY_T);

  p = reinterpret_cast<unsigned char*>(
        (reinterpret_cast<uintptr_t>(p) + 15) & ~uintptr_t(15));
  DATA_T* __restrict__ smem_query_vec = reinterpret_cast<DATA_T*>(p);
  p += padded_dim * sizeof(DATA_T);
  DATA_T* __restrict__ smem_qu_diff   = reinterpret_cast<DATA_T*>(p);

  // ---- Load query vector ----
  DATA_T* query_vec_src = use_range ? graph.get_vector(query_start + query_id)
                                    : query_vectors[query_id];
  for (uint32_t i = threadIdx.x; i < padded_dim; i += BLOCK_SIZE)
    smem_query_vec[i] = query_vec_src[i];

  // ---- CUB temp storage for candidate sort/dedup ----
  // constexpr uint32_t ELEMENTS_PER_THREAD = (MAX_SEARCH_WIDTH - 1) / BLOCK_SIZE + 1;
  // using BlockMergeSortT = cub::BlockMergeSort<ENTRY_T, BLOCK_SIZE, ELEMENTS_PER_THREAD>;
  // using BlockScanT      = cub::BlockScan<uint32_t, BLOCK_SIZE>;
  // union TempStorage {
  //   typename BlockMergeSortT::TempStorage sort_storage;
  //   typename BlockScanT::TempStorage      scan_storage;
  // };
  // __shared__ TempStorage temp_storage;
  // constexpr uint32_t ELEMENTS_PER_THREAD = (MAX_SEARCH_WIDTH - 1) / BLOCK_SIZE + 1;
  // using BlockRadixSortT = cub::BlockRadixSort<uint32_t, BLOCK_SIZE,
  //                                             ELEMENTS_PER_THREAD, uint32_t>;
  // using SmallSortT = cub::BlockRadixSort<uint32_t, GRAPH_CFG::n_neighbors, 1, uint32_t>;
  // using BlockScanT      = cub::BlockScan<uint32_t, BLOCK_SIZE>;
  // union TempStorage {
  //   typename BlockRadixSortT::TempStorage sort_storage;
  //   typename BlockScanT::TempStorage      scan_storage;
  // };
  // __shared__ TempStorage temp_storage;
  constexpr uint32_t TAIL_EPT =
    (GRAPH_CFG::n_neighbors + BLOCK_SIZE - 1) / BLOCK_SIZE;
  using TailSortT  = cub::BlockMergeSort<ENTRY_T, BLOCK_SIZE, TAIL_EPT>;
  using BlockScanT = cub::BlockScan<uint32_t, BLOCK_SIZE>;
  union TempStorage {
    typename TailSortT::TempStorage  tail_sort_storage;
    typename BlockScanT::TempStorage scan_storage;
  };
  __shared__ TempStorage temp_storage;

  // ---- Initialize buffers; seed candidate list with medoid ----
  for (uint32_t i = threadIdx.x; i < result_buffer_size; i += BLOCK_SIZE)
    result_buffer[i] = empty_entry();
  if (threadIdx.x == 0) {
    // Medoid: estimated distance 0 so it pops first (real dist computed on pop).
    result_buffer[0]         = set_distance(set_index(empty_entry(), medoid), 0.0f);
    result_buffer_count[0]   = 1;
    frontier_buffer_count[0] = 0;
  }
  const float inv_norm_denom = 1.0f / globals.norm_denom;
  __syncthreads();

  uint32_t visited_counter = 0;
  uint32_t loop_count      = 0;

  // ---- Main loop ----
  while (loop_count <= limit) {
    ++loop_count;

    // 1) Pop next candidate (lowest estimated distance, not yet visited).
    __syncthreads();
    CLOCK_START(t_pop);
    auto [frontierIdx, found] =
        choose_new_frontier<INDEX_T, DISTANCE_T, BLOCK_SIZE, MAX_SEARCH_WIDTH>(
            result_buffer, result_buffer_count);
    CLOCK_ACCUM(t_pop, PHASE_POP);
    if (!found) break;

    // 2) Mark visited in candidate buffer (so we never pop it again).
    if constexpr (BLOCK_SIZE > 33) {
      if (threadIdx.x == 33)
        result_buffer[frontierIdx] = set_visited(result_buffer[frontierIdx]);
    } else {
      if (threadIdx.x == 1)
        result_buffer[frontierIdx] = set_visited(result_buffer[frontierIdx]);
    }
    __syncthreads();

    const INDEX_T u_gid =
        static_cast<INDEX_T>(get_index(result_buffer[frontierIdx]));

    // 3) Fetch u's full vector. Compute (q-u) into smem and EXACT ||q-u||² in
    //    one pass. This is the ONE I/O per explored node referenced in the paper.
    CLOCK_START(t_exact);
    DATA_T* u_vec = graph.get_vector(u_gid);
    const float exact_dist_u_sq =
        compute_qu_diff_and_l2sq<DATA_T, BLOCK_SIZE>(
            smem_query_vec, u_vec, smem_qu_diff, padded_dim);
    CLOCK_ACCUM(t_exact, PHASE_EXACT);

    // 4) Add u to frontier (top-k explored by EXACT dist) and visited log.
    CLOCK_START(t_frontier);
    if (threadIdx.x == 0) {
      ENTRY_T u_entry =
          set_distance(set_index(empty_entry(), u_gid), exact_dist_u_sq);
      frontier_insert_sorted(u_entry, exact_dist_u_sq,
                             frontier_buffer, frontier_buffer_count, k);

      if (GET_VISITED) {
        visited_results[query_id * MAX_RESULT_SIZE + visited_counter].first  = u_gid;
        visited_results[query_id * MAX_RESULT_SIZE + visited_counter].second =
            static_cast<DISTANCE_T>(exact_dist_u_sq);
      }
    }
    ++visited_counter;
    if (visited_counter == MAX_RESULT_SIZE) break;
    __syncthreads();
    CLOCK_ACCUM(t_frontier, PHASE_FRONTIER);

    // 5) Append u's neighbors to candidates as placeholders (no neighbor I/O).
    CLOCK_START(t_expand);
    const uint32_t offset = result_buffer_count[0];
    __syncthreads();
    add_frontier_out<GRAPH_CFG>(graph, result_buffer, result_buffer_count,
                                u_gid, k, beam_width);
    __syncthreads();
    CLOCK_ACCUM(t_expand, PHASE_EXPAND);

    // 6) ESTIMATE ||q-v||² for the newly-appended neighbors using the LSH info
    //    stored on edges (u, v) — no fetch of v's vector required.
    CLOCK_START(t_estimate);
    const uint8_t n_edges = graph.get_edge_count(u_gid);
    populate_estimated_distances<GRAPH_CFG>(
        smem_qu_diff, exact_dist_u_sq, graph, u_gid, globals, inv_norm_denom,
        result_buffer, offset, n_edges);
    CLOCK_ACCUM(t_estimate, PHASE_ESTIMATE);

    // 7) Sort / dedup / clip the candidate buffer (estimated distances).
    // merge_sort<INDEX_T, DISTANCE_T, BLOCK_SIZE, MAX_SEARCH_WIDTH, BlockMergeSortT>(
    //     result_buffer, result_buffer_count, temp_storage.sort_storage);
    // radix_sort<INDEX_T, DISTANCE_T, BLOCK_SIZE, MAX_SEARCH_WIDTH, BlockRadixSortT>(
    //   result_buffer, result_buffer_count, temp_storage.sort_storage);
    CLOCK_START(t_sort);
    merge_sort_tail<INDEX_T, DISTANCE_T, BLOCK_SIZE, MAX_SEARCH_WIDTH,
                GRAPH_CFG::n_neighbors, TailSortT>(
      result_buffer, result_buffer_count, /*head_len=*/offset,
      merge_scratch, temp_storage.tail_sort_storage);
    CLOCK_ACCUM(t_sort, PHASE_SORT);

    CLOCK_START(t_dedup);
    dedup_results<INDEX_T, DISTANCE_T, BLOCK_SIZE, MAX_SEARCH_WIDTH, BlockScanT>(
        result_buffer, result_buffer_count, temp_storage.scan_storage);
    CLOCK_ACCUM(t_dedup, PHASE_DEDUP);

    CLOCK_START(t_clip);
    clip_k(result_buffer_count, beam_width);
    CLOCK_ACCUM(t_clip,  PHASE_CLIP);
  }

  // ---- Write top-k from frontier_buffer (already sorted ascending),
  //      skipping any soft-deleted vertices. ----
  __syncthreads();
  if (threadIdx.x == 0) {
    const uint32_t fc = frontier_buffer_count[0];
    uint32_t out = 0;
    for (uint32_t i = 0; i < fc && out < k; i++) {
      uint32_t idx = get_index(frontier_buffer[i]);
      if (!graph.is_valid(idx)) continue;
      if (graph.is_deleted(idx)) continue;
      frontier_results[query_id * k + out].first  = static_cast<INDEX_T>(idx);
      frontier_results[query_id * k + out].second =
          static_cast<DISTANCE_T>(get_distance(frontier_buffer[i]));
      out++;
    }
    for (uint32_t i = out; i < k; i++) {
      frontier_results[query_id * k + i].first  = INVALID_INDEX;
      frontier_results[query_id * k + i].second =
          std::numeric_limits<DISTANCE_T>::max();
    }
  }
  if (threadIdx.x == 0 && GET_VISITED) {
    visited_counts[query_id] = visited_counter;
  }
}

// ===== Host: smem size =====
template <typename GRAPH_CFG>
__host__ inline uint32_t get_directional_smem_size(
    uint32_t beam_width, uint32_t k, uint32_t padded_dim)
{
  using DATA_T = typename GRAPH_CFG::data_t;
  uint32_t s = 0;
  s += sizeof(ENTRY_T) * (beam_width + GRAPH_CFG::n_neighbors); // result_buffer
  s += sizeof(uint32_t);                                         // result_buffer_count
  s = (s + 7) & ~7u;
  s += sizeof(ENTRY_T) * k;                                      // frontier_buffer
  s += sizeof(uint32_t);                                         // frontier_buffer_count
  s = (s + 7) & ~7u;
  s += sizeof(ENTRY_T) * GRAPH_CFG::n_neighbors;                 // merge_scratch  (NEW)
  s = (s + 15) & ~15u;
  s += sizeof(DATA_T) * padded_dim;                              // smem_query_vec
  s += sizeof(DATA_T) * padded_dim;                              // smem_qu_diff
  return s;
}

// ===== Host: launcher =====
template <typename Cfg>
__host__ beam_search_result<typename Cfg::graph_cfg_t>
directional_beam_search(
    const beam_search_params<Cfg>&                p,
    const lsh_globals<Cfg::graph_cfg_t::k_ranks>& globals,
    cudaStream_t                                  stream = 0)
{
  using graph_cfg_t = typename Cfg::graph_cfg_t;
  using entry_t     = typename Cfg::entry_t;
  static_assert(graph_cfg_t::use_lsh,
                "directional_beam_search requires graph_cfg::use_lsh");

  const uint32_t n_query_vectors = p.use_range
      ? (p.query_end - p.query_start)
      : p.query_vectors.n_vectors;

  dim3 threads(Cfg::block_size, 1, 1);
  dim3 blocks (n_query_vectors, 1, 1);

  const uint32_t padded_dim = p.graph.get_padded_dim();
  const uint32_t smem = get_directional_smem_size<graph_cfg_t>(
      p.beam_width, p.k, padded_dim);

  beam_search_result<graph_cfg_t> result{};
  cudaMalloc(&result.frontier, sizeof(entry_t) * n_query_vectors * p.k);
  if constexpr (Cfg::get_visited) {
    cudaMalloc(&result.visited,
               sizeof(entry_t) * n_query_vectors * Cfg::max_result_size);
    cudaMalloc(&result.visited_counts, sizeof(uint32_t) * n_query_vectors);
  }

  auto kernel = directional_beam_search_kernel<
      graph_cfg_t, Cfg::block_size,
      Cfg::max_search_width, Cfg::get_visited,
      Cfg::tile_size, Cfg::max_result_size>;

  if (smem > 48 * 1024) {
    cudaFuncSetAttribute(kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  }

  kernel<<<blocks, threads, smem, stream>>>(
      p.graph.view(), globals,
      result.frontier, result.visited, result.visited_counts,
      p.query_vectors, p.use_range, p.query_start, p.query_end,
      p.medoid, p.k, p.beam_width, p.limit);

  cudaError_t err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    std::cerr << "Directional beam search kernel launch failed: "
              << cudaGetErrorString(err) << std::endl;
  }
  return result;
}

// Variant with pre-allocated result buffer, matching beam_search.cuh.
template <typename Cfg>
__host__ void directional_beam_search(
    const beam_search_params<Cfg>&                  p,
    const lsh_globals<Cfg::graph_cfg_t::k_ranks>&   globals,
    beam_search_result<typename Cfg::graph_cfg_t>   result,
    cudaStream_t                                    stream = 0)
{
  using graph_cfg_t = typename Cfg::graph_cfg_t;
  static_assert(graph_cfg_t::use_lsh,
                "directional_beam_search requires graph_cfg::use_lsh");

  const uint32_t n_query_vectors = p.use_range
      ? (p.query_end - p.query_start)
      : p.query_vectors.n_vectors;

  dim3 threads(Cfg::block_size, 1, 1);
  dim3 blocks (n_query_vectors, 1, 1);

  const uint32_t padded_dim = p.graph.get_padded_dim();
  const uint32_t smem = get_directional_smem_size<graph_cfg_t>(
      p.beam_width, p.k, padded_dim);

  auto kernel = directional_beam_search_kernel<
      graph_cfg_t, Cfg::block_size,
      Cfg::max_search_width, Cfg::get_visited,
      Cfg::tile_size, Cfg::max_result_size>;

  if (smem > 48 * 1024) {
    cudaFuncSetAttribute(kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  }

  kernel<<<blocks, threads, smem, stream>>>(
      p.graph.view(), globals,
      result.frontier, result.visited, result.visited_counts,
      p.query_vectors, p.use_range, p.query_start, p.query_end,
      p.medoid, p.k, p.beam_width, p.limit);

  cudaError_t err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    std::cerr << "Directional beam search kernel launch failed: "
              << cudaGetErrorString(err) << std::endl;
  }
}

}  // namespace jasper