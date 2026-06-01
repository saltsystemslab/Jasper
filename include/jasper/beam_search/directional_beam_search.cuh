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
  constexpr uint8_t K_RANKS       = GRAPH_CFG::k_ranks;
  constexpr INDEX_T INVALID_INDEX = static_cast<INDEX_T>(0x7FFFFFFFu);

  // Hoist u-side segment lookup.
  const INDEX_T  u_lid    = u_gid - graph.global_offset;
  const uint32_t u_segid  = static_cast<uint32_t>(u_lid / GRAPH_CFG::vectors_per_segment);
  const uint32_t u_locidx = static_cast<uint32_t>(u_lid % GRAPH_CFG::vectors_per_segment);
  const auto&    u_segment = graph.segments[u_segid];

  // Row stride in uint16 units. For K=4: sizeof(row_t)/2 = 12/2 = 6.
  // For K=8: 20/2 = 10.
  const uint16_t* __restrict__ row_base =
      reinterpret_cast<const uint16_t*>(&u_segment.edge_lshs[u_locidx].rows[0]);
  constexpr uint32_t ROW_U16_STRIDE = sizeof(row_t) / sizeof(uint16_t);

  // c_per_rank in registers.
  float c_reg[K_RANKS];
  #pragma unroll
  for (uint8_t r = 0; r < K_RANKS; ++r) c_reg[r] = globals.c_per_rank[r];

  for (uint32_t e = threadIdx.x; e < n_edges; e += blockDim.x) {
    ENTRY_T entry = result_buffer[offset + e];
    INDEX_T v     = static_cast<INDEX_T>(get_index(entry));
    if (v == INVALID_INDEX) continue;

    const uint16_t* row_ptr = row_base + e * ROW_U16_STRIDE;

    // ─── Load K_RANKS LSH words + mag_sq from same 32B cache sector ──
    uint16_t p[K_RANKS];
    if constexpr (K_RANKS == 4) {
      const uint32_t lsh_lo = __ldg(reinterpret_cast<const uint32_t*>(row_ptr));
      const uint32_t lsh_hi = __ldg(reinterpret_cast<const uint32_t*>(row_ptr) + 1);
      p[0] = static_cast<uint16_t>( lsh_lo        & 0xFFFFu);
      p[1] = static_cast<uint16_t>((lsh_lo >> 16) & 0xFFFFu);
      p[2] = static_cast<uint16_t>( lsh_hi        & 0xFFFFu);
      p[3] = static_cast<uint16_t>((lsh_hi >> 16) & 0xFFFFu);
    } else if constexpr (K_RANKS == 8) {
      const uint32_t* p32 = reinterpret_cast<const uint32_t*>(row_ptr);
      const uint32_t w0 = __ldg(p32 + 0);
      const uint32_t w1 = __ldg(p32 + 1);
      const uint32_t w2 = __ldg(p32 + 2);
      const uint32_t w3 = __ldg(p32 + 3);
      p[0] = static_cast<uint16_t>( w0        & 0xFFFFu);
      p[1] = static_cast<uint16_t>((w0 >> 16) & 0xFFFFu);
      p[2] = static_cast<uint16_t>( w1        & 0xFFFFu);
      p[3] = static_cast<uint16_t>((w1 >> 16) & 0xFFFFu);
      p[4] = static_cast<uint16_t>( w2        & 0xFFFFu);
      p[5] = static_cast<uint16_t>((w2 >> 16) & 0xFFFFu);
      p[6] = static_cast<uint16_t>( w3        & 0xFFFFu);
      p[7] = static_cast<uint16_t>((w3 >> 16) & 0xFFFFu);
    } else {
      #pragma unroll
      for (uint8_t r = 0; r < K_RANKS; ++r) p[r] = __ldg(row_ptr + r);
    }

    // mag_sq sits immediately after the K_RANKS LSH words.
    const __nv_bfloat16 mag_sq_bf = __ldg(
        reinterpret_cast<const __nv_bfloat16*>(row_ptr + K_RANKS));
    const float mag_sq = __bfloat162float(mag_sq_bf);

    // ─── Estimator ───
    float dot_acc = 0.0f;
    #pragma unroll
    for (uint8_t r = 0; r < K_RANKS; ++r) {
      const uint16_t word  = p[r];
      const uint16_t coord = word & uint16_t{0x7FFF};
      const float    sgn   = (word & uint16_t{0x8000}) ? -1.0f : +1.0f;
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

  assert(beam_width + 64 <= MAX_SEARCH_WIDTH);

  // ---- Shared memory layout ----
  //   result_buffer[beam_width+64]    ENTRY_T   (candidate buffer, estimated dists)
  //   result_buffer_count             uint32_t
  //   frontier_buffer[k]              ENTRY_T   (top-k by EXACT dist, sorted asc)
  //   frontier_buffer_count           uint32_t
  //   smem_query_vec[padded_dim]      DATA_T
  //   smem_qu_diff[padded_dim]        DATA_T    (q - u for current explored u)
  const uint32_t result_buffer_size = beam_width + 64;
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
  constexpr uint32_t ELEMENTS_PER_THREAD = (MAX_SEARCH_WIDTH - 1) / BLOCK_SIZE + 1;
  using BlockRadixSortT = cub::BlockRadixSort<uint32_t, BLOCK_SIZE,
                                              ELEMENTS_PER_THREAD, uint32_t>;
  using BlockScanT      = cub::BlockScan<uint32_t, BLOCK_SIZE>;
  union TempStorage {
    typename BlockRadixSortT::TempStorage sort_storage;
    typename BlockScanT::TempStorage      scan_storage;
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
    auto [frontierIdx, found] =
        choose_new_frontier<INDEX_T, DISTANCE_T, BLOCK_SIZE, MAX_SEARCH_WIDTH>(
            result_buffer, result_buffer_count);
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
    DATA_T* u_vec = graph.get_vector(u_gid);
    const float exact_dist_u_sq =
        compute_qu_diff_and_l2sq<DATA_T, BLOCK_SIZE>(
            smem_query_vec, u_vec, smem_qu_diff, padded_dim);

    // 4) Add u to frontier (top-k explored by EXACT dist) and visited log.
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

    // 5) Append u's neighbors to candidates as placeholders (no neighbor I/O).
    const uint32_t offset = result_buffer_count[0];
    __syncthreads();
    add_frontier_out<GRAPH_CFG>(graph, result_buffer, result_buffer_count,
                                u_gid, k, beam_width);
    __syncthreads();

    // 6) ESTIMATE ||q-v||² for the newly-appended neighbors using the LSH info
    //    stored on edges (u, v) — no fetch of v's vector required.
    const uint8_t n_edges = graph.get_edge_count(u_gid);
    populate_estimated_distances<GRAPH_CFG>(
        smem_qu_diff, exact_dist_u_sq, graph, u_gid, globals, inv_norm_denom,
        result_buffer, offset, n_edges);

    // 7) Sort / dedup / clip the candidate buffer (estimated distances).
    // merge_sort<INDEX_T, DISTANCE_T, BLOCK_SIZE, MAX_SEARCH_WIDTH, BlockMergeSortT>(
    //     result_buffer, result_buffer_count, temp_storage.sort_storage);
    radix_sort<INDEX_T, DISTANCE_T, BLOCK_SIZE, MAX_SEARCH_WIDTH, BlockRadixSortT>(
      result_buffer, result_buffer_count, temp_storage.sort_storage);
    dedup_results<INDEX_T, DISTANCE_T, BLOCK_SIZE, MAX_SEARCH_WIDTH, BlockScanT>(
        result_buffer, result_buffer_count, temp_storage.scan_storage);
    clip_k(result_buffer_count, beam_width);
  }

  // ---- Write top-k from frontier_buffer (already sorted ascending) ----
  __syncthreads();
  const uint32_t fc = frontier_buffer_count[0];
  for (uint32_t i = threadIdx.x; i < k; i += BLOCK_SIZE) {
    if (i < fc) {
      frontier_results[query_id * k + i].first  =
          static_cast<INDEX_T>(get_index(frontier_buffer[i]));
      frontier_results[query_id * k + i].second =
          static_cast<DISTANCE_T>(get_distance(frontier_buffer[i]));
    } else {
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
  s += sizeof(ENTRY_T) * (beam_width + 64);   // result_buffer
  s += sizeof(uint32_t);                       // result_buffer_count
  s = (s + 7) & ~7u;
  s += sizeof(ENTRY_T) * k;                    // frontier_buffer
  s += sizeof(uint32_t);                       // frontier_buffer_count
  s = (s + 15) & ~15u;
  s += sizeof(DATA_T) * padded_dim;            // smem_query_vec
  s += sizeof(DATA_T) * padded_dim;            // smem_qu_diff
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