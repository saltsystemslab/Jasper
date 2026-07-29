// jasper/beam_search/pq_beam_search.cuh
//
// Product-Quantization variant of the directional beam search. The beam-search
// skeleton is identical to directional_beam_search.cuh; the only difference is
// how a node's out-neighbors are scored: instead of the cross-polytope LSH
// estimator, this path uses Asymmetric Distance Computation (ADC) over learned
// PQ codebooks (see docs/lsh_beam_search_v5.md).
#pragma once

#include <cub/cub.cuh>

#include "jasper/distance/distance.cuh"
#include "jasper/index/vector.cuh"
#include "jasper/index/graph.cuh"
#include "jasper/beam_search/config.cuh"
#include "jasper/beam_search/entry.cuh"
#include "jasper/beam_search/device_kernels.cuh"
#include "jasper/beam_search/directional_beam_search.cuh"  // reuse helpers + CLOCK_*
#include "jasper/pq/pq_codebooks.cuh"

namespace jasper {

// Optional device-resident cache of a subset of nodes' full-graph data (rotated
// vector, adjacency, edge PQ codes), keyed by global id. When the host PQ search
// pops a node present in the cache, its whole expansion is served from device
// memory instead of host PCIe fetches. All-null (default) disables the cache.
template <typename GRAPH_CFG>
struct device_cache_view {
  const typename GRAPH_CFG::data_t*         vecs   = nullptr;  // [n_cache * stride] rotated
  const typename GRAPH_CFG::edge_list_t*    edges  = nullptr;  // [n_cache] full-graph adjacency
  const uint8_t*                            counts = nullptr;  // [n_cache] edge counts
  const typename GRAPH_CFG::edge_pq_list_t* pqs    = nullptr;  // [n_cache] edge PQ codes
  const int32_t*                            map    = nullptr;  // [n_full] gid -> slot, or -1
  uint32_t                                  stride = 0;        // row stride of vecs (padded_dim)
};

// ===== Build the query ADC lookup table (once per search) =====
// LUT[j*K + c] = <q_j, C_j[c]>, the projection of the query onto every centroid.
// δ = q (the query) for BOTH distance functions, so the table is constant across
// the whole search and is built exactly once — the popped node u enters the
// estimate only through the scalars <q,u> and ||v||² (see the scorer below),
// never through the LUT. This removes the per-node codebook re-read that used to
// dominate the L2 path.
template <typename DATA_T, uint32_t M, uint32_t K>
__device__ __forceinline__ void pq_build_lut(
    const DATA_T* __restrict__ query_vec,
    const float*  __restrict__ centroids,
    uint32_t                   dsub,
    float*        __restrict__ lut)
{
  // One WARP per centroid, lanes striding over the dsub coordinates. For a
  // fixed element the 32 lanes read Cj[0..31] — 32 contiguous global floats,
  // i.e. one coalesced transaction. (The old thread-per-element mapping had
  // adjacent lanes stride dsub floats apart, shattering each warp's read into
  // dsub sectors — the source of the "excessive global load" the profiler saw.)
  const uint32_t total   = M * K;
  const uint32_t lane    = threadIdx.x & 31u;
  const uint32_t warp    = threadIdx.x >> 5;
  const uint32_t n_warps = blockDim.x >> 5;
  for (uint32_t e = warp; e < total; e += n_warps) {
    const uint32_t j  = e / K;                       // subspace
    const float*  Cj = centroids + static_cast<size_t>(e) * dsub;  // (j*K+c)*dsub
    const DATA_T* qj = query_vec + static_cast<size_t>(j) * dsub;  // shared mem
    float acc = 0.0f;
    for (uint32_t t = lane; t < dsub; t += 32u)
      acc += static_cast<float>(qj[t]) * Cj[t];
    #pragma unroll
    for (int d = 16; d > 0; d >>= 1)
      acc += __shfl_xor_sync(0xFFFFFFFFu, acc, d);
    if (lane == 0) lut[e] = acc;
  }
  __syncthreads();
}

// ===== Score each new neighbor v of u via ADC table lookups =====
// The query LUT gives <q, ê> = Σ_j LUT[j, code_j] with ê the PQ reconstruction
// of the residual e = v - u, so <q,v> ≈ <q,u> + <q,ê>. From that:
//   INNER_PRODUCT → <q,v> ≈ qu_dot + Σ_j LUT[j,code_j]      (negated: smaller==closer)
//   L2            → ||q-v||² ≈ ||q||² - 2·<q,v> + ||v||²,
//                   using the EXACT stored ||v||² (graph.get_vector_norm) rather
//                   than a reconstructed norm — both faster and more accurate.
template <typename GRAPH_CFG, distance_func DISTANCE_FUNC, uint32_t M, uint32_t K>
__device__ __forceinline__ void pq_populate_estimated_distances(
    float                                   q_sq,      // ||q||²  (L2 only)
    float                                   qu_dot,    // <q,u>
    typename graph<GRAPH_CFG>::device_view& graph,
    typename GRAPH_CFG::index_t             u_gid,
    const float* __restrict__               lut,
    ENTRY_T*     __restrict__               result_buffer,
    uint32_t                                offset,
    uint8_t                                 n_edges,
    const uint8_t* __restrict__             code_base_override = nullptr)
{
  using INDEX_T = typename GRAPH_CFG::index_t;
  constexpr INDEX_T INVALID_INDEX = static_cast<INDEX_T>(0x7FFFFFFFu);

  // u's PQ code block: from the device cache if provided (avoids the host
  // edge_pqs read), else the segment lookup (same layout math as the LSH path).
  const uint8_t* __restrict__ code_base;
  if (code_base_override != nullptr) {
    code_base = code_base_override;
  } else {
    const INDEX_T  u_lid    = u_gid - graph.global_offset;
    const uint32_t u_segid  = static_cast<uint32_t>(u_lid / GRAPH_CFG::vectors_per_segment);
    const uint32_t u_locidx = static_cast<uint32_t>(u_lid % GRAPH_CFG::vectors_per_segment);
    code_base = graph.segments[u_segid].edge_pqs[u_locidx].code;
  }

  for (uint32_t e = threadIdx.x; e < n_edges; e += blockDim.x) {
    ENTRY_T entry = result_buffer[offset + e];
    INDEX_T v     = static_cast<INDEX_T>(get_index(entry));
    if (v == INVALID_INDEX) continue;

    // Subspace-major codes: code_base[j * N_NEIGHBORS + e]. For a fixed j the
    // warp's edges read consecutive bytes -> one coalesced transaction.
    constexpr uint32_t N = GRAPH_CFG::n_neighbors;
    float acc = qu_dot;                       // <q,u> + Σ_j <q, C_j[code_j]>
    #pragma unroll
    for (uint32_t j = 0; j < M; ++j)
      acc += lut[j * K + code_base[j * N + e]];

    float est_dist;
    if constexpr (DISTANCE_FUNC == distance_func::L2) {
      est_dist = q_sq - 2.0f * acc + graph.get_vector_norm(v);
    } else {  // INNER_PRODUCT
      est_dist = -acc;
    }
    result_buffer[offset + e] = set_distance(entry, est_dist);
  }
  __syncthreads();
}

// ===== The kernel =====
template <typename GRAPH_CFG, uint32_t BLOCK_SIZE, distance_func DISTANCE_FUNC,
          uint32_t MAX_SEARCH_WIDTH, bool GET_VISITED,
          uint32_t TILE_SIZE = 4, uint32_t MAX_RESULT_SIZE = 1024>
__global__ void pq_beam_search_kernel(
    typename graph<GRAPH_CFG>::device_view graph,
    pq_codebooks_view<GRAPH_CFG::pq_m, GRAPH_CFG::pq_k> codebooks,
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
    uint32_t                               limit,
    const typename GRAPH_CFG::index_t*     seeds,       // [n_query*n_seeds] gids, or null
    const typename GRAPH_CFG::distance_t*  seed_dists,  // [n_query*n_seeds] ests, or null
    uint32_t                               n_seeds,     // 0 => single-medoid seed
    device_cache_view<GRAPH_CFG>           cache,       // device hot cache (all-null = disabled)
    float                                  early_slack) // 0 => off; else stop when best cand > slack*kth
{
  using INDEX_T    = typename GRAPH_CFG::index_t;
  using DATA_T     = typename GRAPH_CFG::data_t;
  using DISTANCE_T = typename GRAPH_CFG::distance_t;
  constexpr uint32_t M = GRAPH_CFG::pq_m;
  constexpr uint32_t K = GRAPH_CFG::pq_k;
  static_assert(GRAPH_CFG::use_lsh,
                "pq_beam_search_kernel requires graph_cfg::use_lsh (directional storage)");

  const auto      query_id   = blockIdx.x;
  const uint32_t  padded_dim = graph.get_padded_dim();
  constexpr INDEX_T INVALID_INDEX = static_cast<INDEX_T>(0x7FFFFFFFu);

  assert(beam_width + GRAPH_CFG::n_neighbors <= MAX_SEARCH_WIDTH);

  // ---- Shared memory layout ----
  //   result_buffer[beam_width+n_neighbors]  ENTRY_T
  //   result_buffer_count                    uint32_t
  //   frontier_buffer[k]                     ENTRY_T
  //   frontier_buffer_count                  uint32_t
  //   merge_scratch[n_neighbors]             ENTRY_T
  //   smem_query_vec[padded_dim]             DATA_T
  //   smem_lut[M*K]                          float   (query ADC table)
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

  p = reinterpret_cast<unsigned char*>(
        (reinterpret_cast<uintptr_t>(p) + 15) & ~uintptr_t(15));
  float* __restrict__ smem_lut = reinterpret_cast<float*>(p);

  // ---- Load query vector ----
  DATA_T* query_vec_src = use_range ? graph.get_vector(query_start + query_id)
                                    : query_vectors[query_id];
  for (uint32_t i = threadIdx.x; i < padded_dim; i += BLOCK_SIZE)
    smem_query_vec[i] = query_vec_src[i];

  // ---- CUB temp storage for candidate sort/dedup ----
  constexpr uint32_t TAIL_EPT =
    (GRAPH_CFG::n_neighbors + BLOCK_SIZE - 1) / BLOCK_SIZE;
  using TailSortT  = cub::BlockMergeSort<ENTRY_T, BLOCK_SIZE, TAIL_EPT>;
  using BlockScanT = cub::BlockScan<uint32_t, BLOCK_SIZE>;
  union TempStorage {
    typename TailSortT::TempStorage  tail_sort_storage;
    typename BlockScanT::TempStorage scan_storage;
  };
  __shared__ TempStorage temp_storage;

  // ---- Initialize buffers; seed the candidate list ----
  // Default: a single medoid. Fused pipeline: an injected per-query seed set
  // (top-N from a cheap on-device coarse search) becomes the starting frontier,
  // seeded with the coarse exact distances so the best candidates pop first.
  // The seed count is clamped to beam_width so the first pop + neighbor append
  // cannot overflow result_buffer (capacity beam_width + n_neighbors).
  for (uint32_t i = threadIdx.x; i < result_buffer_size; i += BLOCK_SIZE)
    result_buffer[i] = empty_entry();
  if (threadIdx.x == 0) {
    uint32_t cnt = 0;
    if (seeds != nullptr && n_seeds > 0) {
      const uint32_t cap = beam_width;
      for (uint32_t s = 0; s < n_seeds && cnt < cap; ++s) {
        const INDEX_T sid = seeds[query_id * n_seeds + s];
        if (sid == INVALID_INDEX) continue;
        const float d = (seed_dists != nullptr)
            ? static_cast<float>(seed_dists[query_id * n_seeds + s]) : 0.0f;
        result_buffer[cnt++] = set_distance(set_index(empty_entry(), sid), d);
      }
    }
    if (cnt == 0) {  // no injected seeds → fall back to the medoid
      result_buffer[0] = set_distance(set_index(empty_entry(), medoid), 0.0f);
      cnt = 1;
    }
    result_buffer_count[0]   = cnt;
    frontier_buffer_count[0] = 0;
  }
  __syncthreads();

  uint32_t visited_counter = 0;
  uint32_t loop_count      = 0;

  // The ADC LUT projects the query onto every centroid (δ = q) and is constant
  // for the whole search, so build it ONCE here for both distance functions.
  pq_build_lut<DATA_T, M, K>(
      smem_query_vec, codebooks.centroids, codebooks.dsub, smem_lut);

  // ||q||² (L2 only) — reused every iteration to form ||q-v||² from <q,v>.
  // Plain block reduction over the shared query vector (compute_qu_diff_and_exact
  // can't be reused here: it __ldg's its second arg, which is illegal on smem).
  float q_sq = 0.0f;
  if constexpr (DISTANCE_FUNC == distance_func::L2) {
    float local = 0.0f;
    for (uint32_t i = threadIdx.x; i < padded_dim; i += BLOCK_SIZE) {
      const float x = static_cast<float>(smem_query_vec[i]);
      local += x * x;
    }
    #pragma unroll
    for (int d = 16; d > 0; d >>= 1)
      local += __shfl_xor_sync(0xFFFFFFFFu, local, d);
    constexpr uint32_t N_WARPS = BLOCK_SIZE / 32;
    __shared__ float s_qsq[N_WARPS > 0 ? N_WARPS : 1];
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t wid  = threadIdx.x >> 5;
    if (lane == 0) s_qsq[wid] = local;
    __syncthreads();
    if (wid == 0) {
      float v = (lane < N_WARPS) ? s_qsq[lane] : 0.0f;
      #pragma unroll
      for (int d = 16; d > 0; d >>= 1)
        v += __shfl_xor_sync(0xFFFFFFFFu, v, d);
      if (lane == 0) s_qsq[0] = v;
    }
    __syncthreads();
    q_sq = s_qsq[0];
  }

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

    // 1b) Adaptive early termination: once we have k results, stop as soon as the
    //     best remaining candidate (estimated) can't beat the k-th result within
    //     early_slack. slack 1.0 ≈ exact bound; <1 trades recall for fewer hops.
    if (early_slack > 0.0f && frontier_buffer_count[0] >= k) {
      const float cand_est = get_distance(result_buffer[frontierIdx]);
      const float kth_dist = get_distance(frontier_buffer[k - 1]);
      if (cand_est > early_slack * kth_dist) break;
    }

    // 2) Mark visited in candidate buffer.
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

    // 3) Fetch u's vector; compute <q,u> in one pass. The estimate now needs
    //    only this scalar from u (not the full q-u vector). For L2 the exact
    //    ||q-u||² for the frontier is recovered as ||q||² - 2<q,u> + ||u||².
    //    Device cache: if u is a subsample member, read its (rotated) vector
    //    from the on-device cache instead of a host PCIe fetch.
    CLOCK_START(t_exact);
    // Device cache lookup once per hop: a hit (cslot >= 0) serves u's whole
    // expansion — vector, adjacency, PQ codes — from device memory.
    const int32_t cslot = (cache.map != nullptr) ? cache.map[u_gid] : -1;
    const DATA_T* u_vec = (cslot >= 0)
        ? (cache.vecs + static_cast<size_t>(cslot) * cache.stride)
        : graph.get_vector(u_gid);
    const float qu_dot =
        compute_qu_diff_and_exact<DATA_T, BLOCK_SIZE, distance_func::INNER_PRODUCT>(
            smem_query_vec, u_vec, /*qu_diff=*/nullptr, padded_dim);
    float exact_dist_u;
    if constexpr (DISTANCE_FUNC == distance_func::L2) {
      exact_dist_u = q_sq - 2.0f * qu_dot + graph.get_vector_norm(u_gid);
    } else {
      exact_dist_u = -qu_dot;
    }
    CLOCK_ACCUM(t_exact, PHASE_EXACT);

    // 4) Add u to frontier (top-k explored by EXACT dist) and visited log.
    CLOCK_START(t_frontier);
    if (threadIdx.x == 0) {
      ENTRY_T u_entry =
          set_distance(set_index(empty_entry(), u_gid), exact_dist_u);
      frontier_insert_sorted(u_entry, exact_dist_u,
                             frontier_buffer, frontier_buffer_count, k);
      if (GET_VISITED) {
        visited_results[query_id * MAX_RESULT_SIZE + visited_counter].first  = u_gid;
        visited_results[query_id * MAX_RESULT_SIZE + visited_counter].second =
            static_cast<DISTANCE_T>(exact_dist_u);
      }
    }
    ++visited_counter;
    if (visited_counter == MAX_RESULT_SIZE) break;
    __syncthreads();
    CLOCK_ACCUM(t_frontier, PHASE_FRONTIER);

    // 5) Append u's neighbors to candidates as placeholders (cached adjacency
    //    on a cache hit → no host edge-list fetch).
    CLOCK_START(t_expand);
    const uint32_t offset = result_buffer_count[0];
    __syncthreads();
    const uint8_t n_edges = (cslot >= 0) ? cache.counts[cslot]
                                         : graph.get_edge_count(u_gid);
    if (cslot >= 0) {
      add_frontier_out<GRAPH_CFG>(graph, result_buffer, result_buffer_count,
                                  u_gid, k, beam_width,
                                  cache.edges[cslot].edges, n_edges);
    } else {
      add_frontier_out<GRAPH_CFG>(graph, result_buffer, result_buffer_count,
                                  u_gid, k, beam_width);
    }
    __syncthreads();
    CLOCK_ACCUM(t_expand, PHASE_EXPAND);

    // 6) ESTIMATE distances via PQ ADC using the once-built query LUT. On a
    //    cache hit, u's PQ codes come from the device cache (no host read).
    CLOCK_START(t_estimate);
    const uint8_t* code_override = (cslot >= 0) ? cache.pqs[cslot].code : nullptr;
    pq_populate_estimated_distances<GRAPH_CFG, DISTANCE_FUNC, M, K>(
        q_sq, qu_dot, graph, u_gid, smem_lut,
        result_buffer, offset, n_edges, code_override);
    CLOCK_ACCUM(t_estimate, PHASE_ESTIMATE);

    // 7) Sort / dedup / clip the candidate buffer (estimated distances).
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
__host__ inline uint32_t get_pq_smem_size(
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
  s += sizeof(ENTRY_T) * GRAPH_CFG::n_neighbors;                 // merge_scratch
  s = (s + 15) & ~15u;
  s += sizeof(DATA_T) * padded_dim;                              // smem_query_vec
  s = (s + 15) & ~15u;
  s += sizeof(float) * GRAPH_CFG::pq_m * GRAPH_CFG::pq_k;        // smem_lut
  return s;
}

// ===== Host: launcher =====
template <typename Cfg>
__host__ beam_search_result<typename Cfg::graph_cfg_t>
pq_beam_search(
    const beam_search_params<Cfg>&                                            p,
    pq_codebooks_view<Cfg::graph_cfg_t::pq_m, Cfg::graph_cfg_t::pq_k>          codebooks,
    const typename Cfg::graph_cfg_t::index_t*    d_seeds      = nullptr,
    const typename Cfg::graph_cfg_t::distance_t* d_seed_dists = nullptr,
    uint32_t                                     n_seeds      = 0,
    device_cache_view<typename Cfg::graph_cfg_t> cache        = {},
    cudaStream_t                                                              stream = 0)
{
  using graph_cfg_t = typename Cfg::graph_cfg_t;
  using entry_t     = typename Cfg::entry_t;
  static_assert(graph_cfg_t::use_lsh,
                "pq_beam_search requires graph_cfg::use_lsh (directional storage)");

  const uint32_t n_query_vectors = p.use_range
      ? (p.query_end - p.query_start)
      : p.query_vectors.n_vectors;

  dim3 threads(Cfg::block_size, 1, 1);
  dim3 blocks (n_query_vectors, 1, 1);

  const uint32_t padded_dim = p.graph.get_padded_dim();
  const uint32_t smem = get_pq_smem_size<graph_cfg_t>(
      p.beam_width, p.k, padded_dim);

  beam_search_result<graph_cfg_t> result{};
  cudaMalloc(&result.frontier, sizeof(entry_t) * n_query_vectors * p.k);
  if constexpr (Cfg::get_visited) {
    cudaMalloc(&result.visited,
               sizeof(entry_t) * n_query_vectors * Cfg::max_result_size);
    cudaMalloc(&result.visited_counts, sizeof(uint32_t) * n_query_vectors);
  }

  auto kernel = pq_beam_search_kernel<
      graph_cfg_t, Cfg::block_size, Cfg::dist_func,
      Cfg::max_search_width, Cfg::get_visited,
      Cfg::tile_size, Cfg::max_result_size>;

  if (smem > 48 * 1024) {
    cudaFuncSetAttribute(kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  }

  kernel<<<blocks, threads, smem, stream>>>(
      p.graph.view(), codebooks,
      result.frontier, result.visited, result.visited_counts,
      p.query_vectors, p.use_range, p.query_start, p.query_end,
      p.medoid, p.k, p.beam_width, p.limit,
      d_seeds, d_seed_dists, n_seeds, cache, p.early_slack);

  cudaError_t err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess) {
    std::cerr << "PQ beam search kernel launch failed: "
              << cudaGetErrorString(err) << std::endl;
  }
  return result;
}

}  // namespace jasper
