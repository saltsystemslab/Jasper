// jasper/beam_search/distortion.cuh
//
// Estimator distortion probe: for sampled (node u, query q) pairs, score every
// out-edge (u -> v) with a directional ESTIMATOR exactly as the beam search
// would on popping u, and record it next to the exact distance to v.
//
// The estimate comes from the estimator's own constructor / visit() /
// estimate_neighbors() — the same calls, in the same order, that
// graph_beam_search_kernel makes — so what gets measured is the code that
// searches, not a host-side re-derivation of it. Works with any type that
// fulfills the estimator API (lsh_estimator, pq_estimator, rq_estimator).
//
// Distances use the beam search's convention: squared L2, or the negated dot
// product for INNER_PRODUCT (smaller == closer). The exact side is computed in
// fp32 from the stored fp16 vectors.
#pragma once

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "jasper/index/graph.cuh"
#include "jasper/index/vector.cuh"
#include "jasper/beam_search/entry.cuh"

namespace jasper {

// One block per sample. Writes N_NEIGHBORS rows per sample; slots past the
// node's edge count (or whose neighbor is outside this graph) get
// v = INVALID_INDEX and should be dropped by the reader.
template <typename GRAPH_CFG, uint32_t BLOCK_SIZE, typename ESTIMATOR>
__global__ void estimator_distortion_kernel(
    typename graph<GRAPH_CFG>::device_view    graph,
    typename ESTIMATOR::globals_t             globals,
    vector_view<typename GRAPH_CFG::data_t>   queries,
    const typename GRAPH_CFG::index_t* __restrict__ sample_nodes,
    const uint32_t*                    __restrict__ sample_queries,
    typename GRAPH_CFG::index_t*       __restrict__ out_v,
    float*                             __restrict__ out_exact,
    float*                             __restrict__ out_est)
{
  using INDEX_T = typename GRAPH_CFG::index_t;
  using DATA_T  = typename GRAPH_CFG::data_t;
  constexpr uint32_t N_NEIGHBORS   = GRAPH_CFG::n_neighbors;
  constexpr INDEX_T  INVALID_INDEX = static_cast<INDEX_T>(0x7FFFFFFFu);
  constexpr bool     IS_L2         = (GRAPH_CFG::dist_func == distance_func::L2);

  const uint32_t s          = blockIdx.x;
  const uint32_t padded_dim = graph.get_padded_dim();
  const size_t   row0       = static_cast<size_t>(s) * N_NEIGHBORS;

  // ---- Shared memory: edge buffer | query | estimator extra ----
  extern __shared__ __align__(16) unsigned char smem_raw[];
  unsigned char* p = smem_raw;
  ENTRY_T* edge_buf = reinterpret_cast<ENTRY_T*>(p);
  p += N_NEIGHBORS * sizeof(ENTRY_T);
  p = reinterpret_cast<unsigned char*>(
      (reinterpret_cast<uintptr_t>(p) + 15) & ~uintptr_t(15));
  DATA_T* smem_query_vec = reinterpret_cast<DATA_T*>(p);
  p += padded_dim * sizeof(DATA_T);
  p = reinterpret_cast<unsigned char*>(
      (reinterpret_cast<uintptr_t>(p) + 15) & ~uintptr_t(15));
  unsigned char* extra_smem = p;

  const DATA_T* q_src = queries[sample_queries[s]];
  for (uint32_t i = threadIdx.x; i < padded_dim; i += BLOCK_SIZE)
    smem_query_vec[i] = q_src[i];
  __syncthreads();

  // Constructed unconditionally: its constructor may hold block-wide barriers.
  ESTIMATOR estimator(extra_smem, smem_query_vec, padded_dim, globals);

  const INDEX_T u = sample_nodes[s];
  if (!graph.is_valid(u)) {  // uniform across the block
    for (uint32_t e = threadIdx.x; e < N_NEIGHBORS; e += BLOCK_SIZE)
      out_v[row0 + e] = INVALID_INDEX;
    return;
  }

  // Same sequence as one iteration of graph_beam_search_kernel: visit u, lay
  // its out-edges into the buffer (add_frontier_out's placeholder rule), then
  // let the estimator score them in place.
  estimator.visit(graph, u, smem_query_vec, padded_dim);

  const uint8_t n_edges  = graph.get_edge_count(u);
  const INDEX_T range_lo = graph.global_offset;
  const INDEX_T range_hi = graph.global_offset + graph.n_vectors;
  const INDEX_T* __restrict__ edges = &graph.get_neighbor_list(u).edges[0];
  for (uint32_t e = threadIdx.x; e < N_NEIGHBORS; e += BLOCK_SIZE) {
    const INDEX_T nb = (e < n_edges) ? edges[e] : INVALID_INDEX;
    edge_buf[e] = (nb >= range_lo && nb < range_hi)
                      ? set_index(empty_entry(), nb)
                      : set_visited(empty_entry());
  }
  __syncthreads();

  estimator.estimate_neighbors(graph, u, smem_query_vec, edge_buf, 0, n_edges);

  // ---- Exact distance, one warp per edge ----
  const uint32_t lane    = threadIdx.x & 31u;
  const uint32_t warp    = threadIdx.x >> 5;
  constexpr uint32_t N_WARPS = BLOCK_SIZE / 32;
  for (uint32_t e = warp; e < N_NEIGHBORS; e += N_WARPS) {
    const INDEX_T v = static_cast<INDEX_T>(get_index(edge_buf[e]));
    if (v == INVALID_INDEX) {
      if (lane == 0) out_v[row0 + e] = INVALID_INDEX;
      continue;
    }
    const DATA_T* __restrict__ v_vec = graph.get_vector(v);
    float acc = 0.0f;
    for (uint32_t i = lane; i < padded_dim; i += 32u) {
      const float qi = static_cast<float>(smem_query_vec[i]);
      const float vi = static_cast<float>(v_vec[i]);
      if constexpr (IS_L2) { const float d = qi - vi; acc += d * d; }
      else                 { acc += qi * vi; }
    }
    #pragma unroll
    for (int d = 16; d > 0; d >>= 1)
      acc += __shfl_xor_sync(0xFFFFFFFFu, acc, d);
    if (lane == 0) {
      out_v[row0 + e]     = v;
      out_exact[row0 + e] = IS_L2 ? acc : -acc;
      out_est[row0 + e]   = get_distance(edge_buf[e]);
    }
  }
}

// Host result: N_NEIGHBORS rows per sample, row-major by sample.
template <typename INDEX_T>
struct distortion_samples {
  uint32_t              n_neighbors = 0;
  std::vector<INDEX_T>  v;      // INVALID (0x7FFFFFFF) == empty slot, skip it
  std::vector<float>    exact;
  std::vector<float>    est;
};

// Run the probe for `sample_nodes[i]` against query `sample_queries[i]`.
// `queries` must be device-resident and in the same (rotated) space the codes
// were built in.
template <typename GRAPH_CFG, typename ESTIMATOR, uint32_t BLOCK_SIZE = 128>
__host__ distortion_samples<typename GRAPH_CFG::index_t> estimator_distortion(
    graph<GRAPH_CFG>&                              g,
    typename ESTIMATOR::globals_t                  globals,
    vector_view<typename GRAPH_CFG::data_t>        queries,
    const std::vector<typename GRAPH_CFG::index_t>& sample_nodes,
    const std::vector<uint32_t>&                   sample_queries)
{
  using INDEX_T = typename GRAPH_CFG::index_t;
  using DATA_T  = typename GRAPH_CFG::data_t;
  constexpr uint32_t N = GRAPH_CFG::n_neighbors;

  if (sample_nodes.size() != sample_queries.size())
    throw std::runtime_error("estimator_distortion: sample_nodes / sample_queries size mismatch");
  const uint32_t n_samples = static_cast<uint32_t>(sample_nodes.size());

  distortion_samples<INDEX_T> out;
  out.n_neighbors = N;
  if (n_samples == 0) return out;

  const size_t n_rows = static_cast<size_t>(n_samples) * N;
  INDEX_T*  d_nodes = nullptr;
  uint32_t* d_qids  = nullptr;
  INDEX_T*  d_v     = nullptr;
  float*    d_exact = nullptr;
  float*    d_est   = nullptr;
  cudaMalloc(&d_nodes, sizeof(INDEX_T)  * n_samples);
  cudaMalloc(&d_qids,  sizeof(uint32_t) * n_samples);
  cudaMalloc(&d_v,     sizeof(INDEX_T)  * n_rows);
  cudaMalloc(&d_exact, sizeof(float)    * n_rows);
  cudaMalloc(&d_est,   sizeof(float)    * n_rows);
  cudaMemcpy(d_nodes, sample_nodes.data(),   sizeof(INDEX_T)  * n_samples, cudaMemcpyHostToDevice);
  cudaMemcpy(d_qids,  sample_queries.data(), sizeof(uint32_t) * n_samples, cudaMemcpyHostToDevice);

  const uint32_t padded_dim = g.get_padded_dim();
  uint32_t smem = sizeof(ENTRY_T) * N;
  smem = (smem + 15) & ~15u;
  smem += sizeof(DATA_T) * padded_dim;
  smem = (smem + 15) & ~15u;
  smem += ESTIMATOR::extra_smem_size(padded_dim);

  auto kernel = estimator_distortion_kernel<GRAPH_CFG, BLOCK_SIZE, ESTIMATOR>;
  if (smem > 48 * 1024)
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);

  kernel<<<n_samples, BLOCK_SIZE, smem>>>(
      g.view(), globals, queries, d_nodes, d_qids, d_v, d_exact, d_est);
  cudaError_t err = cudaDeviceSynchronize();
  if (err == cudaSuccess) err = cudaGetLastError();

  if (err == cudaSuccess) {
    out.v.resize(n_rows);
    out.exact.resize(n_rows);
    out.est.resize(n_rows);
    cudaMemcpy(out.v.data(),     d_v,     sizeof(INDEX_T) * n_rows, cudaMemcpyDeviceToHost);
    cudaMemcpy(out.exact.data(), d_exact, sizeof(float)   * n_rows, cudaMemcpyDeviceToHost);
    cudaMemcpy(out.est.data(),   d_est,   sizeof(float)   * n_rows, cudaMemcpyDeviceToHost);
  }

  cudaFree(d_nodes);
  cudaFree(d_qids);
  cudaFree(d_v);
  cudaFree(d_exact);
  cudaFree(d_est);

  if (err != cudaSuccess)
    throw std::runtime_error(std::string("estimator_distortion kernel failed: ")
                             + cudaGetErrorString(err));
  return out;
}

}  // namespace jasper
