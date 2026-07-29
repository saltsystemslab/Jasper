// fused_query.cu
//
// Fused device-seeded PQ-directional search experiment.
//
//   Construct: build TWO indices from one dataset —
//     • full graph  (all N vectors), PQ-directional storage, moved to host-pinned
//       memory and searched with jasper::pq_search (ADC estimator, cheap hops).
//     • coarse graph (a random 1/N-th subsample), device-resident, searched with
//       the plain jasper::search (main jasper beam search).
//   Search: run the cheap on-device coarse search first; its top-S neighbors
//     (translated from subsample-local ids to global ids) become the starting
//     frontier of the host PQ-directional search, replacing the single medoid.
//
// Both graphs are prerotated with the SAME seed so one rotated query buffer
// serves both, and a global id found in the coarse graph indexes the same
// rotated vector in the full graph. For each beam:limit we report the single-
// medoid BASELINE and the FUSED run side by side (recall / QPS / mean visited).

#include <argparse/argparse.hpp>
#include <thrust/pair.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
#include <cuda_fp16.h>

#include "jasper/jasper.cuh"

#if defined(JASPER_PROFILE_CLOCKS)
namespace jasper {
__device__ uint64_t g_phase_clocks[PHASE_COUNT];
}  // namespace jasper
#endif

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__             \
                << " - " << cudaGetErrorString(err) << std::endl;             \
      throw std::runtime_error(cudaGetErrorString(err));                       \
    }                                                                          \
  } while (0)

// (CONFIG_ID, INDEX_T, N_NEIGHBORS, DATA_T, DISTANCE_T, DIST_FUNC, K_RANKS/M, PACKED_T)
// K_RANKS doubles as the PQ subquantizer count M. use_lsh=true (directional
// storage) is required for the PQ path; the coarse graph reuses the same config
// but never populates LSH/PQ (edge_lshs/edge_pqs stay lazily unallocated).
#define JASPER_FOR_EACH_CONFIG(X)                                                             \
  X(f16_r64_l2_k4_d128,  uint32_t, 64, __half, float, jasper::distance_func::L2,  4, uint8_t)  \
  X(f16_r64_l2_k8_d128,  uint32_t, 64, __half, float, jasper::distance_func::L2,  8, uint8_t)  \
  X(f16_r64_l2_k16_d128, uint32_t, 64, __half, float, jasper::distance_func::L2, 16, uint8_t)  \
  X(f16_r64_l2_k4_d32678,  uint32_t, 64, __half, float, jasper::distance_func::L2,  4, uint16_t)  \
  X(f16_r64_l2_k8_d32678,  uint32_t, 64, __half, float, jasper::distance_func::L2,  8, uint16_t)  \
  X(f16_r64_l2_k16_d32678, uint32_t, 64, __half, float, jasper::distance_func::L2, 16, uint16_t)  \

#define DECLARE_CONFIGS(id, IDX, R, DAT, DIST, FUNC, KR, PACKEDT)                       \
  using cfg_##id = jasper::graph_config<IDX, R, DAT, DIST, FUNC, true, KR, PACKEDT>;    \
  using construct_cfg_##id = jasper::graph_construct_config<cfg_##id, 64, 4, R, 64>;

JASPER_FOR_EACH_CONFIG(DECLARE_CONFIGS)
#undef DECLARE_CONFIGS

// ── Unpack thrust::pair results into separate arrays ───────────
__global__ void unpack_results_kernel(
    const thrust::pair<uint32_t, float>* __restrict__ pairs,
    int32_t* __restrict__ out_indices,
    float*   __restrict__ out_distances,
    uint32_t total) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < total) {
    out_indices[i]   = static_cast<int32_t>(pairs[i].first);
    out_distances[i] = pairs[i].second;
  }
}

// ── Translate coarse (subsample-local) results into full-graph global ids ───
// coarse[i].first is a subsample slot in [0, n_sub) (or a max() padding value);
// sample_global_ids maps that slot to the original global id. Padding / OOB
// entries become INVALID_INDEX (0x7FFFFFFF) which the PQ seed loop skips.
template <typename INDEX_T, typename DIST_T>
__global__ void translate_seeds_kernel(
    const thrust::pair<INDEX_T, DIST_T>* __restrict__ coarse,   // [n_query * S]
    const INDEX_T* __restrict__ sample_global_ids,              // [n_sub]
    INDEX_T                     n_sub,
    INDEX_T* __restrict__       out_seeds,                      // [n_query * S]
    DIST_T*  __restrict__       out_seed_dists,                 // [n_query * S]
    uint32_t                    total) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const INDEX_T sub = coarse[i].first;
  if (sub < n_sub) {
    out_seeds[i]      = sample_global_ids[sub];
    out_seed_dists[i] = coarse[i].second;
  } else {
    out_seeds[i]      = static_cast<INDEX_T>(0x7FFFFFFFu);  // INVALID_INDEX
    out_seed_dists[i] = static_cast<DIST_T>(1e30f);
  }
}

// ── Device cache build: gather each subsample node's full-graph data (rotated
//    vector, adjacency, edge PQ codes) from the full (device) graph into compact
//    arrays, and map global_id -> cache slot (-1 = miss). Run before move_to. ──
template <typename GraphCfg, typename DataT>
__global__ void gather_cache_kernel(
    typename jasper::graph<GraphCfg>::device_view  g,
    const typename GraphCfg::index_t* __restrict__ sample_global_ids,  // [n_sub]
    uint32_t n_sub, uint32_t padded_dim,
    DataT* __restrict__                            out_vecs,   // [n_sub*padded_dim]
    typename GraphCfg::edge_list_t* __restrict__   out_edges,  // [n_sub]
    uint8_t* __restrict__                          out_counts, // [n_sub]
    typename GraphCfg::edge_pq_list_t* __restrict__ out_pqs) { // [n_sub]
  using INDEX_T = typename GraphCfg::index_t;
  const uint32_t slot = blockIdx.x;
  if (slot >= n_sub) return;
  const INDEX_T gid = sample_global_ids[slot];

  // Rotated vector (all threads cooperate).
  const DataT* __restrict__ src = g.get_vector(gid);
  DataT* __restrict__ dst = out_vecs + static_cast<size_t>(slot) * padded_dim;
  for (uint32_t i = threadIdx.x; i < padded_dim; i += blockDim.x) dst[i] = src[i];

  // Adjacency + count + PQ codes (single struct copies; one thread).
  const INDEX_T  lid = gid - g.global_offset;
  const uint32_t seg = static_cast<uint32_t>(lid / GraphCfg::vectors_per_segment);
  const uint32_t loc = static_cast<uint32_t>(lid % GraphCfg::vectors_per_segment);
  if (threadIdx.x == 0) {
    out_counts[slot] = g.get_edge_count(gid);
    out_edges[slot]  = g.get_neighbor_list(gid);   // edges[R] + dist[R]
    out_pqs[slot]    = g.segments[seg].edge_pqs[loc];  // code[R*M]
  }
}

template <typename INDEX_T>
__global__ void build_cache_map_kernel(
    const INDEX_T* __restrict__ sample_global_ids, uint32_t n_sub,
    int32_t* __restrict__ cache_map) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n_sub) cache_map[sample_global_ids[i]] = static_cast<int32_t>(i);
}

// ── In-degree of every node (how many edges point TO it) for hub sampling. ──
template <typename GraphCfg>
__global__ void indegree_kernel(
    typename jasper::graph<GraphCfg>::device_view g,
    uint32_t n_full, uint32_t* __restrict__ indeg) {
  using INDEX_T = typename GraphCfg::index_t;
  const uint32_t u = blockIdx.x * blockDim.x + threadIdx.x;
  if (u >= n_full) return;
  const INDEX_T gid = static_cast<INDEX_T>(u) + g.global_offset;
  const uint8_t cnt = g.get_edge_count(gid);
  const auto& el = g.get_neighbor_list(gid);
  const INDEX_T lo = g.global_offset, hi = g.global_offset + g.n_vectors;
  for (uint8_t e = 0; e < cnt; ++e) {
    const INDEX_T nb = el.edges[e];
    if (nb >= lo && nb < hi) atomicAdd(&indeg[nb - g.global_offset], 1u);
  }
}

void print_available_memory(const char* tag) {
  size_t free_byte, total_byte;
  cudaMemGetInfo(&free_byte, &total_byte);
  double free_mb  = (double)free_byte  / (1024 * 1024);
  double total_mb = (double)total_byte / (1024 * 1024);
  printf("  [mem %-18s] used=%.0f MB free=%.0f MB total=%.0f MB\n",
         tag, total_mb - free_mb, free_mb, total_mb);
}

// ── Ground truth I/O & recall ──────────────────────────────────
struct GroundTruth {
  uint32_t n_queries;
  uint32_t gt_k;
  std::vector<int32_t> indices;
  std::vector<float>   distances;
};

GroundTruth read_groundtruth(const std::string& path, uint32_t k) {
  std::ifstream fin(path, std::ios::binary);
  if (!fin) throw std::runtime_error("Cannot open ground truth file: " + path);
  uint32_t n_queries, gt_k;
  fin.read(reinterpret_cast<char*>(&n_queries), sizeof(n_queries));
  fin.read(reinterpret_cast<char*>(&gt_k),      sizeof(gt_k));
  if (gt_k < k)
    throw std::runtime_error("Requested k=" + std::to_string(k) +
                             " but ground truth only has k=" + std::to_string(gt_k));
  std::vector<uint32_t> all_ids  (static_cast<size_t>(n_queries) * gt_k);
  std::vector<float>    all_dists(static_cast<size_t>(n_queries) * gt_k);
  fin.read(reinterpret_cast<char*>(all_ids.data()),   n_queries * gt_k * sizeof(uint32_t));
  fin.read(reinterpret_cast<char*>(all_dists.data()), n_queries * gt_k * sizeof(float));
  fin.close();
  GroundTruth gt;
  gt.n_queries = n_queries;
  gt.gt_k      = k;
  gt.indices.resize  (static_cast<size_t>(n_queries) * k);
  gt.distances.resize(static_cast<size_t>(n_queries) * k);
  for (uint32_t q = 0; q < n_queries; ++q)
    for (uint32_t j = 0; j < k; ++j) {
      gt.indices  [q * k + j] = static_cast<int32_t>(all_ids[q * gt_k + j]);
      gt.distances[q * k + j] = all_dists[q * gt_k + j];
    }
  return gt;
}

float get_recall(const GroundTruth& gt, const int32_t* result_indices,
                 uint32_t k, uint32_t n_queries) {
  uint64_t hits = 0;
  for (uint32_t q = 0; q < n_queries; ++q) {
    std::set<int32_t> gt_set(gt.indices.data() + q * k,
                             gt.indices.data() + q * k + k);
    for (uint32_t j = 0; j < k; ++j)
      if (gt_set.count(result_indices[q * k + j])) ++hits;
  }
  return static_cast<float>(hits) / (n_queries * k);
}

// ── Rotate query vectors with the construction seed (orthonormal, seed 42) ──
template <typename DataT>
void rotate_queries(DataT* d_queries, uint32_t n_queries, uint32_t dim,
                    uint32_t padded_dim, uint32_t seed, cudaStream_t stream = 0) {
  std::vector<float> h_P_f(dim * dim), h_Pt_f(dim * dim);
  jasper::set_rotation_matrix(dim, h_P_f.data(), h_Pt_f.data(), seed);
  std::vector<__half> h_P(h_P_f.size());
  for (size_t i = 0; i < h_P.size(); ++i) h_P[i] = static_cast<__half>(h_P_f[i]);

  __half* d_P = nullptr;
  cudaMalloc(&d_P, sizeof(__half) * dim * dim);
  cudaMemcpy(d_P, h_P.data(), sizeof(__half) * dim * dim, cudaMemcpyHostToDevice);

  __half* d_out = nullptr;
  size_t data_bytes = sizeof(__half) * n_queries * padded_dim;
  cudaMalloc(&d_out, data_bytes);
  cudaMemset(d_out, 0, data_bytes);

  cublasHandle_t handle;
  cublasCreate(&handle);
  cublasSetStream(handle, stream);
  const float one = 1.0f, zero = 0.0f;
  cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
               dim, n_queries, dim, &one,
               d_P,       CUDA_R_16F, dim,
               d_queries, CUDA_R_16F, padded_dim,
               &zero,
               d_out,     CUDA_R_16F, padded_dim,
               CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
  cublasDestroy(handle);

  cudaMemcpy(d_queries, d_out, data_bytes, cudaMemcpyDeviceToDevice);
  cudaFree(d_out);
  cudaFree(d_P);
}

// ── One PQ-directional round (baseline or fused) ───────────────
// Times pq_search, computes recall@k and mean nodes-visited/query.
template <typename GraphCfg, typename DataT, uint32_t BLOCK = 128>
void run_pq_round(
    jasper::graph<GraphCfg>&                                          g_full,
    jasper::pq_codebooks_view<GraphCfg::pq_m, GraphCfg::pq_k>         codebooks,
    DataT*    d_queries,
    uint32_t  n_queries,
    uint32_t  dim,
    uint32_t  k,
    uint32_t  beam_width,
    uint32_t  limit,
    const typename GraphCfg::index_t*    d_seeds,
    const typename GraphCfg::distance_t* d_seed_dists,
    uint32_t  n_seeds,
    jasper::device_cache_view<GraphCfg>  cache,
    float     early_slack,
    const GroundTruth* gt,
    const char* label) {

  jasper::vector_view<DataT> query_view(d_queries, dim, n_queries, false);
  jasper::search_params params{
    .k = k, .beam_width = beam_width, .limit = limit, .get_visited = true,
    .early_slack = early_slack,
  };

  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  cudaDeviceSynchronize();
  CUDA_CHECK(cudaGetLastError());

  cudaEventRecord(e0);
  auto result = jasper::pq_search<GraphCfg, BLOCK>(g_full, codebooks, query_view, params,
                                  d_seeds, d_seed_dists, n_seeds, cache);
  CUDA_CHECK(cudaGetLastError());
  cudaDeviceSynchronize();
  cudaEventRecord(e1);
  cudaEventSynchronize(e1);

  float duration_ms = 0;
  cudaEventElapsedTime(&duration_ms, e0, e1);
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  float qps = (n_queries * 1000.0f) / duration_ms;

  // Recall.
  float recall = -1.0f;
  if (gt) {
    uint32_t total = n_queries * k;
    int32_t* d_indices   = nullptr;
    float*   d_distances = nullptr;
    CUDA_CHECK(cudaMalloc(&d_indices,   total * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&d_distances, total * sizeof(float)));
    uint32_t threads = 256, blocks = (total + threads - 1) / threads;
    unpack_results_kernel<<<blocks, threads>>>(result.frontier, d_indices, d_distances, total);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<int32_t> h_indices(total);
    CUDA_CHECK(cudaMemcpy(h_indices.data(), d_indices, total * sizeof(int32_t), cudaMemcpyDeviceToHost));
    recall = get_recall(*gt, h_indices.data(), k, n_queries);
    cudaFree(d_indices);
    cudaFree(d_distances);
  }

  // Mean nodes visited per query.
  double mean_visited = 0.0;
  if (result.visited_counts) {
    std::vector<uint32_t> vc(n_queries);
    CUDA_CHECK(cudaMemcpy(vc.data(), result.visited_counts,
                          n_queries * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    mean_visited = std::accumulate(vc.begin(), vc.end(), 0.0) / n_queries;
  }

  std::cout << "  [" << std::left << std::setw(8) << label << std::right << "]"
            << " bw=" << std::setw(4) << beam_width
            << " lim=" << std::setw(4) << limit
            << " | QPS=" << std::setw(8) << std::fixed << std::setprecision(0) << qps
            << " | recall@" << k << "="
            << std::setw(6) << std::setprecision(4) << recall
            << " | visited/q=" << std::setprecision(1) << mean_visited
            << std::endl;

  cudaFree(result.frontier);
  if (result.visited)        cudaFree(result.visited);
  if (result.visited_counts) cudaFree(result.visited_counts);
}

// ── Build a device graph from a host vector_view (prerotate with `seed`) ─────
template <typename GraphCfg, typename ConstructCfg, typename DataT>
jasper::graph<GraphCfg> build_graph(jasper::vector_view<DataT>& h_vecs,
                                    float alpha, size_t workspace_budget,
                                    uint32_t prerotate_seed) {
  jasper::graph_construct_workspace<ConstructCfg> ws;
  uint32_t max_batch_size = std::max<uint32_t>(
      1u, std::min<uint32_t>(ws.max_batch_size_for_budget(workspace_budget),
                             std::max<uint32_t>(1u, h_vecs.n_vectors / 50)));

  jasper::graph_construct_params<ConstructCfg> params;
  params.data_vectors   = h_vecs;
  params.alpha          = alpha;
  params.max_batch_size = max_batch_size;
  params.on_host        = false;      // build on device
  params.prerotate      = true;
  params.prerotate_seed = prerotate_seed;
  return jasper::construct_graph<ConstructCfg>(params);
}

// ── Build coarse subsample graph (device, unrotated) + fill seed buffers ─────
// Templated on the coarse graph's config so its degree R can differ from the
// full graph. The coarse graph is plain (no LSH/PQ) and searched in UNROTATED
// space with unrotated queries — L2 nearest neighbors are rotation-invariant,
// so the global ids it returns still index the correct rotated vectors in the
// full graph. Returns the coarse search+translate time in ms.
template <typename CoarseCfg, typename CoarseConstruct, typename DataT>
float fill_seeds(jasper::vector_view<DataT> sub_view,
                 jasper::vector_view<DataT> qc_view,
                 const typename CoarseCfg::index_t* d_sample_global_ids,
                 uint32_t n_sub, uint32_t seeds,
                 uint32_t coarse_beam, uint32_t coarse_limit,
                 typename CoarseCfg::index_t*    d_seeds,
                 typename CoarseCfg::distance_t* d_seed_dists,
                 float alpha, size_t workspace_budget) {
  using index_t    = typename CoarseCfg::index_t;
  using distance_t = typename CoarseCfg::distance_t;

  jasper::graph_construct_workspace<CoarseConstruct> ws;
  jasper::graph_construct_params<CoarseConstruct> p;
  p.data_vectors   = sub_view;
  p.alpha          = alpha;
  p.on_host        = false;      // device-resident
  p.prerotate      = false;      // plain L2; NN are rotation-invariant
  p.max_batch_size = std::max<uint32_t>(1u,
      std::min<uint32_t>(ws.max_batch_size_for_budget(workspace_budget),
                         std::max<uint32_t>(1u, n_sub / 50)));
  auto g_coarse = jasper::construct_graph<CoarseConstruct>(p);

  const uint32_t n_queries = qc_view.n_vectors;
  const uint32_t eff_beam  = std::max(coarse_beam, seeds);  // must return >= seeds

  cudaEvent_t c0, c1; cudaEventCreate(&c0); cudaEventCreate(&c1);
  cudaEventRecord(c0);
  auto coarse = jasper::search(g_coarse, qc_view,
      jasper::search_params{.k = seeds, .beam_width = eff_beam,
                            .limit = coarse_limit, .get_visited = false});
  uint32_t total = n_queries * seeds, th = 256, bl = (total + th - 1) / th;
  translate_seeds_kernel<index_t, distance_t><<<bl, th>>>(
      coarse.frontier, d_sample_global_ids, static_cast<index_t>(n_sub),
      d_seeds, d_seed_dists, total);
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaEventRecord(c1); cudaEventSynchronize(c1);
  float ms = 0; cudaEventElapsedTime(&ms, c0, c1);
  cudaEventDestroy(c0); cudaEventDestroy(c1);
  cudaFree(coarse.frontier);
  g_coarse.deallocate();
  return ms;
}

// ── Construct full index once; compare baseline / fused / fused+device-cache ──
template <typename GraphCfg, typename ConstructCfg, typename DataT>
void construct_and_run(const std::string& base_path,
                       const std::string& query_path,
                       const std::string& gt_path,
                       const std::string& src_dtype,
                       uint32_t dim,
                       float alpha,
                       size_t workspace_budget,
                       uint32_t k,
                       uint32_t seeds,
                       uint32_t coarse_beam,
                       uint32_t coarse_limit,
                       double   fraction,
                       uint32_t coarse_R,
                       const std::string& sampler,
                       uint32_t pq_train,
                       bool full_on_host,
                       uint64_t sample_seed,
                       const std::vector<std::pair<uint32_t, uint32_t>>& beam_limit_pairs) {
  using index_t    = typename GraphCfg::index_t;
  using distance_t = typename GraphCfg::distance_t;
  constexpr uint32_t prerotate_seed = 42;

  // Coarse graph configs (plain, no LSH/PQ) derived from the full config, one
  // per supported degree R. R is a compile-time template param.
  using CoarseCfg32 = jasper::graph_config<index_t, 32, DataT, distance_t, GraphCfg::dist_func>;
  using CoarseCfg64 = jasper::graph_config<index_t, 64, DataT, distance_t, GraphCfg::dist_func>;
  using CoarseCon32 = jasper::graph_construct_config<CoarseCfg32, 64, 4, 32, 64>;
  using CoarseCon64 = jasper::graph_construct_config<CoarseCfg64, 64, 4, 64, 64>;

  seeds = std::max<uint32_t>(1u, seeds);

  // ── Load base vectors (original, unrotated) ──
  jasper::vector_view<DataT> h_vecs;
  if (src_dtype == "float")
    h_vecs = jasper::load_vectors_from_file_cast<DataT, float>(base_path);
  else if (src_dtype == "uint8")
    h_vecs = jasper::load_vectors_from_file_cast<DataT, uint8_t>(base_path);
  else
    throw std::runtime_error("Unsupported source dtype: " + src_dtype);
  if (!h_vecs.data)
    throw std::runtime_error("Failed to load vectors from: " + base_path);
  if (h_vecs.dim != dim)
    throw std::runtime_error("File dim=" + std::to_string(h_vecs.dim) +
                             " != --dim=" + std::to_string(dim));
  const uint32_t n_full = h_vecs.n_vectors;
  const uint32_t padded_dim = h_vecs.padded_dim;
  std::cout << "  Loaded " << n_full << " vectors, dim=" << dim
            << " padded=" << padded_dim << std::endl;

  uint32_t n_sub = static_cast<uint32_t>(fraction * n_full);
  n_sub = std::max<uint32_t>(1u, std::min(n_sub, n_full));

  // ── Build FULL graph from a COPY so h_vecs stays unrotated (needed for the
  //    unrotated subsample vectors + coverage sampling, both done post-build). ──
  std::cout << "  Building full graph (device)..." << std::endl;
  jasper::vector_view<DataT> h_build =
      jasper::vector_view<DataT>::allocate(dim, n_full, /*on_host=*/true);
  std::memcpy(h_build.data, h_vecs.data, h_vecs.size_bytes());
  auto g_full = build_graph<GraphCfg, ConstructCfg, DataT>(
      h_build, alpha, workspace_budget, prerotate_seed);
  cudaFreeHost(h_build.data);   // rotated copy; the graph owns its own vectors

  std::cout << "  Training PQ codebooks (n_train=" << pq_train
            << ", M=" << GraphCfg::pq_m << ", K=" << GraphCfg::pq_k << ")..." << std::endl;
  auto cb = g_full.generate_pq_codebooks(pq_train);
  g_full.populate_edge_pq(cb);
  g_full.compute_vector_norms();
  CUDA_CHECK(cudaDeviceSynchronize());

  // ── Choose the subsample's global ids by the requested sampler ──
  std::vector<index_t> sample_global_ids(n_sub);
  std::mt19937_64 rng(sample_seed);
  if (sampler == "random") {
    std::vector<uint32_t> perm(n_full);
    std::iota(perm.begin(), perm.end(), 0u);
    for (uint32_t i = 0; i < n_sub; ++i) {
      std::uniform_int_distribution<uint32_t> d(i, n_full - 1);
      std::swap(perm[i], perm[d(rng)]);
      sample_global_ids[i] = static_cast<index_t>(perm[i]);
    }
  } else if (sampler == "hub") {
    // Pick the n_sub most-pointed-to nodes (navigation hubs) by in-degree.
    uint32_t* d_indeg = nullptr;
    CUDA_CHECK(cudaMalloc(&d_indeg, static_cast<size_t>(n_full) * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(d_indeg, 0, static_cast<size_t>(n_full) * sizeof(uint32_t)));
    indegree_kernel<GraphCfg><<<(n_full + 127) / 128, 128>>>(g_full.view(), n_full, d_indeg);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<uint32_t> indeg(n_full);
    CUDA_CHECK(cudaMemcpy(indeg.data(), d_indeg, n_full * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    cudaFree(d_indeg);
    std::vector<uint32_t> ids(n_full);
    std::iota(ids.begin(), ids.end(), 0u);
    std::nth_element(ids.begin(), ids.begin() + n_sub, ids.end(),
                     [&](uint32_t a, uint32_t b) { return indeg[a] > indeg[b]; });
    for (uint32_t i = 0; i < n_sub; ++i) sample_global_ids[i] = static_cast<index_t>(ids[i]);
  } else if (sampler == "stratified") {
    // Even spatial coverage: bucket each point by the sign of its first K coords
    // relative to the per-dim mean, then systematic-sample across bucket order.
    const uint32_t kb = std::min<uint32_t>(16u, dim);
    std::vector<double> mean(kb, 0.0);
    for (uint32_t i = 0; i < n_full; ++i) {
      const DataT* v = h_vecs[i];
      for (uint32_t j = 0; j < kb; ++j) mean[j] += static_cast<double>(static_cast<float>(v[j]));
    }
    for (uint32_t j = 0; j < kb; ++j) mean[j] /= n_full;
    std::vector<uint32_t> bucket(n_full);
    for (uint32_t i = 0; i < n_full; ++i) {
      const DataT* v = h_vecs[i];
      uint32_t b = 0;
      for (uint32_t j = 0; j < kb; ++j)
        b |= (static_cast<float>(v[j]) >= mean[j] ? 1u : 0u) << j;
      bucket[i] = b;
    }
    std::vector<uint32_t> ids(n_full);
    std::iota(ids.begin(), ids.end(), 0u);
    std::shuffle(ids.begin(), ids.end(), rng);                    // random tiebreak
    std::stable_sort(ids.begin(), ids.end(),
                     [&](uint32_t a, uint32_t b) { return bucket[a] < bucket[b]; });
    for (uint32_t i = 0; i < n_sub; ++i)                          // systematic pick
      sample_global_ids[i] = static_cast<index_t>(ids[static_cast<size_t>(i) * n_full / n_sub]);
  } else {
    throw std::runtime_error("Unknown --sampler: " + sampler + " (random|hub|stratified)");
  }
  std::cout << "  Subsample: " << n_sub << " pts ("
            << std::fixed << std::setprecision(1) << (100.0 * n_sub / n_full)
            << "%), sampler=" << sampler << ", coarse_R=" << coarse_R << std::endl;

  // sample_global_ids → device.
  index_t* d_sample_global_ids = nullptr;
  CUDA_CHECK(cudaMalloc(&d_sample_global_ids, n_sub * sizeof(index_t)));
  CUDA_CHECK(cudaMemcpy(d_sample_global_ids, sample_global_ids.data(),
                        n_sub * sizeof(index_t), cudaMemcpyHostToDevice));

  // Subsample vectors (unrotated) for the coarse graph, then h_vecs is done.
  jasper::vector_view<DataT> sub_vecs =
      jasper::vector_view<DataT>::allocate(dim, n_sub, /*on_host=*/true);
  for (uint32_t j = 0; j < n_sub; ++j)
    std::memcpy(sub_vecs[j], h_vecs[sample_global_ids[j]],
                static_cast<size_t>(padded_dim) * sizeof(DataT));
  cudaFreeHost(h_vecs.data);

  // ── Device cache: gather each subsample node's ROTATED vector + adjacency +
  //    PQ codes from the full graph (still on device here) + a global_id→slot
  //    map, so a cache-hit hop in the host search is fully device-served. ──
  using edge_list_t    = typename GraphCfg::edge_list_t;
  using edge_pq_list_t = typename GraphCfg::edge_pq_list_t;
  DataT*          d_cache_vecs   = nullptr;
  edge_list_t*    d_cache_edges  = nullptr;
  uint8_t*        d_cache_counts = nullptr;
  edge_pq_list_t* d_cache_pqs    = nullptr;
  int32_t*        d_cache_map    = nullptr;
  CUDA_CHECK(cudaMalloc(&d_cache_vecs,   static_cast<size_t>(n_sub) * padded_dim * sizeof(DataT)));
  CUDA_CHECK(cudaMalloc(&d_cache_edges,  static_cast<size_t>(n_sub) * sizeof(edge_list_t)));
  CUDA_CHECK(cudaMalloc(&d_cache_counts, static_cast<size_t>(n_sub) * sizeof(uint8_t)));
  CUDA_CHECK(cudaMalloc(&d_cache_pqs,    static_cast<size_t>(n_sub) * sizeof(edge_pq_list_t)));
  gather_cache_kernel<GraphCfg, DataT><<<n_sub, 128>>>(
      g_full.view(), d_sample_global_ids, n_sub, padded_dim,
      d_cache_vecs, d_cache_edges, d_cache_counts, d_cache_pqs);
  CUDA_CHECK(cudaMalloc(&d_cache_map, static_cast<size_t>(n_full) * sizeof(int32_t)));
  CUDA_CHECK(cudaMemset(d_cache_map, 0xFF, static_cast<size_t>(n_full) * sizeof(int32_t)));  // -1
  build_cache_map_kernel<index_t><<<(n_sub + 255) / 256, 256>>>(
      d_sample_global_ids, n_sub, d_cache_map);
  CUDA_CHECK(cudaDeviceSynchronize());

  jasper::device_cache_view<GraphCfg> cache;
  cache.vecs   = d_cache_vecs;
  cache.edges  = d_cache_edges;
  cache.counts = d_cache_counts;
  cache.pqs    = d_cache_pqs;
  cache.map    = d_cache_map;
  cache.stride = padded_dim;
  const jasper::device_cache_view<GraphCfg> no_cache{};  // all-null = disabled

  std::cout << "  Moving full graph to " << (full_on_host ? "host-pinned" : "device")
            << " memory..." << std::endl;
  g_full.move_to(full_on_host);
  print_available_memory("after full build");

  // Seed buffers (allocated once query count is known, below).
  index_t*    d_seeds      = nullptr;
  distance_t* d_seed_dists = nullptr;

  // ── Load queries: UNROTATED (coarse) + ROTATED (full) ──
  jasper::vector_view<DataT> h_q;
  if (src_dtype == "float")
    h_q = jasper::load_vectors_from_file_cast<DataT, float>(query_path);
  else
    h_q = jasper::load_vectors_from_file_cast<DataT, uint8_t>(query_path);
  if (!h_q.data) throw std::runtime_error("Failed to load queries: " + query_path);
  if (h_q.dim != dim) throw std::runtime_error("Query dim mismatch");
  const uint32_t n_queries = h_q.n_vectors;
  const uint32_t q_padded  = h_q.padded_dim;
  auto d_qc_view = h_q.to_device();        // unrotated (for the coarse graph)
  cudaFreeHost(h_q.data);
  DataT* d_qc = d_qc_view.data;
  DataT* d_qf = nullptr;                    // rotated (for the full PQ graph)
  size_t q_bytes = sizeof(DataT) * static_cast<size_t>(n_queries) * q_padded;
  CUDA_CHECK(cudaMalloc(&d_qf, q_bytes));
  CUDA_CHECK(cudaMemcpy(d_qf, d_qc, q_bytes, cudaMemcpyDeviceToDevice));
  rotate_queries(d_qf, n_queries, dim, q_padded, prerotate_seed);
  jasper::vector_view<DataT> qc_view(d_qc, dim, n_queries, false);
  std::cout << "  " << n_queries << " queries loaded (unrotated + rotated)" << std::endl;

  // ── Ground truth ──
  GroundTruth gt;
  const bool has_gt = !gt_path.empty();
  if (has_gt) {
    gt = read_groundtruth(gt_path, k);
    if (gt.n_queries != n_queries)
      throw std::runtime_error("Ground truth query count mismatch");
  }

  // ── Seed buffers + fill from the coarse device search ──
  CUDA_CHECK(cudaMalloc(&d_seeds,      static_cast<size_t>(n_queries) * seeds * sizeof(index_t)));
  CUDA_CHECK(cudaMalloc(&d_seed_dists, static_cast<size_t>(n_queries) * seeds * sizeof(distance_t)));
  float coarse_ms = (coarse_R == 32)
      ? fill_seeds<CoarseCfg32, CoarseCon32, DataT>(
          sub_vecs, qc_view, d_sample_global_ids, n_sub, seeds,
          coarse_beam, coarse_limit, d_seeds, d_seed_dists, alpha, workspace_budget)
      : fill_seeds<CoarseCfg64, CoarseCon64, DataT>(
          sub_vecs, qc_view, d_sample_global_ids, n_sub, seeds,
          coarse_beam, coarse_limit, d_seeds, d_seed_dists, alpha, workspace_budget);
  cudaFreeHost(sub_vecs.data);
  std::cout << "  Coarse search+translate: " << std::fixed << std::setprecision(2)
            << coarse_ms << " ms (" << std::setprecision(0)
            << (n_queries * 1000.0f / coarse_ms) << " QPS)" << std::endl;

  // ── Reference: baseline / fused / fused+cache (block 128, no early-term) ──
  std::cout << "\n=== baseline / fused / fused+cache  (seeds=" << seeds
            << ", subsample " << std::setprecision(3) << fraction
            << ", coarse_R=" << coarse_R << ", sampler=" << sampler
            << ", k=" << k << ") ===" << std::endl;
  { auto [bw, lim] = beam_limit_pairs.front();  // warmup
    run_pq_round<GraphCfg, DataT>(g_full, cb.view(), d_qf, n_queries, dim, k,
                                  bw, lim, nullptr, nullptr, 0, no_cache, 0.0f,
                                  nullptr, "warmup");
    run_pq_round<GraphCfg, DataT>(g_full, cb.view(), d_qf, n_queries, dim, k,
                                  bw, lim, d_seeds, d_seed_dists, seeds, cache, 0.0f,
                                  nullptr, "warmup"); }
  for (auto& [bw, lim] : beam_limit_pairs) {
    run_pq_round<GraphCfg, DataT>(g_full, cb.view(), d_qf, n_queries, dim, k,
                                  bw, lim, nullptr, nullptr, 0, no_cache, 0.0f,
                                  has_gt ? &gt : nullptr, "baseline");
    run_pq_round<GraphCfg, DataT>(g_full, cb.view(), d_qf, n_queries, dim, k,
                                  bw, lim, d_seeds, d_seed_dists, seeds, no_cache, 0.0f,
                                  has_gt ? &gt : nullptr, "fused");
    run_pq_round<GraphCfg, DataT>(g_full, cb.view(), d_qf, n_queries, dim, k,
                                  bw, lim, d_seeds, d_seed_dists, seeds, cache, 0.0f,
                                  has_gt ? &gt : nullptr, "fused+cache");
  }

  // ── 3b) Block-size sweep (fused+cache, no early-term; block 128 shown above) ──
  std::cout << "\n=== block-size sweep (fused+cache) ===" << std::endl;
  for (auto& [bw, lim] : beam_limit_pairs)
    run_pq_round<GraphCfg, DataT, 64>(g_full, cb.view(), d_qf, n_queries, dim, k,
                                  bw, lim, d_seeds, d_seed_dists, seeds, cache, 0.0f,
                                  has_gt ? &gt : nullptr, "cache b64");
  for (auto& [bw, lim] : beam_limit_pairs)
    run_pq_round<GraphCfg, DataT, 256>(g_full, cb.view(), d_qf, n_queries, dim, k,
                                  bw, lim, d_seeds, d_seed_dists, seeds, cache, 0.0f,
                                  has_gt ? &gt : nullptr, "cache b256");

  // ── 3a) Early-termination sweep (fused+cache, block 256 — the block winner).
  //     Compare vs the "cache b256" (no early-term) rows above: slack 1.10-1.20
  //     is lossless while cutting hops; <1 trades recall for more speed. ──
  std::cout << "\n=== early-termination sweep (fused+cache, block 256) ===" << std::endl;
  for (float slack : {1.20f, 1.10f, 1.00f, 0.95f}) {
    char lbl[28]; std::snprintf(lbl, sizeof(lbl), "c256 s%.2f", slack);
    for (auto& [bw, lim] : beam_limit_pairs)
      run_pq_round<GraphCfg, DataT, 256>(g_full, cb.view(), d_qf, n_queries, dim, k,
                                    bw, lim, d_seeds, d_seed_dists, seeds, cache, slack,
                                    has_gt ? &gt : nullptr, lbl);
  }

  // Cleanup.
  cudaFree(d_seeds);
  cudaFree(d_seed_dists);
  cudaFree(d_sample_global_ids);
  cudaFree(d_cache_vecs);
  cudaFree(d_cache_edges);
  cudaFree(d_cache_counts);
  cudaFree(d_cache_pqs);
  cudaFree(d_cache_map);
  cudaFree(d_qc);
  cudaFree(d_qf);
  cb.free();
  g_full.deallocate();
}

// ── Config dispatch ────────────────────────────────────────────
template <jasper::distance_func Func, uint32_t KR, typename PackedT>
bool config_matches(const std::string& datatype, uint64_t n_neighbors,
                    const std::string& distance, uint64_t R, uint32_t k_ranks,
                    uint32_t dim) {
  std::string expected_dist = (Func == jasper::distance_func::L2) ? "l2" : "ip";
  bool dtype_ok = (datatype == "float" || datatype == "uint8");
  bool dim_ok   = (sizeof(PackedT) == 1) ? (dim <= 128) : (dim > 128);
  return dtype_ok && n_neighbors == R && distance == expected_dist &&
         k_ranks == KR && dim_ok;
}

int main(int argc, char** argv) {
  argparse::ArgumentParser program("fused_query");

  program.add_argument("--base", "-f").required()
    .help("Base vectors filename (constructs both indices).");
  program.add_argument("--queries", "-q").required()
    .help("Query vectors filename.");
  program.add_argument("--groundtruth", "-g").default_value(std::string{})
    .help("Ground truth .bin (optional, for recall).");
  program.add_argument("--datatype", "-t").required().choices("uint8", "float")
    .help("Source vector datatype (cast to __half).");
  program.add_argument("--distance", "-d").default_value(std::string{"l2"}).choices("l2", "ip")
    .help("Distance type.");
  program.add_argument("--n_neighbors", "-n").default_value(uint64_t{64}).scan<'u', uint64_t>()
    .help("Neighbors per vector (graph degree R).");
  program.add_argument("--dim").required().scan<'u', uint32_t>().help("Vector dimension.");
  program.add_argument("--k_ranks", "-r").default_value(uint32_t{8}).scan<'u', uint32_t>()
    .choices(4u, 8u, 16u).help("PQ subquantizers M (= LSH k_ranks): 4, 8, or 16.");
  program.add_argument("--k", "-k").default_value(uint32_t{10}).scan<'u', uint32_t>()
    .help("k nearest neighbors to return.");
  program.add_argument("--alpha", "-a").default_value(1.2f).scan<'g', float>()
    .help("Pruning factor alpha.");
  program.add_argument("--seeds", "-S").default_value(uint32_t{32}).scan<'u', uint32_t>()
    .help("Fixed #seeds injected into the host frontier (default 32).");
  program.add_argument("--coarse").default_value(std::string{"64:128"})
    .help("Fixed coarse device search quality, as beam:limit (default 64:128).");
  program.add_argument("--subsample").default_value(0.1).scan<'g', double>()
    .help("Subsample fraction for the coarse device index (default 0.10).");
  program.add_argument("--coarse_R").default_value(uint32_t{32}).scan<'u', uint32_t>()
    .choices(32u, 64u).help("Coarse device graph degree R (32 or 64; default 32).");
  program.add_argument("--sampler").default_value(std::string{"random"})
    .choices("random", "hub", "stratified")
    .help("Subsample selection: random | hub (in-degree) | stratified (coverage).");
  program.add_argument("--pq_train").default_value(uint32_t{40000}).scan<'u', uint32_t>()
    .help("PQ codebook training sample size.");
  program.add_argument("--sample_seed").default_value(uint64_t{7}).scan<'u', uint64_t>()
    .help("RNG seed for subsample selection.");
  program.add_argument("--full_on_device").default_value(false).implicit_value(true)
    .help("Keep the full graph device-resident (default: host-pinned).");
  program.add_argument("--workspace_budget", "-w").default_value(size_t{10ULL << 30})
    .action([](const std::string& v) -> size_t {
      size_t pos = 0; double num = std::stod(v, &pos);
      std::string u = v.substr(pos); for (auto& c : u) c = std::toupper(c);
      if (u == "B")  return (size_t)num;
      if (u == "KB") return (size_t)(num * 1024);
      if (u == "MB") return (size_t)(num * 1024 * 1024);
      if (u == "GB") return (size_t)(num * 1024 * 1024 * 1024);
      if (u == "TB") return (size_t)(num * 1024.0 * 1024 * 1024 * 1024);
      throw std::runtime_error("Invalid size unit: " + u);
    });
  program.add_argument("--beam_limits", "-b")
    .default_value(std::vector<std::string>{
      "16:128", "32:128", "64:128", "128:256", "256:512"})
    .nargs(argparse::nargs_pattern::at_least_one)
    .help("beam_width:limit pairs for the host PQ search.");

  try {
    program.parse_args(argc, argv);
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl << program;
    return 1;
  }

  auto base_path   = program.get<std::string>("--base");
  auto query_path  = program.get<std::string>("--queries");
  auto gt_path     = program.get<std::string>("--groundtruth");
  auto datatype    = program.get<std::string>("--datatype");
  auto distance    = program.get<std::string>("--distance");
  auto n_neighbors = program.get<uint64_t>("--n_neighbors");
  auto dim         = program.get<uint32_t>("--dim");
  auto k_ranks     = program.get<uint32_t>("--k_ranks");
  auto k           = program.get<uint32_t>("--k");
  auto alpha       = program.get<float>("--alpha");
  auto seeds        = program.get<uint32_t>("--seeds");
  auto coarse_str   = program.get<std::string>("--coarse");
  auto subsample    = program.get<double>("--subsample");
  auto coarse_R     = program.get<uint32_t>("--coarse_R");
  auto sampler      = program.get<std::string>("--sampler");
  auto pq_train    = program.get<uint32_t>("--pq_train");
  auto sample_seed = program.get<uint64_t>("--sample_seed");
  auto full_on_dev = program.get<bool>("--full_on_device");
  auto workspace   = program.get<size_t>("--workspace_budget");
  auto beam_args   = program.get<std::vector<std::string>>("--beam_limits");

  std::vector<std::pair<uint32_t, uint32_t>> beam_limit_pairs;
  for (auto& s : beam_args) {
    auto colon = s.find(':');
    if (colon == std::string::npos) { std::cerr << "Bad beam:limit: " << s << "\n"; return 1; }
    beam_limit_pairs.emplace_back(
        (uint32_t)std::stoul(s.substr(0, colon)),
        (uint32_t)std::stoul(s.substr(colon + 1)));
  }

  uint32_t coarse_beam, coarse_limit;
  {
    auto colon = coarse_str.find(':');
    if (colon == std::string::npos) { std::cerr << "Bad --coarse beam:limit: " << coarse_str << "\n"; return 1; }
    coarse_beam  = (uint32_t)std::stoul(coarse_str.substr(0, colon));
    coarse_limit = (uint32_t)std::stoul(coarse_str.substr(colon + 1));
  }

  std::cout << "=== fused_query ===" << std::endl;
  std::cout << "  base:        " << base_path << std::endl;
  std::cout << "  queries:     " << query_path << std::endl;
  std::cout << "  groundtruth: " << (gt_path.empty() ? "(none)" : gt_path) << std::endl;
  std::cout << "  dim:         " << dim << "  R=" << n_neighbors
            << "  M(k_ranks)=" << k_ranks << "  k=" << k << std::endl;
  std::cout << "  seeds:       " << seeds << "  coarse=" << coarse_str << std::endl;
  std::cout << "  subsample:   " << subsample << "  coarse_R=" << coarse_R
            << "  sampler=" << sampler << std::endl;
  std::cout << "  full graph:  " << (full_on_dev ? "device" : "host-pinned") << std::endl;

  try {
    #define TRY_DISPATCH(id, IDX, R, DAT, DIST, FUNC, KR, PACKEDT)                          \
      if (config_matches<FUNC, KR, PACKEDT>(datatype, n_neighbors, distance, R, k_ranks, dim)) { \
        std::cout << "  Config: " #id << std::endl;                                         \
        construct_and_run<cfg_##id, construct_cfg_##id, DAT>(                               \
            base_path, query_path, gt_path, datatype, dim, alpha, workspace, k,             \
            seeds, coarse_beam, coarse_limit, subsample, coarse_R, sampler, pq_train,       \
            /*full_on_host=*/!full_on_dev, sample_seed, beam_limit_pairs);                  \
        std::cout << "  Done." << std::endl;                                                \
        return 0;                                                                           \
      }
    JASPER_FOR_EACH_CONFIG(TRY_DISPATCH)
    #undef TRY_DISPATCH

    throw std::runtime_error(
      "Unsupported config: datatype=" + datatype +
      ", n_neighbors=" + std::to_string(n_neighbors) +
      ", distance=" + distance + ", k_ranks=" + std::to_string(k_ranks) +
      ", dim=" + std::to_string(dim));
  } catch (const std::exception& err) {
    std::cerr << "Error: " << err.what() << std::endl;
    return 1;
  }
  return 0;
}
