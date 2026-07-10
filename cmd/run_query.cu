#include <argparse/argparse.hpp>
#include <thrust/pair.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <cstdint>
#include <cstring>
#include <set>
#include <stdexcept>
#include <vector>
#include <chrono>
#include <type_traits>
#include <cuda_fp16.h>
#include <cublas_v2.h>

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
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__            \
                << " - " << cudaGetErrorString(err) << std::endl;             \
      throw std::runtime_error(cudaGetErrorString(err));                       \
    }                                                                          \
  } while (0)

// ── Config table ───────────────────────────────────────────────
#define JASPER_FOR_EACH_CONFIG(X)                                              \
  X(f16_r32_l2,   uint32_t, 32,  __half,   float, jasper::distance_func::L2)             \
  X(f16_r64_l2,   uint32_t, 64,  __half,   float, jasper::distance_func::L2)             \
  X(f16_r32_ip,   uint32_t, 32,  __half,   float, jasper::distance_func::INNER_PRODUCT)  \
  X(f16_r64_ip,   uint32_t, 64,  __half,   float, jasper::distance_func::INNER_PRODUCT)

#define DECLARE_CONFIGS_(id, IDX, R, DAT, DIST, FUNC)                          \
  using cfg_##id = jasper::graph_config<IDX, R, DAT, DIST, FUNC>;

JASPER_FOR_EACH_CONFIG(DECLARE_CONFIGS_)
#undef DECLARE_CONFIGS_

// ── LSH config table ───────────────────────────────────────────
// Configs with use_lsh=true, used when --lsh is passed. The graph file format
// is identical to a non-LSH index (LSH coords are not serialized), so any index
// produced by create_index can be loaded here and have its LSH populated at
// query time. K_RANKS and PACKED_T mirror create_lsh_index.
// (CONFIG_ID, INDEX_T, N_NEIGHBORS, DATA_T, DISTANCE_T, DIST_FUNC, K_RANKS, PACKED_T)
#define JASPER_FOR_EACH_LSH_CONFIG(X)                                          \
  X(lsh_f16_r32_l2_k4_d128,  uint32_t, 32, __half, float, jasper::distance_func::L2,  4, uint8_t)  \
  X(lsh_f16_r32_l2_k8_d128,  uint32_t, 32, __half, float, jasper::distance_func::L2,  8, uint8_t)  \
  X(lsh_f16_r32_l2_k16_d128, uint32_t, 32, __half, float, jasper::distance_func::L2, 16, uint8_t)  \
  X(lsh_f16_r64_l2_k4_d128,  uint32_t, 64, __half, float, jasper::distance_func::L2,  4, uint8_t)  \
  X(lsh_f16_r64_l2_k8_d128,  uint32_t, 64, __half, float, jasper::distance_func::L2,  8, uint8_t)  \
  X(lsh_f16_r64_l2_k16_d128, uint32_t, 64, __half, float, jasper::distance_func::L2, 16, uint8_t)  \
  X(lsh_f16_r64_l2_k4_d32678,  uint32_t, 64, __half, float, jasper::distance_func::L2,  4, uint16_t)  \
  X(lsh_f16_r64_l2_k8_d32678,  uint32_t, 64, __half, float, jasper::distance_func::L2,  8, uint16_t)  \
  X(lsh_f16_r64_l2_k16_d32678, uint32_t, 64, __half, float, jasper::distance_func::L2, 16, uint16_t)  \
  X(lsh_f16_r32_ip_k4_d128,  uint32_t, 32, __half, float, jasper::distance_func::INNER_PRODUCT,  4, uint8_t)  \
  X(lsh_f16_r32_ip_k8_d128,  uint32_t, 32, __half, float, jasper::distance_func::INNER_PRODUCT,  8, uint8_t)  \
  X(lsh_f16_r32_ip_k16_d128, uint32_t, 32, __half, float, jasper::distance_func::INNER_PRODUCT, 16, uint8_t)  \
  X(lsh_f16_r64_ip_k4_d128,  uint32_t, 64, __half, float, jasper::distance_func::INNER_PRODUCT,  4, uint8_t)  \
  X(lsh_f16_r64_ip_k8_d128,  uint32_t, 64, __half, float, jasper::distance_func::INNER_PRODUCT,  8, uint8_t)  \
  X(lsh_f16_r64_ip_k16_d128, uint32_t, 64, __half, float, jasper::distance_func::INNER_PRODUCT, 16, uint8_t)  \
  X(lsh_f16_r64_ip_k4_d32678,  uint32_t, 64, __half, float, jasper::distance_func::INNER_PRODUCT,  4, uint16_t)  \
  X(lsh_f16_r64_ip_k8_d32678,  uint32_t, 64, __half, float, jasper::distance_func::INNER_PRODUCT,  8, uint16_t)  \
  X(lsh_f16_r64_ip_k16_d32678, uint32_t, 64, __half, float, jasper::distance_func::INNER_PRODUCT, 16, uint16_t)

#define DECLARE_LSH_CONFIGS_(id, IDX, R, DAT, DIST, FUNC, KR, PACKEDT)         \
  using cfg_##id = jasper::graph_config<IDX, R, DAT, DIST, FUNC, true, KR, PACKEDT>;

JASPER_FOR_EACH_LSH_CONFIG(DECLARE_LSH_CONFIGS_)
#undef DECLARE_LSH_CONFIGS_

// All configs use __half for storage. Source dtype (float / uint8) is
// applied at load time; it doesn't affect the chosen config.
template <jasper::distance_func Func>
bool config_matches(const std::string& datatype, uint64_t n_neighbors,
                    const std::string& distance, uint64_t R) {
  std::string expected_dist = (Func == jasper::distance_func::L2) ? "l2" : "ip";
  bool dtype_ok = (datatype == "float" || datatype == "uint8");
  return dtype_ok && n_neighbors == R && distance == expected_dist;
}

// LSH variant: also matches on k_ranks and selects the packed_t by dim
// (uint8_t packing → dim <= 128; uint16_t packing → dim > 128).
template <jasper::distance_func Func, uint32_t KR, typename PackedT>
bool lsh_config_matches(const std::string& datatype, uint64_t n_neighbors,
                        const std::string& distance, uint64_t R,
                        uint32_t k_ranks, uint32_t dim) {
  std::string expected_dist = (Func == jasper::distance_func::L2) ? "l2" : "ip";
  bool dtype_ok = (datatype == "float" || datatype == "uint8");
  bool dim_ok   = (sizeof(PackedT) == 1) ? (dim <= 128) : (dim > 128);
  return dtype_ok && n_neighbors == R && distance == expected_dist &&
         k_ranks == KR && dim_ok;
}

// ── Unpack thrust::pair results into separate arrays ───────────
__global__ void unpack_results_kernel(
    const thrust::pair<uint32_t, float>* __restrict__ pairs,
    int32_t* __restrict__ out_indices,
    float* __restrict__ out_distances,
    uint32_t total) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < total) {
    out_indices[i]  = static_cast<int32_t>(pairs[i].first);
    out_distances[i] = pairs[i].second;
  }
}

// ── Read ground truth ──────────────────────────────────────────
struct GroundTruth {
  uint32_t n_queries;
  uint32_t gt_k;
  std::vector<int32_t> indices;   // [n_queries * k]
  std::vector<float>   distances; // [n_queries * k]
};

GroundTruth read_groundtruth(const std::string& path, uint32_t k) {
  std::ifstream fin(path, std::ios::binary);
  if (!fin) throw std::runtime_error("Cannot open ground truth file: " + path);

  uint32_t n_queries, gt_k;
  fin.read(reinterpret_cast<char*>(&n_queries), sizeof(n_queries));
  fin.read(reinterpret_cast<char*>(&gt_k), sizeof(gt_k));

  if (gt_k < k)
    throw std::runtime_error(
      "Requested k=" + std::to_string(k) +
      " but ground truth only has k=" + std::to_string(gt_k));

  // Read full gt arrays, then slice to k
  std::vector<uint32_t> all_ids(static_cast<size_t>(n_queries) * gt_k);
  std::vector<float> all_dists(static_cast<size_t>(n_queries) * gt_k);
  fin.read(reinterpret_cast<char*>(all_ids.data()), n_queries * gt_k * sizeof(uint32_t));
  fin.read(reinterpret_cast<char*>(all_dists.data()), n_queries * gt_k * sizeof(float));
  fin.close();

  // Slice [:, :k]
  GroundTruth gt;
  gt.n_queries = n_queries;
  gt.gt_k = k;
  gt.indices.resize(static_cast<size_t>(n_queries) * k);
  gt.distances.resize(static_cast<size_t>(n_queries) * k);

  for (uint32_t q = 0; q < n_queries; q++) {
    for (uint32_t j = 0; j < k; j++) {
      gt.indices[q * k + j]   = static_cast<int32_t>(all_ids[q * gt_k + j]);
      gt.distances[q * k + j] = all_dists[q * gt_k + j];
    }
  }

  return gt;
}

// ── Compute recall ─────────────────────────────────────────────
float get_recall(const GroundTruth& gt,
                 const int32_t* result_indices,
                 uint32_t k, uint32_t n_queries) {
  uint64_t hits = 0;
  for (uint32_t q = 0; q < n_queries; q++) {
    std::set<int32_t> gt_set(
      gt.indices.data() + q * k,
      gt.indices.data() + q * k + k);
    for (uint32_t j = 0; j < k; j++) {
      if (gt_set.count(result_indices[q * k + j]))
        hits++;
    }
  }
  float recall = static_cast<float>(hits) / (n_queries * k);
  std::cout << "  Recall@" << k << ": "
            << std::fixed << std::setprecision(4) << recall << std::endl;
  return recall;
}

// ── Single search round ────────────────────────────────────────
template <typename GraphCfg, typename DataT>
void run_search_round(jasper::graph<GraphCfg>& graph,
                      DataT* d_queries,
                      uint32_t n_queries,
                      uint32_t dim,
                      uint32_t k,
                      uint32_t beam_width,
                      uint32_t limit,
                      const GroundTruth* gt,
                      bool print_throughput) {

  jasper::vector_view<DataT> query_view(d_queries, dim, n_queries, false);

  jasper::search_params params{
    .k          = k,
    .beam_width = beam_width,
    .limit      = limit,
    .get_visited = false,
  };

  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);

  cudaDeviceSynchronize();
  CUDA_CHECK(cudaGetLastError());

  cudaEventRecord(e0);
  auto result = jasper::search(graph, query_view, params);
  CUDA_CHECK(cudaGetLastError());
  cudaDeviceSynchronize();
  CUDA_CHECK(cudaGetLastError());
  cudaEventRecord(e1);
  cudaEventSynchronize(e1);

  float duration_ms = 0;
  cudaEventElapsedTime(&duration_ms, e0, e1);
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);

  if (print_throughput) {
    float throughput = (n_queries * 1000.0f) / duration_ms;
    std::cout << "  beam_width=" << beam_width
              << " limit=" << limit
              << " duration=" << std::fixed << std::setprecision(2) << duration_ms << "ms"
              << " throughput=" << std::setprecision(0) << throughput << " QPS"
              << std::endl;
  }

  // Unpack results
  uint32_t total = n_queries * k;
  int32_t* d_indices = nullptr;
  float* d_distances = nullptr;
  CUDA_CHECK(cudaMalloc(&d_indices, total * sizeof(int32_t)));
  CUDA_CHECK(cudaMalloc(&d_distances, total * sizeof(float)));

  uint32_t threads = 256;
  uint32_t blocks = (total + threads - 1) / threads;
  unpack_results_kernel<<<blocks, threads>>>(
    result.frontier, d_indices, d_distances, total);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<int32_t> h_indices(total);
  std::vector<float> h_distances(total);
  CUDA_CHECK(cudaMemcpy(h_indices.data(), d_indices, total * sizeof(int32_t), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_distances.data(), d_distances, total * sizeof(float), cudaMemcpyDeviceToHost));

  if (gt) {
    get_recall(*gt, h_indices.data(), k, n_queries);
  }

  cudaFree(d_indices);
  cudaFree(d_distances);
  cudaFree(result.frontier);
}

// ── Single directional-search round (LSH) ─────────────────────
template <typename GraphCfg, typename DataT>
void run_directional_round(jasper::graph<GraphCfg>& graph,
                           const jasper::lsh_globals<GraphCfg::k_ranks>& globals,
                           DataT* d_queries,
                           uint32_t n_queries,
                           uint32_t dim,
                           uint32_t k,
                           uint32_t beam_width,
                           uint32_t limit,
                           const GroundTruth* gt,
                           bool print_throughput) {

  jasper::vector_view<DataT> query_view(d_queries, dim, n_queries, false);

  jasper::search_params params{
    .k          = k,
    .beam_width = beam_width,
    .limit      = limit,
    .get_visited = false,
  };

  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);

  cudaDeviceSynchronize();
  CUDA_CHECK(cudaGetLastError());

  jasper::reset_phase_clocks();

  cudaEventRecord(e0);
  auto result = jasper::directional_search(graph, globals, query_view, params);
  CUDA_CHECK(cudaGetLastError());
  cudaDeviceSynchronize();
  CUDA_CHECK(cudaGetLastError());
  cudaEventRecord(e1);
  cudaEventSynchronize(e1);

  float duration_ms = 0;
  cudaEventElapsedTime(&duration_ms, e0, e1);
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);

  if (print_throughput) {
    float throughput = (n_queries * 1000.0f) / duration_ms;
    std::cout << "  beam_width=" << beam_width
              << " limit=" << limit
              << " duration=" << std::fixed << std::setprecision(2) << duration_ms << "ms"
              << " throughput=" << std::setprecision(0) << throughput << " QPS"
              << std::endl;
  }

  // Unpack results
  uint32_t total = n_queries * k;
  int32_t* d_indices = nullptr;
  float* d_distances = nullptr;
  CUDA_CHECK(cudaMalloc(&d_indices, total * sizeof(int32_t)));
  CUDA_CHECK(cudaMalloc(&d_distances, total * sizeof(float)));

  uint32_t threads = 256;
  uint32_t blocks = (total + threads - 1) / threads;
  unpack_results_kernel<<<blocks, threads>>>(
    result.frontier, d_indices, d_distances, total);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<int32_t> h_indices(total);
  std::vector<float> h_distances(total);
  CUDA_CHECK(cudaMemcpy(h_indices.data(), d_indices, total * sizeof(int32_t), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_distances.data(), d_distances, total * sizeof(float), cudaMemcpyDeviceToHost));

  if (gt) {
    get_recall(*gt, h_indices.data(), k, n_queries);
  }

  if (print_throughput) jasper::print_phase_clocks();

  cudaFree(d_indices);
  cudaFree(d_distances);
  cudaFree(result.frontier);
}

// ── Single PQ directional-search round (ADC) ───────────────────
template <typename GraphCfg, typename DataT>
void run_pq_round(jasper::graph<GraphCfg>& graph,
                  jasper::pq_codebooks_view<GraphCfg::pq_m, GraphCfg::pq_k> codebooks,
                  DataT* d_queries,
                  uint32_t n_queries,
                  uint32_t dim,
                  uint32_t k,
                  uint32_t beam_width,
                  uint32_t limit,
                  const GroundTruth* gt,
                  bool print_throughput) {

  jasper::vector_view<DataT> query_view(d_queries, dim, n_queries, false);

  jasper::search_params params{
    .k          = k,
    .beam_width = beam_width,
    .limit      = limit,
    .get_visited = false,
  };

  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);

  cudaDeviceSynchronize();
  CUDA_CHECK(cudaGetLastError());

  jasper::reset_phase_clocks();

  cudaEventRecord(e0);
  auto result = jasper::pq_search(graph, codebooks, query_view, params);
  CUDA_CHECK(cudaGetLastError());
  cudaDeviceSynchronize();
  CUDA_CHECK(cudaGetLastError());
  cudaEventRecord(e1);
  cudaEventSynchronize(e1);

  float duration_ms = 0;
  cudaEventElapsedTime(&duration_ms, e0, e1);
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);

  if (print_throughput) {
    float throughput = (n_queries * 1000.0f) / duration_ms;
    std::cout << "  beam_width=" << beam_width
              << " limit=" << limit
              << " duration=" << std::fixed << std::setprecision(2) << duration_ms << "ms"
              << " throughput=" << std::setprecision(0) << throughput << " QPS"
              << std::endl;
  }

  // Unpack results
  uint32_t total = n_queries * k;
  int32_t* d_indices = nullptr;
  float* d_distances = nullptr;
  CUDA_CHECK(cudaMalloc(&d_indices, total * sizeof(int32_t)));
  CUDA_CHECK(cudaMalloc(&d_distances, total * sizeof(float)));

  uint32_t threads = 256;
  uint32_t blocks = (total + threads - 1) / threads;
  unpack_results_kernel<<<blocks, threads>>>(
    result.frontier, d_indices, d_distances, total);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<int32_t> h_indices(total);
  std::vector<float> h_distances(total);
  CUDA_CHECK(cudaMemcpy(h_indices.data(), d_indices, total * sizeof(int32_t), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_distances.data(), d_distances, total * sizeof(float), cudaMemcpyDeviceToHost));

  if (gt) {
    get_recall(*gt, h_indices.data(), k, n_queries);
  }

  if (print_throughput) jasper::print_phase_clocks();

  cudaFree(d_indices);
  cudaFree(d_distances);
  cudaFree(result.frontier);
}

// Rotate a batch of vectors in place on device by the orthogonal matrix seeded
// with `seed`. Rows are [n_vectors, padded_dim] row-major __half; only the
// first `dim` columns are populated, pad lanes stay zero.
static void rotate_device_vectors(__half* d_vectors,
                                  uint32_t n_vectors,
                                  uint32_t dim,
                                  uint32_t padded_dim,
                                  uint32_t seed,
                                  cublasHandle_t handle) {
  if (n_vectors == 0) return;

  std::vector<float> h_P_f(static_cast<size_t>(dim) * dim);
  std::vector<float> h_Pt_f(static_cast<size_t>(dim) * dim);
  jasper::set_rotation_matrix(dim, h_P_f.data(), h_Pt_f.data(), seed);

  std::vector<__half> h_P(h_P_f.size());
  for (size_t i = 0; i < h_P.size(); ++i)
    h_P[i] = static_cast<__half>(h_P_f[i]);

  __half* d_P = nullptr;
  CUDA_CHECK(cudaMalloc(&d_P, sizeof(__half) * dim * dim));
  CUDA_CHECK(cudaMemcpy(d_P, h_P.data(), sizeof(__half) * dim * dim, cudaMemcpyHostToDevice));

  size_t data_bytes = sizeof(__half) * static_cast<size_t>(n_vectors) * padded_dim;
  __half* d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_out, data_bytes));
  CUDA_CHECK(cudaMemset(d_out, 0, data_bytes));

  const float one = 1.0f, zero = 0.0f;
  cublasGemmEx(
    handle,
    CUBLAS_OP_T, CUBLAS_OP_N,
    dim, n_vectors, dim,
    &one,
    d_P,       CUDA_R_16F, dim,
    d_vectors, CUDA_R_16F, padded_dim,
    &zero,
    d_out,     CUDA_R_16F, padded_dim,
    CUBLAS_COMPUTE_32F,
    CUBLAS_GEMM_DEFAULT);

  CUDA_CHECK(cudaMemcpy(d_vectors, d_out, data_bytes, cudaMemcpyDeviceToDevice));
  cudaFree(d_out);
  cudaFree(d_P);
}

// Rotate every stored vector of a graph in place. Orthogonal rotation preserves
// L2/inner-product, so the graph topology is unchanged; this only decorrelates
// dimensions so the axis-aligned LSH hashing matches what create_lsh_index
// produces via prerotate. Must run before generate_lsh_globals /
// populate_edge_lsh, and the queries must be rotated with the same seed. Works
// on host- or device-resident graphs (host segments are staged to device for
// the cuBLAS GEMM, then copied back).
template <typename GraphCfg>
void rotate_graph_vectors(jasper::graph<GraphCfg>& g, uint32_t seed) {
  using data_t    = typename GraphCfg::data_t;
  using segment_t = jasper::graph_segment<GraphCfg>;
  static_assert(std::is_same<data_t, __half>::value,
                "rotate_graph_vectors requires __half (f16) data_t");

  const uint32_t dim        = g.dim;
  const uint32_t padded_dim = g.get_padded_dim();

  cublasHandle_t handle;
  cublasCreate(&handle);

  std::vector<segment_t> h_segs(g.segments.begin(), g.segments.end());
  for (auto& seg : h_segs) {
    uint32_t n = static_cast<uint32_t>(seg.n_vectors);
    if (n == 0) continue;

    if (g.on_host) {
      // Stage this segment's vectors on device, rotate, then copy back to the
      // (pinned) host buffer that seg.vectors.data points into.
      size_t bytes = sizeof(__half) * static_cast<size_t>(n) * padded_dim;
      __half* d_vecs = nullptr;
      CUDA_CHECK(cudaMalloc(&d_vecs, bytes));
      CUDA_CHECK(cudaMemcpy(d_vecs, seg.vectors.data, bytes, cudaMemcpyHostToDevice));
      rotate_device_vectors(d_vecs, n, dim, padded_dim, seed, handle);
      CUDA_CHECK(cudaMemcpy(seg.vectors.data, d_vecs, bytes, cudaMemcpyDeviceToHost));
      cudaFree(d_vecs);
    } else {
      rotate_device_vectors(seg.vectors.data, n, dim, padded_dim, seed, handle);
    }
  }
  cublasDestroy(handle);
}

// ── Load graph + populate LSH + run directional benchmark ──────
template <typename GraphCfg, typename DataT>
void load_and_bench_lsh(const std::string& index_path,
                        const std::string& query_path,
                        const std::string& gt_path,
                        const std::string& src_dtype,
                        uint32_t dim,
                        uint32_t k,
                        uint32_t rotate_seed,
                        bool do_rotate,
                        bool on_host,
                        const std::vector<std::pair<uint32_t, uint32_t>>& beam_limit_pairs) {

  // Load graph (LSH config; edge_lsh storage is allocated lazily below).
  // Rotation, generate_lsh_globals and populate_edge_lsh all work on host now,
  // so honor --on_host directly at load time.
  std::cout << "Loading index..." << std::endl;
  auto graph = jasper::load_graph_from_file<GraphCfg>(index_path, dim, on_host);
  std::cout << "  " << graph.n_vectors << " vectors, dim=" << graph.dim << std::endl;

  // Rotate stored vectors so the LSH hashing matches create_lsh_index's
  // prerotate. Distance-preserving, so graph edges are unaffected.
  if (do_rotate) {
    std::cout << "Rotating stored vectors (seed=" << rotate_seed << ")..." << std::endl;
    rotate_graph_vectors<GraphCfg>(graph, rotate_seed);
    cudaDeviceSynchronize();
  }

  // Build LSH globals + per-edge LSH coords from the (rotated) graph.
  std::cout << "Sampling for LSH globals..." << std::endl;
  auto t0 = std::chrono::steady_clock::now();
  auto lsh_globals = graph.generate_lsh_globals(/*n_samples=*/32768);
  cudaDeviceSynchronize();
  auto t1 = std::chrono::steady_clock::now();

  std::cout << "Populating edge LSH..." << std::endl;
  graph.populate_edge_lsh();
  cudaDeviceSynchronize();
  auto t2 = std::chrono::steady_clock::now();

  std::cout << "  generate_lsh_globals: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << " ms\n";
  std::cout << "  populate_edge_lsh:    "
            << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms\n";

  // Load queries, casting src_dtype -> DataT (__half)
  std::cout << "Loading queries..." << std::endl;
  jasper::vector_view<DataT> h_queries_view;
  if (src_dtype == "float") {
    h_queries_view = jasper::load_vectors_from_file_cast<DataT, float>(query_path);
  } else if (src_dtype == "uint8") {
    h_queries_view = jasper::load_vectors_from_file_cast<DataT, uint8_t>(query_path);
  } else {
    throw std::runtime_error("Unsupported source dtype: " + src_dtype);
  }
  if (!h_queries_view.data)
    throw std::runtime_error("Failed to load queries from: " + query_path);

  if (h_queries_view.dim != dim)
    throw std::runtime_error(
      "Query file dim=" + std::to_string(h_queries_view.dim) +
      " does not match --dim=" + std::to_string(dim));

  uint32_t n_queries = h_queries_view.n_vectors;
  std::cout << "  " << n_queries << " queries, dim=" << h_queries_view.dim << std::endl;

  auto d_queries_view = h_queries_view.to_device();
  cudaFreeHost(h_queries_view.data);
  DataT* d_queries = d_queries_view.data;

  // Rotate queries with the same seed used for the stored vectors.
  if (do_rotate) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    rotate_device_vectors(d_queries, n_queries, dim,
                          d_queries_view.padded_dim, rotate_seed, handle);
    cublasDestroy(handle);
  }

  // Load ground truth
  GroundTruth gt;
  bool has_gt = !gt_path.empty();
  if (has_gt) {
    std::cout << "Loading ground truth..." << std::endl;
    gt = read_groundtruth(gt_path, k);
    std::cout << "  " << gt.n_queries << " queries, k=" << gt.gt_k << std::endl;
    if (gt.n_queries != n_queries)
      throw std::runtime_error("Ground truth query count mismatch");
  }

  // Warmup
  std::cout << "\nWarmup..." << std::endl;
  auto [warmup_bw, warmup_limit] = beam_limit_pairs.front();
  run_directional_round<GraphCfg, DataT>(
    graph, lsh_globals, d_queries, n_queries, dim, k,
    warmup_bw, warmup_limit, nullptr, false);

  // Benchmark rounds
  std::cout << "\n=== Directional Beam Search Benchmark (k=" << k << ") ===" << std::endl;
  for (auto& [bw, lim] : beam_limit_pairs) {
    run_directional_round<GraphCfg, DataT>(
      graph, lsh_globals, d_queries, n_queries, dim, k,
      bw, lim, has_gt ? &gt : nullptr, true);
  }

  cudaFree(d_queries);
  graph.deallocate();
}

// ── Load graph + train PQ + run PQ directional benchmark ───────
template <typename GraphCfg, typename DataT>
void load_and_bench_pq(const std::string& index_path,
                       const std::string& query_path,
                       const std::string& gt_path,
                       const std::string& src_dtype,
                       uint32_t dim,
                       uint32_t k,
                       uint32_t rotate_seed,
                       bool do_rotate,
                       uint32_t pq_train,
                       bool on_host,
                       const std::vector<std::pair<uint32_t, uint32_t>>& beam_limit_pairs) {

  // Load graph (LSH config; edge_pqs storage is allocated lazily below).
  std::cout << "Loading index..." << std::endl;
  auto graph = jasper::load_graph_from_file<GraphCfg>(index_path, dim, on_host);
  std::cout << "  " << graph.n_vectors << " vectors, dim=" << graph.dim << std::endl;

  // Rotate stored vectors so the PQ residuals match create_lsh_index's
  // prerotate. Distance-preserving, so graph edges are unaffected.
  if (do_rotate) {
    std::cout << "Rotating stored vectors (seed=" << rotate_seed << ")..." << std::endl;
    rotate_graph_vectors<GraphCfg>(graph, rotate_seed);
    cudaDeviceSynchronize();
  }

  // Train PQ codebooks on sampled edge residuals, then encode every edge.
  std::cout << "Training PQ codebooks..." << std::endl;
  auto t0 = std::chrono::steady_clock::now();
  auto pq_codebooks = graph.generate_pq_codebooks(pq_train);
  cudaDeviceSynchronize();
  auto t1 = std::chrono::steady_clock::now();

  std::cout << "Populating edge PQ codes..." << std::endl;
  graph.populate_edge_pq(pq_codebooks);
  graph.compute_vector_norms();  // exact ||v||² for the PQ L2 estimator
  cudaDeviceSynchronize();
  auto t2 = std::chrono::steady_clock::now();

  std::cout << "  generate_pq_codebooks: "
            << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() << " ms\n";
  std::cout << "  populate_edge_pq:      "
            << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms\n";

  // Load queries, casting src_dtype -> DataT (__half)
  std::cout << "Loading queries..." << std::endl;
  jasper::vector_view<DataT> h_queries_view;
  if (src_dtype == "float") {
    h_queries_view = jasper::load_vectors_from_file_cast<DataT, float>(query_path);
  } else if (src_dtype == "uint8") {
    h_queries_view = jasper::load_vectors_from_file_cast<DataT, uint8_t>(query_path);
  } else {
    throw std::runtime_error("Unsupported source dtype: " + src_dtype);
  }
  if (!h_queries_view.data)
    throw std::runtime_error("Failed to load queries from: " + query_path);

  if (h_queries_view.dim != dim)
    throw std::runtime_error(
      "Query file dim=" + std::to_string(h_queries_view.dim) +
      " does not match --dim=" + std::to_string(dim));

  uint32_t n_queries = h_queries_view.n_vectors;
  std::cout << "  " << n_queries << " queries, dim=" << h_queries_view.dim << std::endl;

  auto d_queries_view = h_queries_view.to_device();
  cudaFreeHost(h_queries_view.data);
  DataT* d_queries = d_queries_view.data;

  // Rotate queries with the same seed used for the stored vectors.
  if (do_rotate) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    rotate_device_vectors(d_queries, n_queries, dim,
                          d_queries_view.padded_dim, rotate_seed, handle);
    cublasDestroy(handle);
  }

  // Load ground truth
  GroundTruth gt;
  bool has_gt = !gt_path.empty();
  if (has_gt) {
    std::cout << "Loading ground truth..." << std::endl;
    gt = read_groundtruth(gt_path, k);
    std::cout << "  " << gt.n_queries << " queries, k=" << gt.gt_k << std::endl;
    if (gt.n_queries != n_queries)
      throw std::runtime_error("Ground truth query count mismatch");
  }

  // Warmup
  std::cout << "\nWarmup..." << std::endl;
  auto [warmup_bw, warmup_limit] = beam_limit_pairs.front();
  run_pq_round<GraphCfg, DataT>(
    graph, pq_codebooks.view(), d_queries, n_queries, dim, k,
    warmup_bw, warmup_limit, nullptr, false);

  // Benchmark rounds
  std::cout << "\n=== PQ Directional Beam Search Benchmark (k=" << k << ") ===" << std::endl;
  for (auto& [bw, lim] : beam_limit_pairs) {
    run_pq_round<GraphCfg, DataT>(
      graph, pq_codebooks.view(), d_queries, n_queries, dim, k,
      bw, lim, has_gt ? &gt : nullptr, true);
  }

  cudaFree(d_queries);
  pq_codebooks.free();
  graph.deallocate();
}

// ── Load graph + run all beam widths ───────────────────────────
template <typename GraphCfg, typename DataT>
void load_and_bench(const std::string& index_path,
                    const std::string& query_path,
                    const std::string& gt_path,
                    const std::string& src_dtype,
                    uint32_t dim,
                    uint32_t k,
                    bool on_host,
                    const std::vector<std::pair<uint32_t, uint32_t>>& beam_limit_pairs) {

  // Load graph (optionally directly into host memory).
  std::cout << "Loading index..." << std::endl;
  auto graph = jasper::load_graph_from_file<GraphCfg>(index_path, dim, on_host);
  std::cout << "  " << graph.n_vectors << " vectors, dim=" << graph.dim << std::endl;

  // Load queries, casting src_dtype -> DataT (__half)
  std::cout << "Loading queries..." << std::endl;
  jasper::vector_view<DataT> h_queries_view;
  if (src_dtype == "float") {
    h_queries_view = jasper::load_vectors_from_file_cast<DataT, float>(query_path);
  } else if (src_dtype == "uint8") {
    h_queries_view = jasper::load_vectors_from_file_cast<DataT, uint8_t>(query_path);
  } else {
    throw std::runtime_error("Unsupported source dtype: " + src_dtype);
  }
  if (!h_queries_view.data)
    throw std::runtime_error("Failed to load queries from: " + query_path);

  if (h_queries_view.dim != dim)
    throw std::runtime_error(
      "Query file dim=" + std::to_string(h_queries_view.dim) +
      " does not match --dim=" + std::to_string(dim));

  uint32_t n_queries = h_queries_view.n_vectors;
  std::cout << "  " << n_queries << " queries, dim=" << h_queries_view.dim << std::endl;

  // Upload padded buffer to device
  auto d_queries_view = h_queries_view.to_device();
  cudaFreeHost(h_queries_view.data);
  DataT* d_queries = d_queries_view.data;

  // Load ground truth
  GroundTruth gt;
  bool has_gt = !gt_path.empty();
  if (has_gt) {
    std::cout << "Loading ground truth..." << std::endl;
    gt = read_groundtruth(gt_path, k);
    std::cout << "  " << gt.n_queries << " queries, k=" << gt.gt_k << std::endl;
    if (gt.n_queries != n_queries)
      throw std::runtime_error("Ground truth query count mismatch");
  }

  // Warmup
  std::cout << "\nWarmup..." << std::endl;
  auto [warmup_bw, warmup_limit] = beam_limit_pairs.front();
  run_search_round<GraphCfg, DataT>(
    graph, d_queries, n_queries, dim, k,
    warmup_bw, warmup_limit, nullptr, false);

  // Benchmark rounds
  std::cout << "\n=== Benchmark (k=" << k << ") ===" << std::endl;
  for (auto& [bw, lim] : beam_limit_pairs) {
    run_search_round<GraphCfg, DataT>(
      graph, d_queries, n_queries, dim, k,
      bw, lim, has_gt ? &gt : nullptr, true);
  }

  // Cleanup
  cudaFree(d_queries);
  graph.deallocate();
}

// ── Parse "beam_width:limit" pairs ─────────────────────────────
std::vector<std::pair<uint32_t, uint32_t>> parse_beam_limit_pairs(
    const std::vector<std::string>& args) {
  std::vector<std::pair<uint32_t, uint32_t>> pairs;
  for (auto& s : args) {
    auto colon = s.find(':');
    if (colon == std::string::npos)
      throw std::runtime_error("Expected beam_width:limit format, got: " + s);
    uint32_t bw  = static_cast<uint32_t>(std::stoul(s.substr(0, colon)));
    uint32_t lim = static_cast<uint32_t>(std::stoul(s.substr(colon + 1)));
    pairs.emplace_back(bw, lim);
  }
  return pairs;
}

// ── Argument parsing ───────────────────────────────────────────
int main(int argc, char** argv) {
  argparse::ArgumentParser program("run_query");

  program.add_argument("--index", "-i")
    .required()
    .help("Input index filename.");

  program.add_argument("--queries", "-q")
    .required()
    .help("Input query vectors filename.");

  program.add_argument("--groundtruth", "-g")
    .default_value(std::string{})
    .help("Ground truth .bin file (optional).");

  program.add_argument("--datatype", "-t")
    .required()
    .choices("uint8", "float")
    .help("Vector datatype [\"uint8\", \"float\"] (cast to __half internally)");

  program.add_argument("--distance", "-d")
    .default_value(std::string{"l2"})
    .choices("l2", "ip")
    .help("Distance type [\"l2\" (default), \"ip\"]");

  program.add_argument("--n_neighbors", "-n")
    .default_value(uint64_t{64})
    .scan<'u', uint64_t>()
    .help("Number of neighbors per vector in the index (must match index)");

  program.add_argument("--dim")
    .required()
    .scan<'u', uint32_t>()
    .help("Vector dimension");

  program.add_argument("--k", "-k")
    .default_value(uint32_t{10})
    .scan<'u', uint32_t>()
    .help("Number of nearest neighbors to return");

  program.add_argument("--beam_limits", "-b")
    .default_value(std::vector<std::string>{
      "1:128", "2:128", "4:128", "8:128", "16:128", "32:128", "64:128", "128:256", "256:512"
    })
    .nargs(argparse::nargs_pattern::at_least_one)
    .help("beam_width:limit pairs to benchmark (e.g. 16:128 64:256)");

  program.add_argument("--lsh")
    .default_value(false)
    .implicit_value(true)
    .help("Populate LSH on the loaded index and run directional search "
          "(L2 or IP). Index file needs no LSH data; it is generated at load.");

  program.add_argument("--pq")
    .default_value(false)
    .implicit_value(true)
    .help("Train PQ codebooks on the loaded index and run PQ directional "
          "search (ADC). Index file needs no PQ data; it is generated at load. "
          "Mutually exclusive with --lsh.");

  program.add_argument("--pq_train")
    .default_value(uint32_t{40000})
    .scan<'u', uint32_t>()
    .help("Number of sampled edge residuals for PQ k-means training. "
          "Only used with --pq. Default 40000.");

  program.add_argument("--k_ranks", "-r")
    .default_value(uint32_t{4})
    .scan<'u', uint32_t>()
    .choices(4u, 8u, 16u)
    .help("LSH k_ranks / PQ subquantizers M (4, 8, or 16). "
          "Used with --lsh or --pq.");

  program.add_argument("--on_host")
    .default_value(false)
    .implicit_value(true)
    .help("Move the loaded graph to host before querying.");

  program.add_argument("--no_rotate")
    .default_value(false)
    .implicit_value(true)
    .help("Skip rotation of stored vectors / queries in --lsh mode. By default "
          "both are rotated (seed 42) to match create_lsh_index's prerotate.");

  program.add_argument("--rotate_seed")
    .default_value(uint32_t{42})
    .scan<'u', uint32_t>()
    .help("Rotation seed for --lsh mode (must match construction). Default 42.");

  try {
    program.parse_args(argc, argv);
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    return 1;
  }

  auto index_path   = program.get<std::string>("--index");
  auto query_path   = program.get<std::string>("--queries");
  auto gt_path      = program.get<std::string>("--groundtruth");
  auto datatype     = program.get<std::string>("--datatype");
  auto distance     = program.get<std::string>("--distance");
  auto n_neighbors  = program.get<uint64_t>("--n_neighbors");
  auto dim          = program.get<uint32_t>("--dim");
  auto k            = program.get<uint32_t>("--k");
  auto beam_args    = program.get<std::vector<std::string>>("--beam_limits");
  auto use_lsh      = program.get<bool>("--lsh");
  auto use_pq       = program.get<bool>("--pq");
  auto on_host      = program.get<bool>("--on_host");
  auto k_ranks      = program.get<uint32_t>("--k_ranks");
  auto pq_train     = program.get<uint32_t>("--pq_train");
  auto no_rotate    = program.get<bool>("--no_rotate");
  auto rotate_seed  = program.get<uint32_t>("--rotate_seed");

  if (use_lsh && use_pq) {
    std::cerr << "Error: --lsh and --pq are mutually exclusive" << std::endl;
    return 1;
  }

  std::vector<std::pair<uint32_t, uint32_t>> beam_limit_pairs;
  try {
    beam_limit_pairs = parse_beam_limit_pairs(beam_args);
  } catch (const std::exception& err) {
    std::cerr << "Error parsing --beam_limits: " << err.what() << std::endl;
    return 1;
  }

  std::cout << "=== run_query ===" << std::endl;
  std::cout << "  index:        " << index_path << std::endl;
  std::cout << "  queries:      " << query_path << std::endl;
  std::cout << "  groundtruth:  " << (gt_path.empty() ? "(none)" : gt_path) << std::endl;
  std::cout << "  datatype:     " << datatype << std::endl;
  std::cout << "  distance:     " << distance << std::endl;
  std::cout << "  n_neighbors:  " << n_neighbors << std::endl;
  std::cout << "  dim:          " << dim << std::endl;
  std::cout << "  k:            " << k << std::endl;
  std::cout << "  beam_limits:  ";
  for (auto& [bw, lim] : beam_limit_pairs)
    std::cout << bw << ":" << lim << " ";
  std::cout << std::endl;
  std::cout << "  lsh:          " << (use_lsh ? "true" : "false") << std::endl;
  std::cout << "  pq:           " << (use_pq ? "true" : "false") << std::endl;
  std::cout << "  on_host:      " << (on_host ? "true" : "false") << std::endl;
  if (use_lsh || use_pq) {
    std::cout << "  k_ranks:      " << k_ranks << std::endl;
    if (use_pq)
      std::cout << "  pq_train:     " << pq_train << std::endl;
    std::cout << "  rotate:       " << (no_rotate ? "false" : "true") << std::endl;
    if (!no_rotate)
      std::cout << "  rotate_seed:  " << rotate_seed << std::endl;
  }

  try {
    bool dispatched = false;

    if (use_pq) {
      // PQ requires directional (use_lsh) storage, so it dispatches over the
      // same config table; k_ranks selects the number of PQ subquantizers M.
      #define TRY_DISPATCH_PQ(id, IDX, R, DAT, DIST, FUNC, KR, PACKEDT)        \
        if (!dispatched &&                                                     \
            lsh_config_matches<FUNC, KR, PACKEDT>(                             \
                datatype, n_neighbors, distance, R, k_ranks, dim)) {          \
          std::cout << "  Config: " #id << std::endl;                          \
          load_and_bench_pq<cfg_##id, DAT>(                                    \
              index_path, query_path, gt_path, datatype,                       \
              dim, k, rotate_seed, !no_rotate, pq_train, on_host,             \
              beam_limit_pairs);                                               \
          dispatched = true;                                                   \
        }

      JASPER_FOR_EACH_LSH_CONFIG(TRY_DISPATCH_PQ)
      #undef TRY_DISPATCH_PQ

      if (!dispatched) {
        throw std::runtime_error(
          "Unsupported PQ config: datatype=" + datatype +
          ", n_neighbors=" + std::to_string(n_neighbors) +
          ", distance=" + distance +
          ", k_ranks=" + std::to_string(k_ranks));
      }
    } else if (use_lsh) {
      #define TRY_DISPATCH_LSH(id, IDX, R, DAT, DIST, FUNC, KR, PACKEDT)       \
        if (!dispatched &&                                                     \
            lsh_config_matches<FUNC, KR, PACKEDT>(                             \
                datatype, n_neighbors, distance, R, k_ranks, dim)) {          \
          std::cout << "  Config: " #id << std::endl;                          \
          load_and_bench_lsh<cfg_##id, DAT>(                                   \
              index_path, query_path, gt_path, datatype,                       \
              dim, k, rotate_seed, !no_rotate, on_host, beam_limit_pairs);    \
          dispatched = true;                                                   \
        }

      JASPER_FOR_EACH_LSH_CONFIG(TRY_DISPATCH_LSH)
      #undef TRY_DISPATCH_LSH

      if (!dispatched) {
        throw std::runtime_error(
          "Unsupported LSH config: datatype=" + datatype +
          ", n_neighbors=" + std::to_string(n_neighbors) +
          ", distance=" + distance +
          ", k_ranks=" + std::to_string(k_ranks));
      }
    } else {
      #define TRY_DISPATCH(id, IDX, R, DAT, DIST, FUNC)                         \
        if (!dispatched && config_matches<FUNC>(datatype, n_neighbors, distance, R)) { \
          std::cout << "  Config: " #id << std::endl;                           \
          load_and_bench<cfg_##id, DAT>(                                        \
              index_path, query_path, gt_path, datatype,                        \
              dim, k, on_host, beam_limit_pairs);                              \
          dispatched = true;                                                     \
        }

      JASPER_FOR_EACH_CONFIG(TRY_DISPATCH)
      #undef TRY_DISPATCH

      if (!dispatched) {
        throw std::runtime_error(
          "Unsupported config: datatype=" + datatype +
          ", n_neighbors=" + std::to_string(n_neighbors) +
          ", distance=" + distance);
      }
    }
  } catch (const std::exception& err) {
    std::cerr << "Error: " << err.what() << std::endl;
    return 1;
  }

  return 0;
}