#include <argparse/argparse.hpp>
#include <thrust/pair.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <vector>
#include <cstdint>
#include <cstring>
#include <set>
#include <stdexcept>
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
                << " - " << cudaGetErrorString(err) << std::endl;              \
      throw std::runtime_error(cudaGetErrorString(err));                       \
    }                                                                          \
  } while (0)

// (CONFIG_ID, INDEX_T, N_NEIGHBORS, DATA_T, DISTANCE_T, DIST_FUNC, K_RANKS)
#define JASPER_FOR_EACH_CONFIG(X)                                              \
  X(f16_r16_l2_k4,  uint32_t, 16, __half, float, jasper::distance_func::L2,  4)  \
  X(f16_r16_l2_k8,  uint32_t, 16, __half, float, jasper::distance_func::L2,  8)  \
  X(f16_r16_l2_k16, uint32_t, 16, __half, float, jasper::distance_func::L2, 16) \
  X(f16_r32_l2_k4,  uint32_t, 32, __half, float, jasper::distance_func::L2,  4)  \
  X(f16_r32_l2_k8,  uint32_t, 32, __half, float, jasper::distance_func::L2,  8)  \
  X(f16_r32_l2_k16, uint32_t, 32, __half, float, jasper::distance_func::L2, 16)  
  // X(f16_r64_ip_k4,  uint32_t, 64, __half, float, jasper::distance_func::INNER_PRODUCT,  4)  \
  // X(f16_r64_ip_k8,  uint32_t, 64, __half, float, jasper::distance_func::INNER_PRODUCT,  8)  \
  // X(f16_r64_ip_k16, uint32_t, 64, __half, float, jasper::distance_func::INNER_PRODUCT, 16)

// Graph + construct config types. K_RANKS is now a macro parameter.
#define DECLARE_CONFIGS(id, IDX, R, DAT, DIST, FUNC, KR)                       \
  using cfg_##id = jasper::graph_config<IDX, R, DAT, DIST, FUNC, true, KR>;    \
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
    throw std::runtime_error(
      "Requested k=" + std::to_string(k) +
      " but ground truth only has k=" + std::to_string(gt_k));

  std::vector<uint32_t> all_ids  (static_cast<size_t>(n_queries) * gt_k);
  std::vector<float>    all_dists(static_cast<size_t>(n_queries) * gt_k);
  fin.read(reinterpret_cast<char*>(all_ids.data()),
           n_queries * gt_k * sizeof(uint32_t));
  fin.read(reinterpret_cast<char*>(all_dists.data()),
           n_queries * gt_k * sizeof(float));
  fin.close();

  GroundTruth gt;
  gt.n_queries = n_queries;
  gt.gt_k      = k;
  gt.indices.resize  (static_cast<size_t>(n_queries) * k);
  gt.distances.resize(static_cast<size_t>(n_queries) * k);
  for (uint32_t q = 0; q < n_queries; ++q) {
    for (uint32_t j = 0; j < k; ++j) {
      gt.indices  [q * k + j] = static_cast<int32_t>(all_ids[q * gt_k + j]);
      gt.distances[q * k + j] = all_dists[q * gt_k + j];
    }
  }
  return gt;
}

float get_recall(const GroundTruth& gt,
                 const int32_t* result_indices,
                 uint32_t k, uint32_t n_queries) {
  uint64_t hits = 0;
  for (uint32_t q = 0; q < n_queries; ++q) {
    std::set<int32_t> gt_set(
      gt.indices.data() + q * k,
      gt.indices.data() + q * k + k);
    for (uint32_t j = 0; j < k; ++j) {
      if (gt_set.count(result_indices[q * k + j])) ++hits;
    }
  }
  float recall = static_cast<float>(hits) / (n_queries * k);
  std::cout << "    Recall@" << k << ": "
            << std::fixed << std::setprecision(4) << recall << std::endl;
  return recall;
}

// ── Generic helper: time + unpack + optional recall ────────────
// The single SearchFn lambda is the only thing that differs between the
// conventional and directional rounds, so all timing/unpack/recall
// boilerplate lives here.
template <typename SearchFn>
void run_round_generic(
    uint32_t n_queries,
    uint32_t k,
    uint32_t beam_width,
    uint32_t limit,
    const GroundTruth* gt,
    bool print_throughput,
    SearchFn&& search_fn) {

  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);

  cudaDeviceSynchronize();
  CUDA_CHECK(cudaGetLastError());

  cudaEventRecord(e0);
  auto result = search_fn();
  CUDA_CHECK(cudaGetLastError());
  cudaDeviceSynchronize();
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

  // Unpack to host for recall.
  uint32_t total = n_queries * k;
  int32_t* d_indices   = nullptr;
  float*   d_distances = nullptr;
  CUDA_CHECK(cudaMalloc(&d_indices,   total * sizeof(int32_t)));
  CUDA_CHECK(cudaMalloc(&d_distances, total * sizeof(float)));

  uint32_t threads = 256;
  uint32_t blocks  = (total + threads - 1) / threads;
  unpack_results_kernel<<<blocks, threads>>>(
    result.frontier, d_indices, d_distances, total);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<int32_t> h_indices  (total);
  std::vector<float>   h_distances(total);
  CUDA_CHECK(cudaMemcpy(h_indices.data(),   d_indices,   total * sizeof(int32_t), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_distances.data(), d_distances, total * sizeof(float),   cudaMemcpyDeviceToHost));

  if (gt) get_recall(*gt, h_indices.data(), k, n_queries);

  cudaFree(d_indices);
  cudaFree(d_distances);
  cudaFree(result.frontier);
}

// ── Single conventional beam-search round ──────────────────────
template <typename GraphCfg, typename DataT>
void run_beam_search_round(
    jasper::graph<GraphCfg>& graph,
    DataT*    d_queries,
    uint32_t  n_queries,
    uint32_t  dim,
    uint32_t  k,
    uint32_t  beam_width,
    uint32_t  limit,
    const GroundTruth* gt,
    bool      print_throughput) {

  jasper::vector_view<DataT> query_view(d_queries, dim, n_queries, false);
  jasper::search_params params{
    .k           = k,
    .beam_width  = beam_width,
    .limit       = limit,
    .get_visited = false,
  };

  run_round_generic(
    n_queries, k, beam_width, limit, gt, print_throughput,
    [&]() { return jasper::search(graph, query_view, params); });
}

// ── Single directional-search round ────────────────────────────
template <typename GraphCfg, typename DataT>
void run_directional_round(
    jasper::graph<GraphCfg>&                       graph,
    const jasper::lsh_globals<GraphCfg::k_ranks>&  globals,
    DataT*    d_queries,
    uint32_t  n_queries,
    uint32_t  dim,
    uint32_t  k,
    uint32_t  beam_width,
    uint32_t  limit,
    const GroundTruth* gt,
    bool      print_throughput) {

  jasper::vector_view<DataT> query_view(d_queries, dim, n_queries, false);
  jasper::search_params params{
    .k           = k,
    .beam_width  = beam_width,
    .limit       = limit,
    .get_visited = false,
  };

  jasper::reset_phase_clocks();
  run_round_generic(
    n_queries, k, beam_width, limit, gt, print_throughput,
    [&]() {
      return jasper::directional_search(graph, globals, query_view, params);
    });
  jasper::print_phase_clocks();
}

template <typename DataT>
// Rotate query vectors using the same seed used during construction
void rotate_queries(DataT* d_queries, uint32_t n_queries, uint32_t dim,
                    uint32_t padded_dim, uint32_t seed, cudaStream_t stream = 0) {
    std::vector<float> h_P_f(dim * dim), h_Pt_f(dim * dim);
    jasper::set_rotation_matrix(dim, h_P_f.data(), h_Pt_f.data(), seed);

    std::vector<__half> h_P(h_P_f.size());
    for (size_t i = 0; i < h_P.size(); ++i)
        h_P[i] = static_cast<__half>(h_P_f[i]);

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
    cublasGemmEx(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        dim,       // m
        n_queries, // n
        dim,       // k
        &one,
        d_P,       CUDA_R_16F, dim,
        d_queries, CUDA_R_16F, padded_dim,
        &zero,
        d_out,     CUDA_R_16F, padded_dim,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT);
    cublasDestroy(handle);

    cudaMemcpy(d_queries, d_out, data_bytes, cudaMemcpyDeviceToDevice);
    cudaFree(d_out);
    cudaFree(d_P);
}

// ── Benchmark driver: runs BOTH conventional and directional ───
template <typename GraphCfg, typename DataT>
void benchmark_all(
    jasper::graph<GraphCfg>&                       graph,
    const jasper::lsh_globals<GraphCfg::k_ranks>&  globals,
    const std::string& query_path,
    const std::string& gt_path,
    const std::string& src_dtype,
    uint32_t dim,
    uint32_t k,
    const std::vector<std::pair<uint32_t, uint32_t>>& beam_limit_pairs) {

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

  // apply rotation to query
  rotate_queries(d_queries, n_queries, dim, d_queries_view.padded_dim, /*seed=*/42);

  GroundTruth gt;
  bool has_gt = !gt_path.empty();
  if (has_gt) {
    std::cout << "Loading ground truth..." << std::endl;
    gt = read_groundtruth(gt_path, k);
    std::cout << "  " << gt.n_queries << " queries, k=" << gt.gt_k << std::endl;
    if (gt.n_queries != n_queries)
      throw std::runtime_error("Ground truth query count mismatch");
  }

  auto [warmup_bw, warmup_limit] = beam_limit_pairs.front();

  // ─── Conventional beam search ───────────────────────────────
  std::cout << "\n=== Conventional Beam Search Benchmark (k=" << k << ") ===" << std::endl;
  std::cout << "Warmup..." << std::endl;
  run_beam_search_round<GraphCfg, DataT>(
    graph, d_queries, n_queries, dim, k,
    warmup_bw, warmup_limit, nullptr, false);

  for (auto& [bw, lim] : beam_limit_pairs) {
    run_beam_search_round<GraphCfg, DataT>(
      graph, d_queries, n_queries, dim, k,
      bw, lim, has_gt ? &gt : nullptr, true);
  }

  // ─── Directional beam search ────────────────────────────────
  std::cout << "\n=== Directional Beam Search Benchmark (k=" << k << ") ===" << std::endl;
  std::cout << "Warmup..." << std::endl;
  run_directional_round<GraphCfg, DataT>(
    graph, globals, d_queries, n_queries, dim, k,
    warmup_bw, warmup_limit, nullptr, false);

  for (auto& [bw, lim] : beam_limit_pairs) {
    run_directional_round<GraphCfg, DataT>(
      graph, globals, d_queries, n_queries, dim, k,
      bw, lim, has_gt ? &gt : nullptr, true);
  }

  cudaFree(d_queries);
}

// ── beam_width:limit parsing ───────────────────────────────────
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

// ── Construct + (optional) benchmark ───────────────────────────
template <typename GraphCfg, typename ConstructCfg, typename DataT>
void construct_and_save(const std::string& filename,
                        const std::string& query_path,
                        const std::string& gt_path,
                        const std::string& src_dtype,
                        uint32_t dim,
                        float alpha,
                        size_t workspace_budget_bytes,
                        uint32_t k,
                        const std::vector<std::pair<uint32_t, uint32_t>>& beam_limit_pairs,
                        bool on_host) {

  // Load + upload base vectors.
  jasper::vector_view<DataT> h_vecs;
  if (src_dtype == "float") {
    h_vecs = jasper::load_vectors_from_file_cast<DataT, float>(filename);
  } else if (src_dtype == "uint8") {
    h_vecs = jasper::load_vectors_from_file_cast<DataT, uint8_t>(filename);
  } else {
    throw std::runtime_error("Unsupported source dtype: " + src_dtype);
  }
  if (!h_vecs.data)
    throw std::runtime_error("Failed to load vectors from: " + filename);
  if (h_vecs.dim != dim)
    throw std::runtime_error(
      "File dim=" + std::to_string(h_vecs.dim) +
      " does not match --dim=" + std::to_string(dim));

  std::cout << "  Loaded " << h_vecs.n_vectors << " vectors, dim=" << dim << std::endl;

  // auto d_vecs = h_vecs.to_device();
  // cudaFreeHost(h_vecs.data);

  // Construction params.
  jasper::graph_construct_params<ConstructCfg> params;
  jasper::graph_construct_workspace<ConstructCfg> ws;
  uint32_t max_batch_size = min(
    ws.max_batch_size_for_budget(workspace_budget_bytes),
    h_vecs.n_vectors / 50
  );

  params.data_vectors   = h_vecs;
  params.alpha          = alpha;
  params.max_batch_size = max_batch_size;
  params.on_host        = false;
  params.prerotate      = true;

  std::cout << "  max_batch_size=" << max_batch_size << std::endl;
  std::cout << "  Constructing graph..." << std::endl;
  auto g = jasper::construct_graph<ConstructCfg>(params);

  std::cout << "  Populating graph lsh..." << std::endl;
  g.populate_edge_lsh();

  std::cout << "  Sampling for lsh global..." << std::endl;
  auto lsh_globals = g.generate_lsh_globals(/*n_samples=*/32768);

  g.move_to(on_host); // move to host

  // ── Benchmark step ───────────────────────────────────────────
  if (!query_path.empty()) {
    benchmark_all<GraphCfg, DataT>(
      g, lsh_globals, query_path, gt_path, src_dtype,
      dim, k, beam_limit_pairs);
  } else {
    std::cout << "\n  No --queries provided; skipping benchmark." << std::endl;
  }

  cudaFreeHost(h_vecs.data);
  g.deallocate();
}

// ── Dispatch ───────────────────────────────────────────────────
template <jasper::distance_func Func, uint32_t KR>
bool config_matches(const std::string& datatype, uint64_t n_neighbors,
                    const std::string& distance, uint64_t R, uint32_t k_ranks) {
  std::string expected_dist = (Func == jasper::distance_func::L2) ? "l2" : "ip";
  bool dtype_ok = (datatype == "float" || datatype == "uint8");
  return dtype_ok && n_neighbors == R && distance == expected_dist && k_ranks == KR;
}

void dispatch(const std::string& datatype,
              uint64_t n_neighbors,
              const std::string& distance,
              uint32_t k_ranks,
              const std::string& filename,
              const std::string& query_path,
              const std::string& gt_path,
              uint32_t dim,
              float alpha,
              size_t workspace_budget_bytes,
              uint32_t k,
              const std::vector<std::pair<uint32_t, uint32_t>>& beam_limit_pairs,
              bool on_host) {

  #define TRY_DISPATCH(id, IDX, R, DAT, DIST, FUNC, KR)                       \
    if (config_matches<FUNC, KR>(datatype, n_neighbors, distance, R, k_ranks)) { \
      std::cout << "  Config: " #id << std::endl;                             \
      construct_and_save<cfg_##id, construct_cfg_##id, DAT>(                  \
          filename, query_path, gt_path, datatype,                            \
          dim, alpha, workspace_budget_bytes, k, beam_limit_pairs, on_host);  \
      return;                                                                  \
    }

  JASPER_FOR_EACH_CONFIG(TRY_DISPATCH)
  #undef TRY_DISPATCH

  throw std::runtime_error(
    "Unsupported config: datatype=" + datatype +
    ", n_neighbors=" + std::to_string(n_neighbors) +
    ", distance=" + distance +
    ", k_ranks=" + std::to_string(k_ranks));
}

// ── Argument parsing ───────────────────────────────────────────
int main(int argc, char** argv) {
  argparse::ArgumentParser program("create_lsh_index");

  program.add_argument("--filename", "-f")
    .required()
    .help("Input vector filename.");

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
    .help("Number of neighbors per vector (32, 64, 128)");

  program.add_argument("--dim")
    .required()
    .scan<'u', uint32_t>()
    .help("Vector dimension");

  program.add_argument("--alpha", "-a")
    .default_value(1.2f)
    .scan<'g', float>()
    .help("Pruning factor alpha.");

  program.add_argument("--workspace_budget", "-w")
    .help("Device memory budget for construction (e.g. 5GB, 200MB)")
    .default_value(size_t{10ULL * 1024 * 1024 * 1024})
    .action([](const std::string& value) -> size_t {
      size_t pos = 0;
      double number = std::stod(value, &pos);
      std::string unit = value.substr(pos);
      for (auto& c : unit) c = std::toupper(c);
      if (unit == "B")  return static_cast<size_t>(number);
      if (unit == "KB") return static_cast<size_t>(number * 1024);
      if (unit == "MB") return static_cast<size_t>(number * 1024 * 1024);
      if (unit == "GB") return static_cast<size_t>(number * 1024 * 1024 * 1024);
      if (unit == "TB") return static_cast<size_t>(number * 1024 * 1024 * 1024 * 1024);
      throw std::runtime_error("Invalid size unit: \"" + unit +
                               "\" - expected B, KB, MB, GB, or TB");
    });

  // Benchmark args.
  program.add_argument("--queries", "-q")
    .default_value(std::string{})
    .help("Query vectors filename. If omitted, no benchmark runs.");

  program.add_argument("--groundtruth", "-g")
    .default_value(std::string{})
    .help("Ground truth .bin file (optional, used to print recall).");

  program.add_argument("--k", "-k")
    .default_value(uint32_t{10})
    .scan<'u', uint32_t>()
    .help("k for the benchmark search.");

  program.add_argument("--beam_limits", "-b")
    .default_value(std::vector<std::string>{
      "1:128", "2:128", "4:128", "8:128", "16:128",
      "32:128", "64:128", "128:256", "256:512", "512:1024", "1024:2048"
    })
    .nargs(argparse::nargs_pattern::at_least_one)
    .help("beam_width:limit pairs to benchmark (e.g. 16:128 64:256)");

  program.add_argument("--on_host")
        .default_value(false)
        .implicit_value(true)
        .help("query on host");

  program.add_argument("--k_ranks", "-r")
    .default_value(uint32_t{4})
    .scan<'u', uint32_t>()
    .choices(4u, 8u, 16u)
    .help("LSH k_ranks (4, 8, or 16)");

  try {
    program.parse_args(argc, argv);
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    return 1;
  }

  auto filename         = program.get<std::string>("--filename");
  auto datatype         = program.get<std::string>("--datatype");
  auto distance         = program.get<std::string>("--distance");
  auto n_neighbors      = program.get<uint64_t>("--n_neighbors");
  auto dim              = program.get<uint32_t>("--dim");
  auto alpha            = program.get<float>("--alpha");
  auto workspace_budget = program.get<size_t>("--workspace_budget");
  auto query_path       = program.get<std::string>("--queries");
  auto gt_path          = program.get<std::string>("--groundtruth");
  auto k                = program.get<uint32_t>("--k");
  auto beam_args        = program.get<std::vector<std::string>>("--beam_limits");
  auto on_host          = program.get<bool>("--on_host");
  auto k_ranks = program.get<uint32_t>("--k_ranks");

  std::vector<std::pair<uint32_t, uint32_t>> beam_limit_pairs;
  try {
    beam_limit_pairs = parse_beam_limit_pairs(beam_args);
  } catch (const std::exception& err) {
    std::cerr << "Error parsing --beam_limits: " << err.what() << std::endl;
    return 1;
  }

  std::cout << "=== create_lsh_index ===" << std::endl;
  std::cout << "  filename:         " << filename << std::endl;
  std::cout << "  datatype:         " << datatype << std::endl;
  std::cout << "  distance:         " << distance << std::endl;
  std::cout << "  n_neighbors:      " << n_neighbors << std::endl;
  std::cout << "  dim:              " << dim << std::endl;
  std::cout << "  alpha:            " << std::fixed << std::setprecision(2) << alpha << std::endl;
  std::cout << "  workspace_budget: "
            << (workspace_budget / (1024.0 * 1024.0 * 1024.0)) << " GB" << std::endl;
  std::cout << "  queries:          " << (query_path.empty() ? "(none)" : query_path) << std::endl;
  std::cout << "  groundtruth:      " << (gt_path.empty()    ? "(none)" : gt_path)    << std::endl;
  std::cout << "  k:                " << k << std::endl;
  std::cout << "  beam_limits:      ";
  for (auto& [bw, lim] : beam_limit_pairs) std::cout << bw << ":" << lim << " ";
  std::cout << std::endl;
  std::cout << "  on_host:          " << on_host << std::endl;
  std::cout << "  k_ranks:          " << k_ranks << std::endl;

  try {
    dispatch(datatype, n_neighbors, distance, k_ranks,
             filename, query_path, gt_path,
             dim, alpha, workspace_budget,
             k, beam_limit_pairs, on_host);
    std::cout << "  Done." << std::endl;
  } catch (const std::exception& err) {
    std::cerr << "Error: " << err.what() << std::endl;
    return 1;
  }

  return 0;
}