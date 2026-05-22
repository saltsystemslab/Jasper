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
#include <cuda_fp16.h>

#include "jasper/jasper.cuh"

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

// All configs use __half for storage. Source dtype (float / uint8) is
// applied at load time; it doesn't affect the chosen config.
template <jasper::distance_func Func>
bool config_matches(const std::string& datatype, uint64_t n_neighbors,
                    const std::string& distance, uint64_t R) {
  std::string expected_dist = (Func == jasper::distance_func::L2) ? "l2" : "ip";
  bool dtype_ok = (datatype == "float" || datatype == "uint8");
  return dtype_ok && n_neighbors == R && distance == expected_dist;
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

// ── Load graph + run all beam widths ───────────────────────────
template <typename GraphCfg, typename DataT>
void load_and_bench(const std::string& index_path,
                    const std::string& query_path,
                    const std::string& gt_path,
                    const std::string& src_dtype,
                    uint32_t dim,
                    uint32_t k,
                    const std::vector<std::pair<uint32_t, uint32_t>>& beam_limit_pairs) {

  // Load graph
  std::cout << "Loading index..." << std::endl;
  auto graph = jasper::load_graph_from_file<GraphCfg>(index_path, dim);
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

  try {
    bool dispatched = false;

    #define TRY_DISPATCH(id, IDX, R, DAT, DIST, FUNC)                         \
      if (!dispatched && config_matches<FUNC>(datatype, n_neighbors, distance, R)) { \
        std::cout << "  Config: " #id << std::endl;                           \
        load_and_bench<cfg_##id, DAT>(                                        \
            index_path, query_path, gt_path, datatype,                        \
            dim, k, beam_limit_pairs);                                        \
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
  } catch (const std::exception& err) {
    std::cerr << "Error: " << err.what() << std::endl;
    return 1;
  }

  return 0;
}