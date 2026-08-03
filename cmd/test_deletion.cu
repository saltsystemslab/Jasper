// test_deletion — load a prebuilt index, exercise the deletion procedure
// (mark_deleted → consolidate → compact) against real queries / ground truth,
// and assert the key correctness invariant: deleted ids never appear in
// search results. Modeled on run_query.cu.
//
// Example (bigann100M, R=64, dim=128, uint8/L2):
//   ./test_deletion -i $INDEX_DIR/bigann100M -q $DATA_DIR/bigann/bigann10kquery \
//       -g $GT_DIR/GT_100M/bigann-100M -t uint8 -d l2 -n 64 --dim 128 \
//       -k 10 -b 64 --delete_fraction 0.2

#include <argparse/argparse.hpp>
#include <thrust/pair.h>
#include <cuda_fp16.h>

#include <iostream>
#include <iomanip>
#include <fstream>
#include <random>
#include <set>
#include <unordered_set>
#include <stdexcept>
#include <string>
#include <vector>

#include "jasper/jasper.cuh"

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__             \
                << " - " << cudaGetErrorString(err) << std::endl;             \
      throw std::runtime_error(cudaGetErrorString(err));                       \
    }                                                                          \
  } while (0)

// ── Config table (must match the index that was built) ─────────────────────
#define JASPER_FOR_EACH_CONFIG(X)                                              \
  X(f16_r32_l2,   uint32_t, 32,  __half,   float, jasper::distance_func::L2)             \
  X(f16_r64_l2,   uint32_t, 64,  __half,   float, jasper::distance_func::L2)             \
  X(f16_r32_ip,   uint32_t, 32,  __half,   float, jasper::distance_func::INNER_PRODUCT)  \
  X(f16_r64_ip,   uint32_t, 64,  __half,   float, jasper::distance_func::INNER_PRODUCT)

#define DECLARE_CFG(id, IDX, R, DAT, DIST, FUNC)                               \
  using cfg_##id = jasper::graph_config<IDX, R, DAT, DIST, FUNC>;             \
  using construct_cfg_##id = jasper::graph_construct_config<cfg_##id, 64, 4, R, 64>;
JASPER_FOR_EACH_CONFIG(DECLARE_CFG)
#undef DECLARE_CFG

template <jasper::distance_func Func>
bool config_matches(const std::string& datatype, uint64_t n_neighbors,
                    const std::string& distance, uint64_t R) {
  std::string expected = (Func == jasper::distance_func::L2) ? "l2" : "ip";
  bool dtype_ok = (datatype == "float" || datatype == "uint8");
  return dtype_ok && n_neighbors == R && distance == expected;
}

__global__ void unpack_results_kernel(
    const thrust::pair<uint32_t, float>* __restrict__ pairs,
    int32_t* __restrict__ out_indices, uint32_t total) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < total) out_indices[i] = static_cast<int32_t>(pairs[i].first);
}

// ── Ground truth ────────────────────────────────────────────────────────────
struct GroundTruth {
  uint32_t n_queries = 0, k = 0;
  std::vector<int32_t> indices;  // [n_queries * k]
};

GroundTruth read_groundtruth(const std::string& path, uint32_t k) {
  std::ifstream fin(path, std::ios::binary);
  if (!fin) throw std::runtime_error("Cannot open ground truth: " + path);
  uint32_t n_queries, gt_k;
  fin.read(reinterpret_cast<char*>(&n_queries), sizeof(n_queries));
  fin.read(reinterpret_cast<char*>(&gt_k), sizeof(gt_k));
  if (gt_k < k) throw std::runtime_error("gt k too small");
  std::vector<uint32_t> all_ids((size_t)n_queries * gt_k);
  fin.read(reinterpret_cast<char*>(all_ids.data()),
           (size_t)n_queries * gt_k * sizeof(uint32_t));
  GroundTruth gt;
  gt.n_queries = n_queries;
  gt.k = k;
  gt.indices.resize((size_t)n_queries * k);
  for (uint32_t q = 0; q < n_queries; q++)
    for (uint32_t j = 0; j < k; j++)
      gt.indices[q * k + j] = static_cast<int32_t>(all_ids[q * gt_k + j]);
  return gt;
}

// Recall@k counting only ground-truth entries that are still live (not in
// `deleted`). With no deletions this is ordinary recall@k.
float live_recall(const GroundTruth& gt, const std::vector<int32_t>& res,
                  uint32_t k, uint32_t n_queries,
                  const std::unordered_set<uint32_t>& deleted) {
  uint64_t hits = 0, total = 0;
  for (uint32_t q = 0; q < n_queries; q++) {
    std::set<int32_t> got(res.data() + q * k, res.data() + q * k + k);
    for (uint32_t j = 0; j < k; j++) {
      int32_t gt_id = gt.indices[q * k + j];
      if (deleted.count(static_cast<uint32_t>(gt_id))) continue;  // expected gone
      total++;
      if (got.count(gt_id)) hits++;
    }
  }
  return total ? static_cast<float>(hits) / static_cast<float>(total) : 0.0f;
}

// ── Run one search, return host result indices [n_queries * k] ──────────────
template <typename GraphCfg, typename DataT>
std::vector<int32_t> run_search(const jasper::graph<GraphCfg>& g,
                                DataT* d_queries, uint32_t n_queries,
                                uint32_t k, uint32_t beam_width) {
  jasper::vector_view<DataT> qv(d_queries, g.dim, n_queries, false);
  jasper::search_params params{.k = k, .beam_width = beam_width,
                               .limit = beam_width * 2, .get_visited = false};
  auto result = jasper::search(g, qv, params);

  uint32_t total = n_queries * k;
  int32_t* d_idx = nullptr;
  CUDA_CHECK(cudaMalloc(&d_idx, total * sizeof(int32_t)));
  uint32_t threads = 256, blocks = (total + threads - 1) / threads;
  // Translate internal slots -> stable ids (matches the FFI search path).
  jasper::translate_slots_to_ids_kernel<GraphCfg><<<blocks, threads>>>(
      g.view(), result.frontier, total);
  unpack_results_kernel<<<blocks, threads>>>(result.frontier, d_idx, total);
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<int32_t> h_idx(total);
  CUDA_CHECK(cudaMemcpy(h_idx.data(), d_idx, total * sizeof(int32_t),
                        cudaMemcpyDeviceToHost));
  cudaFree(d_idx);
  cudaFree(result.frontier);
  return h_idx;
}

// Count result ids that are in the deleted set (must be 0 after a delete).
uint64_t count_violations(const std::vector<int32_t>& res,
                          const std::unordered_set<uint32_t>& deleted) {
  uint64_t v = 0;
  for (int32_t id : res)
    if (id >= 0 && deleted.count(static_cast<uint32_t>(id))) v++;
  return v;
}

// ── Main test flow for one config ───────────────────────────────────────────
// Construct a graph from raw base vectors (mirrors cmd/create_index.cu).
template <typename GraphCfg, typename ConstructCfg, typename DataT>
jasper::graph<GraphCfg> build_index(const std::string& base_path,
                                    const std::string& src_dtype, uint32_t dim,
                                    float alpha, size_t workspace_budget) {
  jasper::vector_view<DataT> h_base;
  if (src_dtype == "float")
    h_base = jasper::load_vectors_from_file_cast<DataT, float>(base_path);
  else
    h_base = jasper::load_vectors_from_file_cast<DataT, uint8_t>(base_path);
  if (!h_base.data) throw std::runtime_error("Failed to load base vectors");
  if (h_base.dim != dim) throw std::runtime_error("base dim mismatch");
  std::cout << "  Constructing index from " << h_base.n_vectors
            << " base vectors (dim=" << dim << ") ..." << std::endl;

  auto d_base = h_base.to_device();
  cudaFreeHost(h_base.data);

  jasper::graph_construct_params<ConstructCfg> params;
  jasper::graph_construct_workspace<ConstructCfg> ws;
  uint32_t max_batch_size = min(
      ws.max_batch_size_for_budget(workspace_budget), d_base.n_vectors / 50);
  params.data_vectors   = d_base;
  params.alpha          = alpha;
  params.max_batch_size = max_batch_size;
  params.on_host        = false;
  std::cout << "  max_batch_size=" << max_batch_size << std::endl;

  auto g = jasper::construct_graph<ConstructCfg>(params);
  cudaFree(d_base.data);
  return g;
}

template <typename GraphCfg, typename ConstructCfg, typename DataT>
int run_test(const std::string& index_path, const std::string& base_path,
             const std::string& query_path,
             const std::string& gt_path, const std::string& src_dtype,
             uint32_t dim, uint32_t k, uint32_t beam_width,
             double delete_fraction, uint32_t rounds, uint64_t seed,
             bool do_compact, size_t workspace_budget) {
  using index_t = typename GraphCfg::index_t;

  jasper::graph<GraphCfg> g{};
  if (!base_path.empty()) {
    g = build_index<GraphCfg, ConstructCfg, DataT>(
        base_path, src_dtype, dim, 1.2f, workspace_budget);
  } else {
    std::cout << "Loading index from " << index_path << " ..." << std::endl;
    g = jasper::load_graph_from_file<GraphCfg>(index_path, dim);
    // load_graph_from_file doesn't build the forward stable_id -> slot table
    // (see graph.cuh) — rebuild it here so mark_deleted() can resolve ids.
    jasper::rebuild_id_map<GraphCfg>(g);
  }
  std::cout << "  " << g.n_vectors << " vectors, dim=" << g.dim
            << ", medoid=" << g.medoid << std::endl;

  std::cout << "Loading queries..." << std::endl;
  jasper::vector_view<DataT> h_q;
  if (src_dtype == "float")
    h_q = jasper::load_vectors_from_file_cast<DataT, float>(query_path);
  else
    h_q = jasper::load_vectors_from_file_cast<DataT, uint8_t>(query_path);
  if (!h_q.data) throw std::runtime_error("Failed to load queries");
  if (h_q.dim != dim) throw std::runtime_error("query dim mismatch");
  uint32_t n_queries = h_q.n_vectors;
  auto d_qv = h_q.to_device();
  cudaFreeHost(h_q.data);
  DataT* d_queries = d_qv.data;
  std::cout << "  " << n_queries << " queries" << std::endl;

  bool has_gt = !gt_path.empty();
  GroundTruth gt;
  if (has_gt) {
    gt = read_groundtruth(gt_path, k);
    if (gt.n_queries != n_queries)
      throw std::runtime_error("gt query count mismatch");
  }

  std::unordered_set<uint32_t> empty_set;
  int failures = 0;

  // ---- Phase 1: baseline ----
  {
    auto res = run_search<GraphCfg, DataT>(g, d_queries, n_queries, k, beam_width);
    std::cout << "\n[baseline] n_live=" << (uint64_t)(g.n_vectors - g.n_deleted);
    if (has_gt)
      std::cout << " recall@" << k << "="
                << live_recall(gt, res, k, n_queries, empty_set);
    std::cout << std::endl;
  }

  // ---- Deletion rounds: each round marks delete_fraction of the ORIGINAL
  //      vector count as fresh ids, verifies search excludes them, then
  //      consolidates. `removed` accumulates across rounds. ----
  const uint64_t original_n = g.n_vectors;
  const uint64_t per_round  = (uint64_t)(delete_fraction * (double)original_n);
  std::mt19937_64 rng(seed);
  std::unordered_set<uint32_t> removed;
  removed.reserve(per_round * rounds * 2);

  for (uint32_t r = 1; r <= rounds; r++) {
    // Sample `per_round` fresh ids not already removed.
    std::unordered_set<uint32_t> this_round;
    std::uniform_int_distribution<uint32_t> dist(0, (uint32_t)original_n - 1);
    while (this_round.size() < per_round) {
      uint32_t id = dist(rng);
      if (!removed.count(id)) this_round.insert(id);
    }
    std::vector<index_t> batch(this_round.begin(), this_round.end());
    jasper::mark_deleted<ConstructCfg>(g, batch.data(), (index_t)batch.size());
    for (uint32_t id : this_round) removed.insert(id);

    if (g.n_deleted != batch.size()) {
      std::cerr << "FAIL: round " << r << " n_tombstoned=" << (uint64_t)g.n_deleted
                << " != marked " << batch.size() << std::endl;
      failures++;
    }

    // soft-deleted search: no removed id may appear.
    {
      auto res = run_search<GraphCfg, DataT>(g, d_queries, n_queries, k, beam_width);
      uint64_t v = count_violations(res, removed);
      std::cout << "\n[round " << r << "/" << rounds << " soft-deleted] removed_total="
                << removed.size() << " n_tombstoned=" << (uint64_t)g.n_deleted
                << " violations=" << v;
      if (has_gt)
        std::cout << " live_recall@" << k << "="
                  << live_recall(gt, res, k, n_queries, removed);
      std::cout << std::endl;
      if (v != 0) { std::cerr << "FAIL: deleted ids in round " << r << " soft results\n"; failures++; }
    }

    // consolidate, then re-verify.
    jasper::consolidate<ConstructCfg>(g, 1.2f);
    {
      if (g.n_deleted != 0) {
        std::cerr << "FAIL: round " << r << " n_tombstoned=" << (uint64_t)g.n_deleted
                  << " after consolidate\n";
        failures++;
      }
      auto res = run_search<GraphCfg, DataT>(g, d_queries, n_queries, k, beam_width);
      uint64_t v = count_violations(res, removed);
      std::cout << "[round " << r << "/" << rounds << " consolidated] removed_total="
                << removed.size() << " violations=" << v;
      if (has_gt)
        std::cout << " live_recall@" << k << "="
                  << live_recall(gt, res, k, n_queries, removed);
      std::cout << std::endl;
      if (v != 0) { std::cerr << "FAIL: deleted ids in round " << r << " consolidated results\n"; failures++; }
    }
  }

  std::cout << "\n[summary] removed " << removed.size() << " / " << original_n
            << " (" << (100.0 * removed.size() / original_n) << "%) over "
            << rounds << " rounds" << std::endl;

  // ---- Phase 4: compact. Stable ids survive, so recall vs GT stays valid and
  //      deleted ids must still never reappear. (Disabled unless built with
  //      JASPER_ENABLE_COMPACT.) ----
  if (do_compact && !JASPER_ENABLE_COMPACT) {
    std::cout << "[compact] disabled in this build (mark + consolidate only); skipping\n";
  }
#if JASPER_ENABLE_COMPACT
  if (do_compact) {
    uint64_t expected_live = (uint64_t)g.n_vectors;  // n_deleted already 0
    jasper::compact<ConstructCfg>(g);
    std::cout << "[compacted] n_vectors=" << (uint64_t)g.n_vectors << std::endl;
    if ((uint64_t)g.n_vectors > expected_live) {
      std::cerr << "FAIL: compact grew n_vectors\n";
      failures++;
    }
    auto res = run_search<GraphCfg, DataT>(g, d_queries, n_queries, k, beam_width);
    uint64_t v = count_violations(res, removed);
    std::cout << "[compacted] violations=" << v;
    if (has_gt)
      std::cout << " live_recall@" << k << "="
                << live_recall(gt, res, k, n_queries, removed);
    std::cout << std::endl;
    if (v != 0) {
      std::cerr << "FAIL: deleted stable id reappeared after compact\n";
      failures++;
    }
  }
#endif  // JASPER_ENABLE_COMPACT

  cudaFree(d_queries);
  g.deallocate();

  std::cout << "\n===== " << (failures == 0 ? "PASS" : "FAIL")
            << " (" << failures << " failures) =====" << std::endl;
  return failures == 0 ? 0 : 1;
}

int main(int argc, char** argv) {
  argparse::ArgumentParser program("test_deletion");
  program.add_argument("--index", "-i").default_value(std::string{})
    .help("Prebuilt index filename (omit if using --base)");
  program.add_argument("--base", "-B").default_value(std::string{})
    .help("Raw base vectors to construct the index from (instead of --index)");
  program.add_argument("--workspace_budget", "-w").default_value(uint64_t{10ULL<<30})
    .scan<'u', uint64_t>().help("Construction workspace budget in bytes");
  program.add_argument("--queries", "-q").required().help("Query vectors");
  program.add_argument("--groundtruth", "-g").default_value(std::string{});
  program.add_argument("--datatype", "-t").required().choices("uint8", "float");
  program.add_argument("--distance", "-d").default_value(std::string{"l2"}).choices("l2", "ip");
  program.add_argument("--n_neighbors", "-n").default_value(uint64_t{64}).scan<'u', uint64_t>();
  program.add_argument("--dim").required().scan<'u', uint32_t>();
  program.add_argument("--k", "-k").default_value(uint32_t{10}).scan<'u', uint32_t>();
  program.add_argument("--beam_width", "-b").default_value(uint32_t{64}).scan<'u', uint32_t>();
  program.add_argument("--delete_fraction", "-f").default_value(0.2).scan<'g', double>();
  program.add_argument("--rounds", "-r").default_value(uint32_t{1}).scan<'u', uint32_t>()
    .help("Number of delete+consolidate rounds (each deletes delete_fraction of original)");
  program.add_argument("--seed").default_value(uint64_t{42}).scan<'u', uint64_t>();
  program.add_argument("--compact").default_value(false).implicit_value(true)
    .help("Also run compact() after consolidate()");

  try {
    program.parse_args(argc, argv);
  } catch (const std::exception& err) {
    std::cerr << err.what() << std::endl << program;
    return 1;
  }

  auto index_path  = program.get<std::string>("--index");
  auto base_path   = program.get<std::string>("--base");
  auto workspace   = program.get<uint64_t>("--workspace_budget");
  auto query_path  = program.get<std::string>("--queries");
  auto gt_path     = program.get<std::string>("--groundtruth");
  auto datatype    = program.get<std::string>("--datatype");
  auto distance    = program.get<std::string>("--distance");
  auto n_neighbors = program.get<uint64_t>("--n_neighbors");
  auto dim         = program.get<uint32_t>("--dim");
  auto k           = program.get<uint32_t>("--k");
  auto beam_width  = program.get<uint32_t>("--beam_width");
  auto del_frac    = program.get<double>("--delete_fraction");
  auto rounds      = program.get<uint32_t>("--rounds");
  auto seed        = program.get<uint64_t>("--seed");
  auto do_compact  = program.get<bool>("--compact");

  if (index_path.empty() == base_path.empty()) {
    std::cerr << "Error: provide exactly one of --index or --base\n";
    return 1;
  }

  std::cout << "=== test_deletion ===\n"
            << "  " << (base_path.empty() ? ("index=" + index_path)
                                          : ("base=" + base_path)) << "\n"
            << "  datatype=" << datatype << " distance=" << distance
            << " R=" << n_neighbors << " dim=" << dim << "\n"
            << "  k=" << k << " beam_width=" << beam_width
            << " delete_fraction=" << del_frac << " rounds=" << rounds
            << " compact=" << (do_compact ? "yes" : "no") << std::endl;

  try {
    bool dispatched = false;
    int rc = 0;
    #define TRY_DISPATCH(id, IDX, R, DAT, DIST, FUNC)                          \
      if (!dispatched && config_matches<FUNC>(datatype, n_neighbors, distance, R)) { \
        std::cout << "  Config: " #id << std::endl;                           \
        rc = run_test<cfg_##id, construct_cfg_##id, DAT>(                      \
            index_path, base_path, query_path, gt_path, datatype, dim, k,     \
            beam_width, del_frac, rounds, seed, do_compact, (size_t)workspace);\
        dispatched = true;                                                     \
      }
    JASPER_FOR_EACH_CONFIG(TRY_DISPATCH)
    #undef TRY_DISPATCH
    if (!dispatched)
      throw std::runtime_error("Unsupported config");
    return rc;
  } catch (const std::exception& err) {
    std::cerr << "Error: " << err.what() << std::endl;
    return 1;
  }
}
