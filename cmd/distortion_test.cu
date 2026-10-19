// distortion_test — accuracy microbenchmarks for the directional edge
// estimators (cross-polytope LSH, PQ, RaBitQ), separate from run_query so the
// benchmark binary stays untouched.
//
// Loads a plain index, rotates it, builds ONE estimator's codes at load (same
// path as run_query without --prepopulated), then runs any of:
//
//   --pops <csv>        Search every query at each --beam_limits pair with
//                       visited logging on, and record recall@k against the
//                       number of nodes popped per query. Recall per POP takes
//                       the estimator's run-time cost out of the comparison:
//                       a method that needs fewer pops for the same recall
//                       ranks better, whatever its QPS.
//
//   --trajectory <csv>  Search at --traj_beam, then score the out-edges of
//                       (query, popped node) pairs drawn uniformly from the
//                       real trajectories — the pairs the search actually
//                       ranks, rather than a random node against a random
//                       query. `step` / `n_steps` locate each pair in its
//                       trajectory so late pops can be selected afterwards.
//
//   --random <csv>      Score the out-edges of random nodes against random
//                       queries (the original Fig. 8 sampling).
//
// -m conventional runs plain beam search (exact distance to every neighbor)
// for --pops only, as the baseline for vectors fetched per query: directional
// search fetches one vector per pop, conventional search one per appended
// neighbor. --pops reports both counts as mean_vectors_fetched.
//
// The edge-scoring CSVs share one schema (step = n_steps = -1 for --random):
//   method,k_ranks,distance,sample,u,query,step,n_steps,v,exact,est
// exact/est are squared L2. They feed scripts/plot_distortion.py and
// scripts/neighborhood_ranking.py unchanged.
//
// Only L2 configs are compiled, to keep build time down; the table mirrors
// run_query's L2 rows.
#include <argparse/argparse.hpp>
#include <thrust/pair.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <cublas_v2.h>
#include <cuda_fp16.h>

#include "jasper/jasper.cuh"
#include "jasper/beam_search/distortion.cuh"

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__            \
                << " - " << cudaGetErrorString(err) << std::endl;             \
      throw std::runtime_error(cudaGetErrorString(err));                       \
    }                                                                          \
  } while (0)

// (CONFIG_ID, INDEX_T, N_NEIGHBORS, DATA_T, DISTANCE_T, DIST_FUNC, K_RANKS, PACKED_T)
#define DT_FOR_EACH_CONFIG(X)                                                                        \
  X(r32_k4_d128,    uint32_t, 32, __half, float, jasper::distance_func::L2,  4, uint8_t)              \
  X(r32_k8_d128,    uint32_t, 32, __half, float, jasper::distance_func::L2,  8, uint8_t)              \
  X(r32_k16_d128,   uint32_t, 32, __half, float, jasper::distance_func::L2, 16, uint8_t)              \
  X(r64_k4_d128,    uint32_t, 64, __half, float, jasper::distance_func::L2,  4, uint8_t)              \
  X(r64_k8_d128,    uint32_t, 64, __half, float, jasper::distance_func::L2,  8, uint8_t)              \
  X(r64_k16_d128,   uint32_t, 64, __half, float, jasper::distance_func::L2, 16, uint8_t)              \
  X(r64_k4_d32678,  uint32_t, 64, __half, float, jasper::distance_func::L2,  4, uint16_t)             \
  X(r64_k8_d32678,  uint32_t, 64, __half, float, jasper::distance_func::L2,  8, uint16_t)             \
  X(r64_k16_d32678, uint32_t, 64, __half, float, jasper::distance_func::L2, 16, uint16_t)

#define DECLARE_CONFIGS_(id, IDX, R, DAT, DIST, FUNC, KR, PACKEDT)             \
  using cfg_##id = jasper::graph_config<IDX, R, DAT, DIST, FUNC, true, KR, PACKEDT>;
DT_FOR_EACH_CONFIG(DECLARE_CONFIGS_)
#undef DECLARE_CONFIGS_

// Search / sampling block size. Matches the default of directional_search,
// pq_search and rq_search, so the estimator instantiations are the same ones
// the searches run.
constexpr uint32_t kBlock = 128;

enum class Method { LSH, PQ, RQ, CONV };

static const char* method_name(Method m) {
  switch (m) {
    case Method::LSH: return "lsh";
    case Method::PQ:  return "pq";
    case Method::RQ:  return "rabitq";
    case Method::CONV: return "conventional";
  }
  return "?";
}

struct Opts {
  std::string index_path, query_path, gt_path, datatype;
  uint32_t    dim = 0, k = 10, pq_train = 40000;
  uint64_t    rotate_seed = 42;
  bool        do_rotate = true, on_host = false;
  Method      method = Method::LSH;

  std::string pops_out, traj_out, random_out;
  std::vector<std::pair<uint32_t, uint32_t>> beam_limits;
  std::pair<uint32_t, uint32_t> traj_beam{64, 256};
  uint32_t    samples = 10000;
  uint64_t    seed    = 7;
};

// ── Ground truth / recall (same format as run_query) ───────────────────────
struct GroundTruth {
  uint32_t n_queries = 0, k = 0;
  std::vector<uint32_t> ids;  // [n_queries * k]
};

static GroundTruth read_groundtruth(const std::string& path, uint32_t k) {
  std::ifstream fin(path, std::ios::binary);
  if (!fin) throw std::runtime_error("Cannot open ground truth file: " + path);
  uint32_t n, gk;
  fin.read(reinterpret_cast<char*>(&n), sizeof(n));
  fin.read(reinterpret_cast<char*>(&gk), sizeof(gk));
  if (gk < k)
    throw std::runtime_error("Requested k=" + std::to_string(k) +
                             " but ground truth only has k=" + std::to_string(gk));
  std::vector<uint32_t> all(static_cast<size_t>(n) * gk);
  fin.read(reinterpret_cast<char*>(all.data()), all.size() * sizeof(uint32_t));
  GroundTruth gt{n, k, std::vector<uint32_t>(static_cast<size_t>(n) * k)};
  for (uint32_t q = 0; q < n; ++q)
    for (uint32_t j = 0; j < k; ++j) gt.ids[q * k + j] = all[q * gk + j];
  return gt;
}

// ── Rotation (copied from run_query; see there for the reasoning) ─────────
static void rotate_device_vectors(__half* d_vectors, uint32_t n_vectors, uint32_t dim,
                                  uint32_t padded_dim, uint64_t seed,
                                  cublasHandle_t handle) {
  if (n_vectors == 0) return;
  std::vector<float> h_P_f(static_cast<size_t>(dim) * dim);
  std::vector<float> h_Pt_f(static_cast<size_t>(dim) * dim);
  jasper::set_rotation_matrix(dim, h_P_f.data(), h_Pt_f.data(), seed);
  std::vector<__half> h_P(h_P_f.size());
  for (size_t i = 0; i < h_P.size(); ++i) h_P[i] = static_cast<__half>(h_P_f[i]);

  __half* d_P = nullptr;
  CUDA_CHECK(cudaMalloc(&d_P, sizeof(__half) * dim * dim));
  CUDA_CHECK(cudaMemcpy(d_P, h_P.data(), sizeof(__half) * dim * dim, cudaMemcpyHostToDevice));
  const size_t bytes = sizeof(__half) * static_cast<size_t>(n_vectors) * padded_dim;
  __half* d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_out, bytes));
  CUDA_CHECK(cudaMemset(d_out, 0, bytes));
  const float one = 1.0f, zero = 0.0f;
  cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, dim, n_vectors, dim, &one,
               d_P, CUDA_R_16F, dim, d_vectors, CUDA_R_16F, padded_dim, &zero,
               d_out, CUDA_R_16F, padded_dim, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
  CUDA_CHECK(cudaMemcpy(d_vectors, d_out, bytes, cudaMemcpyDeviceToDevice));
  cudaFree(d_out);
  cudaFree(d_P);
}

template <typename GraphCfg>
static void rotate_graph_vectors(jasper::graph<GraphCfg>& g, uint64_t seed) {
  using segment_t = jasper::graph_segment<GraphCfg>;
  const uint32_t dim = g.dim, padded_dim = g.get_padded_dim();
  cublasHandle_t handle;
  cublasCreate(&handle);
  std::vector<segment_t> h_segs(g.segments.begin(), g.segments.end());
  for (auto& seg : h_segs) {
    const uint32_t n = static_cast<uint32_t>(seg.n_vectors);
    if (n == 0) continue;
    if (g.on_host) {
      const size_t bytes = sizeof(__half) * static_cast<size_t>(n) * padded_dim;
      __half* d = nullptr;
      CUDA_CHECK(cudaMalloc(&d, bytes));
      CUDA_CHECK(cudaMemcpy(d, seg.vectors.data, bytes, cudaMemcpyHostToDevice));
      rotate_device_vectors(d, n, dim, padded_dim, seed, handle);
      CUDA_CHECK(cudaMemcpy(seg.vectors.data, d, bytes, cudaMemcpyDeviceToHost));
      cudaFree(d);
    } else {
      rotate_device_vectors(seg.vectors.data, n, dim, padded_dim, seed, handle);
    }
  }
  cublasDestroy(handle);
}

// ── Search with visited logging, for whichever estimator was built ─────────
template <typename GraphCfg>
struct Codes {
  jasper::lsh_globals<GraphCfg::k_ranks>                     lsh{};
  jasper::pq_codebooks<GraphCfg::pq_m, GraphCfg::pq_k>       pq{};
  jasper::rq_globals<GraphCfg::rq_dim, GraphCfg::rq_bits>    rq{};
};

template <typename GraphCfg>
struct Trajectories {
  uint32_t n_queries = 0, stride = 0;
  std::vector<thrust::pair<uint32_t, float>> visited;  // [n_queries * stride], pop order
  std::vector<uint32_t> counts;                        // pops per query
  std::vector<uint64_t> fetched;                       // full vectors read per query
  std::vector<uint32_t> topk;                          // [n_queries * k]
};

// Vectors conventional search reads for one query, from its visited log.
// populate_distances computes an exact distance for every valid neighbor it
// appends (duplicates included, deduplication comes after), plus the medoid it
// seeds with, so the count is 1 + the out-degrees of the expanded pops. A query
// that hit max_result_size breaks right after recording its last pop, before
// expanding it, so that pop's neighbors are not counted. Exact for a single
// (unsharded-view) graph, where every neighbor is in range.
template <typename GraphCfg>
__global__ void conventional_vectors_fetched(
    typename jasper::graph<GraphCfg>::device_view g,
    const thrust::pair<uint32_t, float>* __restrict__ visited,
    const uint32_t* __restrict__ counts, uint32_t stride, uint32_t n_queries,
    uint64_t* __restrict__ out) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_queries) return;
  const uint32_t n        = counts[i];
  const uint32_t expanded = (n == stride) ? n - 1 : n;
  uint64_t s = 1;
  for (uint32_t j = 0; j < expanded; ++j)
    s += g.get_edge_count(visited[static_cast<size_t>(i) * stride + j].first);
  out[i] = s;
}

template <typename GraphCfg, Method M>
static Trajectories<GraphCfg> search_logged(jasper::graph<GraphCfg>& g, Codes<GraphCfg>& c,
                                            jasper::vector_view<__half> q, uint32_t k,
                                            uint32_t bw, uint32_t limit) {
  jasper::search_params p{
    .k = k, .beam_width = bw, .limit = limit,
    .get_visited = true,
    .max_result_size = limit + 1,  // the loop pops at most limit + 1 nodes
  };
  jasper::beam_search_result<GraphCfg> r{};
  if constexpr (M == Method::LSH) r = jasper::directional_search(g, c.lsh, q, p);
  if constexpr (M == Method::PQ)  r = jasper::pq_search(g, c.pq.view(), q, p);
  if constexpr (M == Method::RQ)  r = jasper::rq_search(g, c.rq, q, p);
  if constexpr (M == Method::CONV) r = jasper::search(g, q, p);
  CUDA_CHECK(cudaDeviceSynchronize());

  Trajectories<GraphCfg> t;
  t.n_queries = q.n_vectors;
  t.stride    = p.max_result_size;
  t.visited.resize(static_cast<size_t>(t.n_queries) * t.stride);
  t.counts.resize(t.n_queries);
  std::vector<thrust::pair<uint32_t, float>> front(static_cast<size_t>(t.n_queries) * k);
  CUDA_CHECK(cudaMemcpy(t.visited.data(), r.visited, sizeof(t.visited[0]) * t.visited.size(),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(t.counts.data(), r.visited_counts, sizeof(uint32_t) * t.n_queries,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(front.data(), r.frontier, sizeof(front[0]) * front.size(),
                        cudaMemcpyDeviceToHost));
  t.topk.resize(front.size());
  for (size_t i = 0; i < front.size(); ++i) t.topk[i] = front[i].first;

  // Directional search reads exactly one full vector per pop (visit() of u).
  t.fetched.assign(t.counts.begin(), t.counts.end());
  if constexpr (M == Method::CONV) {
    uint64_t* d_fetched = nullptr;
    CUDA_CHECK(cudaMalloc(&d_fetched, sizeof(uint64_t) * t.n_queries));
    conventional_vectors_fetched<GraphCfg><<<(t.n_queries + 255) / 256, 256>>>(
        g.view(), r.visited, r.visited_counts, t.stride, t.n_queries, d_fetched);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(t.fetched.data(), d_fetched, sizeof(uint64_t) * t.n_queries,
                          cudaMemcpyDeviceToHost));
    cudaFree(d_fetched);
  }
  cudaFree(r.frontier);
  cudaFree(r.visited);
  cudaFree(r.visited_counts);
  return t;
}

// ── Edge-scoring CSV ───────────────────────────────────────────────────────
template <typename GraphCfg, Method M>
static void score_and_write(jasper::graph<GraphCfg>& g, Codes<GraphCfg>& c,
                            jasper::vector_view<__half> q,
                            const std::vector<uint32_t>& nodes,
                            const std::vector<uint32_t>& qids,
                            const std::vector<int32_t>& steps,
                            const std::vector<int32_t>& n_steps,
                            const std::string& path) {
  using G = GraphCfg;
  jasper::distortion_samples<uint32_t> res;
  if constexpr (M == Method::LSH)
    res = jasper::estimator_distortion<G, jasper::lsh_estimator<G, G::dist_func, kBlock>, kBlock>(
        g, c.lsh, q, nodes, qids);
  if constexpr (M == Method::PQ)
    res = jasper::estimator_distortion<G, jasper::pq_estimator<G, G::dist_func, kBlock>, kBlock>(
        g, c.pq.view(), q, nodes, qids);
  if constexpr (M == Method::RQ)
    res = jasper::estimator_distortion<G, jasper::rq_estimator<G, G::dist_func, kBlock>, kBlock>(
        g, c.rq, q, nodes, qids);

  std::ofstream out(path);
  if (!out) throw std::runtime_error("Cannot open output: " + path);
  out << "method,k_ranks,distance,sample,u,query,step,n_steps,v,exact,est\n"
      << std::setprecision(9);
  constexpr uint32_t INVALID = 0x7FFFFFFFu;
  size_t rows = 0;
  for (size_t s = 0; s < nodes.size(); ++s) {
    for (uint32_t e = 0; e < res.n_neighbors; ++e) {
      const size_t i = s * res.n_neighbors + e;
      if (res.v[i] == INVALID) continue;
      out << method_name(M) << ',' << static_cast<uint32_t>(G::k_ranks) << ",l2," << s << ','
          << nodes[s] << ',' << qids[s] << ',' << steps[s] << ',' << n_steps[s] << ','
          << res.v[i] << ',' << res.exact[i] << ',' << res.est[i] << '\n';
      ++rows;
    }
  }
  std::cout << "  wrote " << rows << " edges from " << nodes.size()
            << " neighborhoods -> " << path << std::endl;
}

// ── The three experiments ──────────────────────────────────────────────────
template <typename GraphCfg, Method M>
static void run_pops(jasper::graph<GraphCfg>& g, Codes<GraphCfg>& c,
                     jasper::vector_view<__half> q, const GroundTruth& gt, const Opts& o) {
  std::ofstream out(o.pops_out);
  if (!out) throw std::runtime_error("Cannot open output: " + o.pops_out);
  out << "method,k_ranks,beam_width,limit,recall,mean_pops,p50_pops,p95_pops,capped_frac,"
         "mean_vectors_fetched\n";
  std::cout << "\n=== recall vs pops (" << method_name(M) << " k_ranks="
            << static_cast<uint32_t>(GraphCfg::k_ranks) << ") ===" << std::endl;

  for (auto [bw, lim] : o.beam_limits) {
    auto t = search_logged<GraphCfg, M>(g, c, q, o.k, bw, lim);

    uint64_t hits = 0;
    for (uint32_t i = 0; i < t.n_queries; ++i) {
      std::set<uint32_t> truth(gt.ids.begin() + size_t(i) * o.k,
                               gt.ids.begin() + size_t(i + 1) * o.k);
      for (uint32_t j = 0; j < o.k; ++j) hits += truth.count(t.topk[size_t(i) * o.k + j]);
    }
    const double recall = double(hits) / (double(t.n_queries) * o.k);

    std::vector<uint32_t> cnt = t.counts;
    std::sort(cnt.begin(), cnt.end());
    double mean = 0;
    for (uint32_t x : cnt) mean += x;
    mean /= cnt.size();
    // A query that used every pop it was allowed stopped on the limit, not on
    // convergence; worth knowing when reading the curve's tail.
    const double capped =
        double(std::count(cnt.begin(), cnt.end(), t.stride)) / cnt.size();
    double fetched = 0;
    for (uint64_t x : t.fetched) fetched += double(x);
    fetched /= t.fetched.size();

    out << method_name(M) << ',' << static_cast<uint32_t>(GraphCfg::k_ranks) << ','
        << bw << ',' << lim << ',' << std::setprecision(6) << recall << ',' << mean << ','
        << cnt[cnt.size() / 2] << ',' << cnt[cnt.size() * 95 / 100] << ',' << capped << ','
        << fetched << '\n';
    std::cout << "  beam_width=" << bw << " limit=" << lim << std::fixed
              << std::setprecision(4) << " recall@" << o.k << "=" << recall
              << std::setprecision(1) << " mean_pops=" << mean
              << " p95_pops=" << cnt[cnt.size() * 95 / 100]
              << std::setprecision(3) << " capped=" << capped
              << std::setprecision(1) << " vectors_fetched=" << fetched << std::endl;
    std::cout.unsetf(std::ios::fixed);
  }
  std::cout << "  wrote " << o.pops_out << std::endl;
}

template <typename GraphCfg, Method M>
static void run_trajectory(jasper::graph<GraphCfg>& g, Codes<GraphCfg>& c,
                           jasper::vector_view<__half> q, const Opts& o) {
  auto [bw, lim] = o.traj_beam;
  std::cout << "\n=== trajectory sampling (" << method_name(M) << " k_ranks="
            << static_cast<uint32_t>(GraphCfg::k_ranks) << ", beam_width=" << bw
            << " limit=" << lim << ", " << o.samples << " pops, seed " << o.seed
            << ") ===" << std::endl;
  auto t = search_logged<GraphCfg, M>(g, c, q, o.k, bw, lim);

  // Uniform over every (query, pop) the search made. Each method samples its
  // OWN trajectories — the pops it actually ranks — so pairs differ by method;
  // compare distributions, not rows.
  std::vector<uint64_t> prefix(t.n_queries + 1, 0);
  for (uint32_t i = 0; i < t.n_queries; ++i) prefix[i + 1] = prefix[i] + t.counts[i];
  if (prefix.back() == 0) throw std::runtime_error("searches popped nothing");

  std::mt19937_64 rng(o.seed);
  std::uniform_int_distribution<uint64_t> pick(0, prefix.back() - 1);
  std::vector<uint32_t> nodes(o.samples), qids(o.samples);
  std::vector<int32_t>  steps(o.samples), n_steps(o.samples);
  for (uint32_t s = 0; s < o.samples; ++s) {
    const uint64_t x = pick(rng);
    const uint32_t qi = static_cast<uint32_t>(
        std::upper_bound(prefix.begin(), prefix.end(), x) - prefix.begin() - 1);
    const uint32_t step = static_cast<uint32_t>(x - prefix[qi]);
    qids[s]    = qi;
    nodes[s]   = t.visited[size_t(qi) * t.stride + step].first;
    steps[s]   = static_cast<int32_t>(step);
    n_steps[s] = static_cast<int32_t>(t.counts[qi]);
  }
  score_and_write<GraphCfg, M>(g, c, q, nodes, qids, steps, n_steps, o.traj_out);
}

template <typename GraphCfg, Method M>
static void run_random(jasper::graph<GraphCfg>& g, Codes<GraphCfg>& c,
                       jasper::vector_view<__half> q, const Opts& o) {
  std::cout << "\n=== random sampling (" << method_name(M) << " k_ranks="
            << static_cast<uint32_t>(GraphCfg::k_ranks) << ", " << o.samples
            << " neighborhoods, seed " << o.seed << ") ===" << std::endl;
  // Depends only on (n_vectors, n_queries, seed): every method and k on the
  // same index scores the same (u, q) pairs.
  std::mt19937_64 rng(o.seed);
  std::uniform_int_distribution<uint64_t> pick_u(0, g.n_vectors - 1);
  std::uniform_int_distribution<uint32_t> pick_q(0, q.n_vectors - 1);
  std::vector<uint32_t> nodes(o.samples), qids(o.samples);
  for (uint32_t s = 0; s < o.samples; ++s) {
    nodes[s] = static_cast<uint32_t>(g.global_offset + pick_u(rng));
    qids[s]  = pick_q(rng);
  }
  std::vector<int32_t> none(o.samples, -1);
  score_and_write<GraphCfg, M>(g, c, q, nodes, qids, none, none, o.random_out);
}

// ── Load, rotate, build codes, run ─────────────────────────────────────────
template <typename GraphCfg, Method M>
static void run(const Opts& o) {
  if constexpr (M == Method::RQ) {
    if (!o.do_rotate)
      throw std::runtime_error("rabitq requires a rotated basis: drop --no_rotate");
  }

  std::cout << "Loading index..." << std::endl;
  auto graph = jasper::load_graph_from_file<GraphCfg>(o.index_path, o.dim, o.on_host);
  if (!o.on_host) jasper::rebuild_id_map<GraphCfg>(graph);
  std::cout << "  " << graph.n_vectors << " vectors, dim=" << graph.dim << std::endl;

  // An index written prerotated already stores rotated vectors: rotating them
  // again would put them in a different space from the queries. Use its seed
  // for the queries and leave the vectors alone.
  bool     rotate_vectors = o.do_rotate;
  bool     rotate_queries = o.do_rotate;
  uint64_t seed           = o.rotate_seed;
  const auto rot = jasper::read_rotation_info<GraphCfg>(o.index_path, o.dim);
  if (rot.known && rot.prerotate) {
    std::cout << "  index is prerotated (seed " << rot.seed
              << "); rotating queries only" << std::endl;
    rotate_vectors = false;
    rotate_queries = true;
    seed           = rot.seed;
  } else if constexpr (M == Method::CONV) {
    // Exact distances are rotation-invariant and there are no codes to build,
    // so skip the (host-side, at scale slow) rotation entirely.
    rotate_vectors = rotate_queries = false;
  }
  if (rotate_vectors) {
    std::cout << "Rotating stored vectors (seed=" << seed << ")..." << std::endl;
    rotate_graph_vectors<GraphCfg>(graph, seed);
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  Codes<GraphCfg> codes;
  auto t0 = std::chrono::steady_clock::now();
  if constexpr (M == Method::LSH) {
    std::cout << "Building LSH codes..." << std::endl;
    codes.lsh = graph.generate_lsh_globals(/*n_samples=*/32768);
    graph.populate_edge_lsh();
  } else if constexpr (M == Method::PQ) {
    std::cout << "Training PQ codebooks + encoding edges..." << std::endl;
    codes.pq = graph.generate_pq_codebooks(o.pq_train);
    graph.populate_edge_pq(codes.pq);
    graph.compute_vector_norms();
  } else if constexpr (M == Method::RQ) {
    std::cout << "Building RaBitQ codes (rq_dim=" << GraphCfg::rq_dim << ")..." << std::endl;
    codes.rq = graph.generate_rq_globals();
    graph.populate_edge_rq(codes.rq);
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  std::cout << "  built in "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::steady_clock::now() - t0).count() << " ms" << std::endl;

  // Queries, cast to __half and rotated into the codes' space.
  jasper::vector_view<__half> hq;
  if (o.datatype == "float")      hq = jasper::load_vectors_from_file_cast<__half, float>(o.query_path);
  else if (o.datatype == "uint8") hq = jasper::load_vectors_from_file_cast<__half, uint8_t>(o.query_path);
  else throw std::runtime_error("Unsupported datatype: " + o.datatype);
  if (!hq.data) throw std::runtime_error("Failed to load queries from: " + o.query_path);
  if (hq.dim != o.dim)
    throw std::runtime_error("Query dim " + std::to_string(hq.dim) +
                             " does not match --dim " + std::to_string(o.dim));
  auto dq = hq.to_device();
  cudaFreeHost(hq.data);
  std::cout << "  " << dq.n_vectors << " queries" << std::endl;
  if (rotate_queries) {
    cublasHandle_t h;
    cublasCreate(&h);
    rotate_device_vectors(dq.data, dq.n_vectors, o.dim, dq.padded_dim, seed, h);
    cublasDestroy(h);
  }

  if (!o.pops_out.empty()) {
    const auto gt = read_groundtruth(o.gt_path, o.k);
    if (gt.n_queries != dq.n_vectors)
      throw std::runtime_error("Ground truth query count mismatch");
    run_pops<GraphCfg, M>(graph, codes, dq, gt, o);
  }
  if (!o.traj_out.empty())   run_trajectory<GraphCfg, M>(graph, codes, dq, o);
  if (!o.random_out.empty()) run_random<GraphCfg, M>(graph, codes, dq, o);

  cudaFree(dq.data);
  if constexpr (M == Method::PQ) codes.pq.free();
  graph.deallocate();
}

static std::pair<uint32_t, uint32_t> parse_pair(const std::string& s) {
  const auto c = s.find(':');
  if (c == std::string::npos)
    throw std::runtime_error("Expected beam_width:limit, got: " + s);
  return {static_cast<uint32_t>(std::stoul(s.substr(0, c))),
          static_cast<uint32_t>(std::stoul(s.substr(c + 1)))};
}

int main(int argc, char** argv) {
  argparse::ArgumentParser program("distortion_test");
  program.add_argument("--index", "-i").required().help("Plain index (no codes needed).");
  program.add_argument("--queries", "-q").required().help("Query vectors.");
  program.add_argument("--groundtruth", "-g").default_value(std::string{})
      .help("Ground truth .bin (required for --pops).");
  program.add_argument("--datatype", "-t").required().choices("uint8", "float");
  program.add_argument("--n_neighbors", "-n").default_value(uint64_t{64}).scan<'u', uint64_t>();
  program.add_argument("--dim").required().scan<'u', uint32_t>();
  program.add_argument("--k", "-k").default_value(uint32_t{10}).scan<'u', uint32_t>();
  program.add_argument("--method", "-m").required()
      .choices("lsh", "pq", "rabitq", "conventional")
      .help("Edge estimator; conventional = exact-distance beam search (--pops only).");
  program.add_argument("--k_ranks", "-r").default_value(uint32_t{4}).scan<'u', uint32_t>()
      .choices(4u, 8u, 16u);
  program.add_argument("--pq_train").default_value(uint32_t{40000}).scan<'u', uint32_t>();
  program.add_argument("--on_host").default_value(false).implicit_value(true);
  program.add_argument("--no_rotate").default_value(false).implicit_value(true);
  program.add_argument("--rotate_seed").default_value(uint64_t{42}).scan<'u', uint64_t>();

  program.add_argument("--pops").default_value(std::string{})
      .help("CSV: recall@k vs nodes popped per query, one row per --beam_limits pair.");
  program.add_argument("--beam_limits", "-b")
      .default_value(std::vector<std::string>{"8:64", "16:128", "32:128", "64:256",
                                              "128:256", "256:512", "512:1024"})
      .nargs(argparse::nargs_pattern::at_least_one);
  program.add_argument("--trajectory").default_value(std::string{})
      .help("CSV: edge scores for (query, popped node) pairs from real searches.");
  program.add_argument("--traj_beam").default_value(std::string{"64:256"})
      .help("beam_width:limit of the searches --trajectory samples from.");
  program.add_argument("--random").default_value(std::string{})
      .help("CSV: edge scores for random nodes against random queries.");
  program.add_argument("--samples").default_value(uint32_t{10000}).scan<'u', uint32_t>()
      .help("Neighborhoods scored by --trajectory / --random.");
  program.add_argument("--seed").default_value(uint64_t{7}).scan<'u', uint64_t>();

  try {
    program.parse_args(argc, argv);
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl << program;
    return 1;
  }

  Opts o;
  o.index_path  = program.get<std::string>("--index");
  o.query_path  = program.get<std::string>("--queries");
  o.gt_path     = program.get<std::string>("--groundtruth");
  o.datatype    = program.get<std::string>("--datatype");
  o.dim         = program.get<uint32_t>("--dim");
  o.k           = program.get<uint32_t>("--k");
  o.pq_train    = program.get<uint32_t>("--pq_train");
  o.on_host     = program.get<bool>("--on_host");
  o.do_rotate   = !program.get<bool>("--no_rotate");
  o.rotate_seed = program.get<uint64_t>("--rotate_seed");
  o.pops_out    = program.get<std::string>("--pops");
  o.traj_out    = program.get<std::string>("--trajectory");
  o.random_out  = program.get<std::string>("--random");
  o.samples     = program.get<uint32_t>("--samples");
  o.seed        = program.get<uint64_t>("--seed");
  const auto method   = program.get<std::string>("--method");
  const auto n_nb     = program.get<uint64_t>("--n_neighbors");
  const auto k_ranks  = program.get<uint32_t>("--k_ranks");
  o.method = method == "lsh" ? Method::LSH : method == "pq" ? Method::PQ
           : method == "rabitq" ? Method::RQ : Method::CONV;

  try {
    for (auto& s : program.get<std::vector<std::string>>("--beam_limits"))
      o.beam_limits.push_back(parse_pair(s));
    o.traj_beam = parse_pair(program.get<std::string>("--traj_beam"));
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
  if (o.pops_out.empty() && o.traj_out.empty() && o.random_out.empty()) {
    std::cerr << "Error: nothing to do; pass --pops, --trajectory and/or --random" << std::endl;
    return 1;
  }
  if (o.method == Method::CONV && (!o.traj_out.empty() || !o.random_out.empty())) {
    std::cerr << "Error: -m conventional has no estimator to score; it supports --pops only"
              << std::endl;
    return 1;
  }
  if (!o.pops_out.empty() && o.gt_path.empty()) {
    std::cerr << "Error: --pops needs --groundtruth" << std::endl;
    return 1;
  }

  std::cout << "=== distortion_test: " << method << " k_ranks=" << k_ranks
            << " R=" << n_nb << " dim=" << o.dim << " ===" << std::endl;

  try {
    bool dispatched = false;
    #define TRY_DISPATCH(id, IDX, R, DAT, DIST, FUNC, KR, PACKEDT)                  \
      if (!dispatched && n_nb == R && k_ranks == KR &&                              \
          ((sizeof(PACKEDT) == 1) ? (o.dim <= 128) : (o.dim > 128))) {             \
        std::cout << "  Config: " #id << std::endl;                                 \
        switch (o.method) {                                                         \
          case Method::LSH: run<cfg_##id, Method::LSH>(o); break;                   \
          case Method::PQ:  run<cfg_##id, Method::PQ>(o);  break;                   \
          case Method::RQ:  run<cfg_##id, Method::RQ>(o);  break;                   \
          case Method::CONV: run<cfg_##id, Method::CONV>(o); break;                 \
        }                                                                           \
        dispatched = true;                                                          \
      }
    DT_FOR_EACH_CONFIG(TRY_DISPATCH)
    #undef TRY_DISPATCH
    if (!dispatched)
      throw std::runtime_error("Unsupported config: n_neighbors=" + std::to_string(n_nb) +
                               ", k_ranks=" + std::to_string(k_ranks) +
                               ", dim=" + std::to_string(o.dim) +
                               " (L2 only; dim > 128 needs R=64)");
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
  return 0;
}
