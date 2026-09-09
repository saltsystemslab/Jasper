#include <argparse/argparse.hpp>
#include <iostream>
#include <iomanip>
#include <string>
#include <cstdint>
#include <stdexcept>
#include <chrono>
#include <cuda_fp16.h>

#include "jasper/jasper.cuh"

// DiskANN-style merge build: shard with polytope LSH (duplicating each vector
// into n_dup shards), build one intermediate graph per shard at degree R_int,
// then merge into a final graph at degree R_final.
//
// Configs are keyed by (R_int, R_final, DISTANCE). All use __half storage.
// (CONFIG_ID, INDEX_T, R_INT, R_FINAL, DATA_T, DISTANCE_T, DIST_FUNC)
#define JASPER_FOR_EACH_CONFIG(X)                                                        \
  X(f16_r32_32_l2,  uint32_t, 32,  32, __half, float, jasper::distance_func::L2)          \
  X(f16_r32_32_ip,  uint32_t, 32,  32, __half, float, jasper::distance_func::INNER_PRODUCT) \
  X(f16_r32_64_l2,  uint32_t, 32,  64, __half, float, jasper::distance_func::L2)          \
  X(f16_r32_64_ip,  uint32_t, 32,  64, __half, float, jasper::distance_func::INNER_PRODUCT)

#define DECLARE_CONFIGS(id, IDX, RINT, RFIN, DAT, DIST, FUNC)                            \
  using int_cfg_##id = jasper::graph_construct_config<                                    \
      jasper::graph_config<IDX, RINT, DAT, DIST, FUNC>, 64, 4, RINT, 64>;                 \
  using fin_cfg_##id = jasper::graph_construct_config<                                    \
      jasper::graph_config<IDX, RFIN, DAT, DIST, FUNC>, 64, 4, RFIN, 64>;

JASPER_FOR_EACH_CONFIG(DECLARE_CONFIGS)
#undef DECLARE_CONFIGS

// ── Directional configs (--lsh / --pq) ─────────────────────────────────────
// Same (R_int, R_final, DISTANCE) keys, plus use_lsh=true, K_RANKS and the
// packed word width. PACKED_T is picked by dim, exactly as create_lsh_index and
// run_query do it: uint8_t carries a 7-bit coordinate index (dim <= 128),
// uint16_t a 15-bit one (dim > 128).
//
// Only k_ranks = 4 is compiled in. Every entry here instantiates the whole
// construct + populate + merge stack twice (INT and FIN), in one translation
// unit; adding a k_ranks dimension doubles that. If you need another value, add
// the rows and expect the build to get slower — or split this file per config
// the way ffi/ does.
//
// Note k_ranks doubles as the PQ subquantizer count M (graph_config::pq_m), and
// --pq requires padded_dim % M == 0.
// (CONFIG_ID, INDEX_T, R_INT, R_FINAL, DATA_T, DISTANCE_T, DIST_FUNC, K_RANKS, PACKED_T)
#define JASPER_FOR_EACH_DIR_CONFIG(X)                                                    \
  X(d_r32_32_l2_k4_d128,   uint32_t, 32, 32, __half, float, jasper::distance_func::L2,            4, uint8_t)  \
  X(d_r32_64_l2_k4_d128,   uint32_t, 32, 64, __half, float, jasper::distance_func::L2,            4, uint8_t)  \
  X(d_r32_32_l2_k4_dbig,   uint32_t, 32, 32, __half, float, jasper::distance_func::L2,            4, uint16_t) \
  X(d_r32_64_l2_k4_dbig,   uint32_t, 32, 64, __half, float, jasper::distance_func::L2,            4, uint16_t) \
  X(d_r32_32_ip_k4_d128,   uint32_t, 32, 32, __half, float, jasper::distance_func::INNER_PRODUCT, 4, uint8_t)  \
  X(d_r32_64_ip_k4_d128,   uint32_t, 32, 64, __half, float, jasper::distance_func::INNER_PRODUCT, 4, uint8_t)  \
  X(d_r32_32_ip_k4_dbig,   uint32_t, 32, 32, __half, float, jasper::distance_func::INNER_PRODUCT, 4, uint16_t) \
  X(d_r32_64_ip_k4_dbig,   uint32_t, 32, 64, __half, float, jasper::distance_func::INNER_PRODUCT, 4, uint16_t)

#define DECLARE_DIR_CONFIGS(id, IDX, RINT, RFIN, DAT, DIST, FUNC, KR, PACKEDT)           \
  using int_cfg_##id = jasper::graph_construct_config<                                    \
      jasper::graph_config<IDX, RINT, DAT, DIST, FUNC, true, KR, PACKEDT>, 64, 4, RINT, 64>; \
  using fin_cfg_##id = jasper::graph_construct_config<                                    \
      jasper::graph_config<IDX, RFIN, DAT, DIST, FUNC, true, KR, PACKEDT>, 64, 4, RFIN, 64>;

JASPER_FOR_EACH_DIR_CONFIG(DECLARE_DIR_CONFIGS)
#undef DECLARE_DIR_CONFIGS

void print_available_memory() {
  size_t free_byte, total_byte;
  cudaMemGetInfo(&free_byte, &total_byte);
  double free_db  = (double)free_byte  / (1024 * 1024);
  double total_db = (double)total_byte / (1024 * 1024);
  printf("GPU memory usage: used = %f MB, free = %f MB, total = %f MB\n",
         total_db - free_db, free_db, total_db);
}

// Parse human-readable sizes: "5GB", "200MB", etc.
size_t parse_size(const std::string& value) {
  size_t pos = 0;
  double number = std::stod(value, &pos);
  std::string unit = value.substr(pos);
  for (auto& c : unit) c = std::toupper(c);
  if (unit == "B")  return static_cast<size_t>(number);
  if (unit == "KB") return static_cast<size_t>(number * 1024);
  if (unit == "MB") return static_cast<size_t>(number * 1024 * 1024);
  if (unit == "GB") return static_cast<size_t>(number * 1024ULL * 1024 * 1024);
  if (unit == "TB") return static_cast<size_t>(number * 1024ULL * 1024 * 1024 * 1024);
  throw std::runtime_error(
    "Invalid size unit: \"" + unit + "\" - expected B, KB, MB, GB, or TB");
}

template <typename IntCfg, typename FinCfg, typename DataT>
void construct(
    const std::string& filename,
    const std::string& src_dtype,
    float              alpha,
    uint32_t           n_parts,
    uint32_t           n_dup,
    size_t             workspace_budget,
    size_t             n_vector_limit,
    const std::string& index_out,
    jasper::diskann_directional_opts opts = jasper::diskann_directional_opts{})
{
  jasper::vector_view<DataT> vecs;
  if (src_dtype == "float") {
    vecs = jasper::load_vectors_from_file_cast<DataT, float>(filename);
  } else if (src_dtype == "uint8") {
    vecs = jasper::load_vectors_from_file_cast<DataT, uint8_t>(filename);
  } else {
    throw std::runtime_error("Unsupported source dtype: " + src_dtype);
  }
  if (!vecs.data) throw std::runtime_error("Failed to load vectors from: " + filename);

  if (n_vector_limit > 0 && n_vector_limit < vecs.n_vectors) {
    vecs.n_vectors = static_cast<uint32_t>(n_vector_limit);
  }

  std::cout << "  Loaded " << vecs.n_vectors << " vectors, dim=" << vecs.dim << std::endl;

  jasper::graph_construct_params<IntCfg> params;
  params.data_vectors   = vecs;
  params.alpha          = alpha;
  params.max_batch_size = std::max<uint32_t>(1, vecs.n_vectors / n_parts / 20);
  params.on_host        = false;

  auto t0 = std::chrono::steady_clock::now();
  auto ig = jasper::diskann_intermediate_graph<IntCfg, FinCfg>::build(
      params, n_parts, n_dup, workspace_budget, /*seed=*/42, opts);
  auto t1 = std::chrono::steady_clock::now();
  std::cout << "  build (shard + distance-only merge): " << std::fixed << std::setprecision(3)
            << std::chrono::duration<double>(t1 - t0).count() << " s" << std::endl;
  print_available_memory();

  std::cout << "  Average degree: " << ig.avg_degree() << "\n";

  t0 = std::chrono::steady_clock::now();
  std::cout << "  Saving index to: " << index_out << std::endl;
  // build() prerotated `vecs` in place when asked, so either saver writes the
  // vectors that the codes and the graph were built from.
  if (opts.any()) {
    ig.save_directional_to_file(vecs, index_out);
  } else {
    ig.save_to_file(vecs, index_out);
  }
  t1 = std::chrono::steady_clock::now();
  std::cout << "  save: " << std::fixed << std::setprecision(3)
            << std::chrono::duration<double>(t1 - t0).count() << " s" << std::endl;

  if (opts.rotating()) {
    std::cout << "\n  This index is in a ROTATED space (seed "
              << opts.prerotate_seed << "), recorded in the file — readers "
                 "rotate queries to match on their own:\n"
              << "      run_query ... --lsh --prepopulated\n";
  }

  ig.dir.free_models();   // releases the PQ codebooks' device memory
  cudaFreeHost(vecs.data);
}

template <jasper::distance_func Func>
bool config_matches(const std::string& datatype, const std::string& distance,
                    uint64_t r_int, uint64_t r_final, uint64_t RINT, uint64_t RFIN) {
  std::string expected_dist = (Func == jasper::distance_func::L2) ? "l2" : "ip";
  bool dtype_ok = (datatype == "float" || datatype == "uint8");
  return dtype_ok && distance == expected_dist && r_int == RINT && r_final == RFIN;
}

// Directional variant: also keys on k_ranks and on the packed word width, which
// is selected by dim (uint8_t <= 128 < uint16_t) — same rule as run_query.
template <jasper::distance_func Func, uint32_t KR, typename PackedT>
bool dir_config_matches(const std::string& datatype, const std::string& distance,
                        uint64_t r_int, uint64_t r_final, uint64_t RINT, uint64_t RFIN,
                        uint32_t k_ranks, uint32_t dim) {
  std::string expected_dist = (Func == jasper::distance_func::L2) ? "l2" : "ip";
  bool dtype_ok = (datatype == "float" || datatype == "uint8");
  bool dim_ok   = (sizeof(PackedT) == 1) ? (dim <= 128) : (dim > 128);
  return dtype_ok && distance == expected_dist && r_int == RINT && r_final == RFIN
      && k_ranks == KR && dim_ok;
}

void dispatch(
    const std::string& datatype,
    const std::string& distance,
    uint64_t           r_int,
    uint64_t           r_final,
    const std::string& filename,
    float              alpha,
    uint32_t           n_parts,
    uint32_t           n_dup,
    size_t             workspace_budget,
    size_t             n_vector_limit,
    const std::string& index_out,
    const jasper::diskann_directional_opts& opts,
    uint32_t           k_ranks,
    uint32_t           dim)
{
  if (opts.any()) {
    if (dim == 0)
      throw std::runtime_error("--lsh/--pq require --dim (it selects the packed "
                               "word width: uint8 for dim <= 128, uint16 above)");

    #define TRY_DIR_DISPATCH(id, IDX, RINT, RFIN, DAT, DIST, FUNC, KR, PACKEDT)          \
      if (dir_config_matches<FUNC, KR, PACKEDT>(datatype, distance, r_int, r_final,      \
                                                RINT, RFIN, k_ranks, dim)) {             \
        std::cout << "  Config: " #id << std::endl;                                      \
        construct<int_cfg_##id, fin_cfg_##id, DAT>(                                      \
            filename, datatype, alpha, n_parts, n_dup, workspace_budget,                 \
            n_vector_limit, index_out, opts);                                             \
        return;                                                                           \
      }

    JASPER_FOR_EACH_DIR_CONFIG(TRY_DIR_DISPATCH)
    #undef TRY_DIR_DISPATCH

    throw std::runtime_error(
      "Unsupported directional config: distance=" + distance +
      ", r_intermediate=" + std::to_string(r_int) +
      ", r_final=" + std::to_string(r_final) +
      ", k_ranks=" + std::to_string(k_ranks) +
      ", dim=" + std::to_string(dim) +
      " (compiled in: (r_int,r_final) in {(32,32),(32,64)}, k_ranks=4, l2|ip). "
      "Add a row to JASPER_FOR_EACH_DIR_CONFIG for anything else.");
  }

  #define TRY_DISPATCH(id, IDX, RINT, RFIN, DAT, DIST, FUNC)                             \
    if (config_matches<FUNC>(datatype, distance, r_int, r_final, RINT, RFIN)) {          \
      std::cout << "  Config: " #id << std::endl;                                         \
      construct<int_cfg_##id, fin_cfg_##id, DAT>(                                          \
          filename, datatype, alpha, n_parts, n_dup, workspace_budget,                    \
          n_vector_limit, index_out);                                                     \
      return;                                                                             \
    }

  JASPER_FOR_EACH_CONFIG(TRY_DISPATCH)
  #undef TRY_DISPATCH

  throw std::runtime_error(
    "Unsupported config: datatype=" + datatype + ", distance=" + distance +
    ", r_intermediate=" + std::to_string(r_int) +
    ", r_final=" + std::to_string(r_final) +
    " (supported (r_int,r_final): (32,32), (32,64))");
}

int main(int argc, char** argv) {
  argparse::ArgumentParser program("create_diskann_index");

  program.add_argument("--filename", "-f")
    .required().help("Input vector filename.");

  program.add_argument("--datatype", "-t")
    .required().choices("uint8", "float")
    .help("Vector datatype [\"uint8\", \"float\"] (cast to __half internally)");

  program.add_argument("--distance", "-d")
    .default_value(std::string{"l2"}).choices("l2", "ip")
    .help("Distance type [\"l2\" (default), \"ip\"]");

  program.add_argument("--r_intermediate")
    .default_value(uint64_t{32}).scan<'u', uint64_t>()
    .help("Intermediate (per-shard) graph degree R_int (32).");

  program.add_argument("--r_final", "-n")
    .default_value(uint64_t{32}).scan<'u', uint64_t>()
    .help("Final merged graph degree R_final (32 or 64).");

  program.add_argument("--n_parts")
    .required().scan<'u', uint32_t>()
    .help("Number of LSH shards to split the dataset into.");

  program.add_argument("--n_dup")
    .default_value(uint32_t{2}).scan<'u', uint32_t>()
    .help("Number of duplications: shards each vector is placed into (default 2).");

  program.add_argument("--alpha", "-a")
    .default_value(1.2f).scan<'g', float>()
    .help("Pruning factor alpha.");

  program.add_argument("--workspace_budget", "-w")
    .required()
    .help("Device memory budget for the per-shard construction workspace (e.g. 5GB).")
    .action(parse_size);

  program.add_argument("--n_vector")
    .default_value(size_t{0}).scan<'u', size_t>()
    .help("Limit the number of vectors read from file (0 = no limit).");

  program.add_argument("--index_out", "-i")
    .required().help("Output index filename.");

  // ── Directional (LSH/PQ) population ──
  program.add_argument("--dim")
    .default_value(uint32_t{0}).scan<'u', uint32_t>()
    .help("Vector dimensionality. Only needed with --lsh/--pq, where it selects "
          "the packed word width (uint8 for dim <= 128, uint16 above).");

  program.add_argument("--lsh")
    .default_value(false).implicit_value(true)
    .help("Populate per-edge cross-polytope LSH codes during the build and "
          "write them in the index trailer (enables directional_search).");

  program.add_argument("--pq")
    .default_value(false).implicit_value(true)
    .help("Populate per-edge PQ codes + exact vector norms and write them in "
          "the index trailer (enables pq_search). Can be combined with --lsh.");

  program.add_argument("--k_ranks")
    .default_value(uint32_t{4}).scan<'u', uint32_t>()
    .help("LSH rank count, which is also the PQ subquantizer count M. Only 4 is "
          "compiled in; --pq additionally needs padded_dim divisible by it.");

  program.add_argument("--lsh_samples")
    .default_value(uint32_t{32768}).scan<'u', uint32_t>()
    .help("Edges sampled from the pilot shard to calibrate lsh_globals.");

  program.add_argument("--pq_train")
    .default_value(uint32_t{40000}).scan<'u', uint32_t>()
    .help("Edge residuals sampled from the pilot shard to train PQ codebooks.");

  program.add_argument("--no_prerotate")
    .default_value(false).implicit_value(true)
    .help("Skip the global prerotation. Default is to rotate the dataset once "
          "before sharding, which the LSH estimator's calibration relies on; "
          "queries must then use the same --prerotate_seed.");

  program.add_argument("--prerotate_seed")
    .default_value(uint64_t{42}).scan<'u', uint64_t>()
    .help("Seed for the prerotation matrix. NOT stored in the index — pass the "
          "same value to run_query's --rotate_seed.");

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
  auto r_int            = program.get<uint64_t>("--r_intermediate");
  auto r_final          = program.get<uint64_t>("--r_final");
  auto n_parts          = program.get<uint32_t>("--n_parts");
  auto n_dup            = program.get<uint32_t>("--n_dup");
  auto alpha            = program.get<float>("--alpha");
  auto workspace_budget = program.get<size_t>("--workspace_budget");
  auto n_vector_limit   = program.get<size_t>("--n_vector");
  auto index_out        = program.get<std::string>("--index_out");
  auto dim              = program.get<uint32_t>("--dim");
  auto k_ranks          = program.get<uint32_t>("--k_ranks");

  jasper::diskann_directional_opts opts;
  opts.build_lsh      = program.get<bool>("--lsh");
  opts.build_pq       = program.get<bool>("--pq");
  opts.lsh_samples    = program.get<uint32_t>("--lsh_samples");
  opts.pq_train       = program.get<uint32_t>("--pq_train");
  opts.prerotate      = !program.get<bool>("--no_prerotate");
  opts.prerotate_seed = program.get<uint64_t>("--prerotate_seed");

  std::cout << "=== create_diskann_index ===" << std::endl;
  std::cout << "  filename:         " << filename << std::endl;
  std::cout << "  datatype:         " << datatype << std::endl;
  std::cout << "  distance:         " << distance << std::endl;
  std::cout << "  r_intermediate:   " << r_int << std::endl;
  std::cout << "  r_final:          " << r_final << std::endl;
  std::cout << "  n_parts:          " << n_parts << std::endl;
  std::cout << "  n_dup:            " << n_dup << std::endl;
  std::cout << "  alpha:            " << std::fixed << std::setprecision(2) << alpha << std::endl;
  std::cout << "  workspace_budget: "
            << (workspace_budget / (1024.0 * 1024.0 * 1024.0)) << " GB" << std::endl;
  if (n_vector_limit > 0)
    std::cout << "  n_vector:         " << n_vector_limit << " (limit)" << std::endl;
  std::cout << "  index_out:        " << index_out << std::endl;
  if (opts.any()) {
    std::cout << "  lsh:              " << (opts.build_lsh ? "true" : "false") << std::endl;
    std::cout << "  pq:               " << (opts.build_pq ? "true" : "false") << std::endl;
    std::cout << "  k_ranks:          " << k_ranks << std::endl;
    std::cout << "  dim:              " << dim << std::endl;
    std::cout << "  prerotate:        " << (opts.prerotate ? "true" : "false");
    if (opts.prerotate) std::cout << " (seed " << opts.prerotate_seed << ")";
    std::cout << std::endl;
  }

  try {
    dispatch(datatype, distance, r_int, r_final,
             filename, alpha, n_parts, n_dup, workspace_budget,
             n_vector_limit, index_out, opts, k_ranks, dim);
    std::cout << "  Done." << std::endl;
  } catch (const std::exception& err) {
    std::cerr << "Error: " << err.what() << std::endl;
    return 1;
  }

  return 0;
}
