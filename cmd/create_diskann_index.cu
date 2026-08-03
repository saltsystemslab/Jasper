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
    const std::string& index_out)
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
      params, n_parts, n_dup, workspace_budget);
  auto t1 = std::chrono::steady_clock::now();
  std::cout << "  build (shard + distance-only merge): " << std::fixed << std::setprecision(3)
            << std::chrono::duration<double>(t1 - t0).count() << " s" << std::endl;
  print_available_memory();

  std::cout << "  Average degree: " << ig.avg_degree() << "\n";

  t0 = std::chrono::steady_clock::now();
  std::cout << "  Saving index to: " << index_out << std::endl;
  ig.save_to_file(vecs, index_out);
  t1 = std::chrono::steady_clock::now();
  std::cout << "  save: " << std::fixed << std::setprecision(3)
            << std::chrono::duration<double>(t1 - t0).count() << " s" << std::endl;

  cudaFreeHost(vecs.data);
}

template <jasper::distance_func Func>
bool config_matches(const std::string& datatype, const std::string& distance,
                    uint64_t r_int, uint64_t r_final, uint64_t RINT, uint64_t RFIN) {
  std::string expected_dist = (Func == jasper::distance_func::L2) ? "l2" : "ip";
  bool dtype_ok = (datatype == "float" || datatype == "uint8");
  return dtype_ok && distance == expected_dist && r_int == RINT && r_final == RFIN;
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
    const std::string& index_out)
{
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

  try {
    dispatch(datatype, distance, r_int, r_final,
             filename, alpha, n_parts, n_dup, workspace_budget,
             n_vector_limit, index_out);
    std::cout << "  Done." << std::endl;
  } catch (const std::exception& err) {
    std::cerr << "Error: " << err.what() << std::endl;
    return 1;
  }

  return 0;
}
