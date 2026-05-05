#include <argparse/argparse.hpp>
#include <iostream>
#include <iomanip>
#include <string>
#include <cstdint>
#include <stdexcept>
#include <chrono>

#include "jasper/jasper.cuh"

// (CONFIG_ID, INDEX_T, N_NEIGHBORS, DATA_T, DISTANCE_T, DIST_FUNC)
#define JASPER_FOR_EACH_CONFIG(X)                                              \
  X(f32_r32_l2,  uint32_t, 32,  float,   float, jasper::distance_func::L2)             \
  X(f32_r64_l2,  uint32_t, 64,  float,   float, jasper::distance_func::L2)             \
  X(f32_r128_l2, uint32_t, 128, float,   float, jasper::distance_func::L2)             \
  X(f32_r32_ip,  uint32_t, 32,  float,   float, jasper::distance_func::INNER_PRODUCT)  \
  X(f32_r64_ip,  uint32_t, 64,  float,   float, jasper::distance_func::INNER_PRODUCT)  \
  X(u8_r32_l2,   uint32_t, 32,  uint8_t, float, jasper::distance_func::L2)             \
  X(u8_r64_l2,   uint32_t, 64,  uint8_t, float, jasper::distance_func::L2)

#define DECLARE_CONFIGS(id, IDX, R, DAT, DIST, FUNC)                          \
  using cfg_##id          = jasper::graph_config<IDX, R, DAT, DIST, FUNC>;   \
  using construct_cfg_##id = jasper::graph_construct_config<cfg_##id, 64, 4, R, 64>;

JASPER_FOR_EACH_CONFIG(DECLARE_CONFIGS)
#undef DECLARE_CONFIGS

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

template <typename GraphCfg, typename ConstructCfg, typename DataT>
void construct(
    const std::string& filename,
    float              alpha,
    size_t             n_parts,
    size_t             workspace_budget,
    size_t             n_vector_limit)
{
  // Load vectors into pinned host memory (padded rows)
  auto vecs = jasper::load_vectors_from_file<DataT>(filename);
  if (!vecs.data) throw std::runtime_error("Failed to load vectors from: " + filename);

  if (n_vector_limit > 0 && n_vector_limit < vecs.n_vectors) {
    vecs.n_vectors = static_cast<uint32_t>(n_vector_limit);
  }

  std::cout << "  Loaded " << vecs.n_vectors
            << " vectors, dim=" << vecs.dim << std::endl;

  jasper::graph_construct_params<ConstructCfg> params;
  params.data_vectors   = vecs;
  params.alpha          = alpha;
  params.max_batch_size = static_cast<uint32_t>(vecs.n_vectors / n_parts / 20);
  params.on_host        = false;

  auto t0 = std::chrono::steady_clock::now();
  auto ig = jasper::intermediate_graph<GraphCfg, ConstructCfg>::template
      allocate_and_construct(params, n_parts, workspace_budget);
  auto t1 = std::chrono::steady_clock::now();
  double elapsed_s = std::chrono::duration<double>(t1 - t0).count();
  std::cout << "  allocate_and_construct: " << std::fixed << std::setprecision(3)
            << elapsed_s << " s" << std::endl;

  t0 = std::chrono::steady_clock::now();
  ig.merge();
  t1 = std::chrono::steady_clock::now();
  elapsed_s = std::chrono::duration<double>(t1 - t0).count();
  std::cout << "  merge: " << std::fixed << std::setprecision(3)
            << elapsed_s << " s" << std::endl;

  t0 = std::chrono::steady_clock::now();
  auto g = ig.concat();
  t1 = std::chrono::steady_clock::now();
  elapsed_s = std::chrono::duration<double>(t1 - t0).count();
  std::cout << "  concat: " << std::fixed << std::setprecision(3)
            << elapsed_s << " s" << std::endl;

  cudaFreeHost(vecs.data);
  for (auto& g : ig.partitions) g.deallocate();
}

template <typename DataT, jasper::distance_func Func>
bool config_matches(const std::string& datatype, uint64_t n_neighbors,
                    const std::string& distance, uint64_t R) {
  std::string expected_dtype = std::is_same_v<DataT, float> ? "float" : "uint8";
  std::string expected_dist  = (Func == jasper::distance_func::L2) ? "l2" : "ip";
  return datatype == expected_dtype && n_neighbors == R && distance == expected_dist;
}

void dispatch(
    const std::string& datatype,
    uint64_t           n_neighbors,
    const std::string& distance,
    const std::string& filename,
    float              alpha,
    size_t             n_parts,
    size_t             workspace_budget,
    size_t             n_vector_limit)
{
  #define TRY_DISPATCH(id, IDX, R, DAT, DIST, FUNC)                           \
    if (config_matches<DAT, FUNC>(datatype, n_neighbors, distance, R)) {      \
      std::cout << "  Config: " #id << std::endl;                             \
      construct<cfg_##id, construct_cfg_##id, DAT>(                  \
          filename, alpha, n_parts, workspace_budget, n_vector_limit);  \
      return;                                                                  \
    }

  JASPER_FOR_EACH_CONFIG(TRY_DISPATCH)
  #undef TRY_DISPATCH

  throw std::runtime_error(
    "Unsupported config: datatype=" + datatype +
    ", n_neighbors=" + std::to_string(n_neighbors) +
    ", distance=" + distance);
}

int main(int argc, char** argv) {
  argparse::ArgumentParser program("create_scale_index");

  program.add_argument("--filename", "-f")
    .required()
    .help("Input vector filename.");

  program.add_argument("--datatype", "-t")
    .required()
    .choices("uint8", "float")
    .help("Vector datatype [\"uint8\", \"float\"]");

  program.add_argument("--distance", "-d")
    .default_value(std::string{"l2"})
    .choices("l2", "ip")
    .help("Distance type [\"l2\" (default), \"ip\"]");

  program.add_argument("--n_neighbors", "-n")
    .default_value(uint64_t{64})
    .scan<'u', uint64_t>()
    .help("Number of neighbors per vector (32, 64, 128)");

  program.add_argument("--alpha", "-a")
    .default_value(1.2f)
    .scan<'g', float>()
    .help("Pruning factor alpha.");

  program.add_argument("--n_parts")
    .required()
    .scan<'u', size_t>()
    .help("Number of partitions to split the dataset into.");

  program.add_argument("--workspace_budget", "-w")
    .required()
    .help("Device memory budget for the construction workspace (e.g. 5GB, 200MB).")
    .action(parse_size);

  program.add_argument("--n_vector")
    .default_value(size_t{0})
    .scan<'u', size_t>()
    .help("Limit the number of vectors read from file (0 = no limit).");

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
  auto alpha            = program.get<float>("--alpha");
  auto n_parts          = program.get<size_t>("--n_parts");
  auto workspace_budget = program.get<size_t>("--workspace_budget");
  auto n_vector_limit   = program.get<size_t>("--n_vector");

  std::cout << "=== create_scale_index ===" << std::endl;
  std::cout << "  filename:         " << filename << std::endl;
  std::cout << "  datatype:         " << datatype << std::endl;
  std::cout << "  distance:         " << distance << std::endl;
  std::cout << "  n_neighbors:      " << n_neighbors << std::endl;
  std::cout << "  alpha:            " << std::fixed << std::setprecision(2) << alpha << std::endl;
  std::cout << "  n_parts:          " << n_parts << std::endl;
  std::cout << "  workspace_budget: "
            << (workspace_budget / (1024.0 * 1024.0 * 1024.0)) << " GB" << std::endl;
  if (n_vector_limit > 0)
    std::cout << "  n_vector:         " << n_vector_limit << " (limit)" << std::endl;

  try {
    dispatch(datatype, n_neighbors, distance,
             filename, alpha, n_parts, workspace_budget, n_vector_limit);
    std::cout << "  Done." << std::endl;
  } catch (const std::exception& err) {
    std::cerr << "Error: " << err.what() << std::endl;
    return 1;
  }

  return 0;
}
