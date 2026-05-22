#include <argparse/argparse.hpp>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <cstdint>
#include <stdexcept>
#include <cuda_fp16.h>

#include "jasper/jasper.cuh"

// (CONFIG_ID, INDEX_T, N_NEIGHBORS, DATA_T, DISTANCE_T, DIST_FUNC)
#define JASPER_FOR_EACH_CONFIG(X)                                              \
  X(f16_r32_l2,   uint32_t, 32,  __half,   float, jasper::distance_func::L2)             \
  X(f16_r64_l2,   uint32_t, 64,  __half,   float, jasper::distance_func::L2)             \
  X(f16_r32_ip,   uint32_t, 32,  __half,   float, jasper::distance_func::INNER_PRODUCT)  \
  X(f16_r64_ip,   uint32_t, 64,  __half,   float, jasper::distance_func::INNER_PRODUCT)

// Graph + construct config types
#define DECLARE_CONFIGS(id, IDX, R, DAT, DIST, FUNC)                           \
  using cfg_##id = jasper::graph_config<IDX, R, DAT, DIST, FUNC>;             \
  using construct_cfg_##id = jasper::graph_construct_config<cfg_##id, 64, 4, R, 64>;

JASPER_FOR_EACH_CONFIG(DECLARE_CONFIGS)
#undef DECLARE_CONFIGS

template <typename GraphCfg, typename ConstructCfg, typename DataT>
void construct_and_save(const std::string& filename,
                        const std::string& src_dtype,
                        uint32_t dim,
                        float alpha,
                        size_t workspace_budget_bytes,
                        const std::string& index_out) {
  // Load vectors from file as the user-specified source dtype and cast to
  // DataT (__half) on host.
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

  // Upload padded buffer to device
  auto d_vecs = h_vecs.to_device();
  cudaFreeHost(h_vecs.data);

  // Set up construction params
  jasper::graph_construct_params<ConstructCfg> params;
  jasper::graph_construct_workspace<ConstructCfg> ws;
  uint32_t max_batch_size = min(
    ws.max_batch_size_for_budget(workspace_budget_bytes),
    d_vecs.n_vectors / 50
  );

  params.data_vectors   = d_vecs;
  params.alpha          = alpha;
  params.max_batch_size = max_batch_size;
  params.on_host        = false;

  std::cout << "  max_batch_size=" << max_batch_size << std::endl;
  std::cout << "  Constructing graph..." << std::endl;

  auto graph = jasper::construct_graph<ConstructCfg>(params);

  std::cout << "  Saving index to: " << index_out << std::endl;
  jasper::save_graph_to_file(graph, index_out);

  // Cleanup
  cudaFree(d_vecs.data);
  graph.deallocate();
}

// ── Dispatch: (datatype, n_neighbors, distance) → config ───────
// All configs use __half for storage. Source dtype (float / uint8) is
// applied at load time; it doesn't affect the chosen config.
template <jasper::distance_func Func>
bool config_matches(const std::string& datatype, uint64_t n_neighbors,
                    const std::string& distance, uint64_t R) {
  std::string expected_dist = (Func == jasper::distance_func::L2) ? "l2" : "ip";
  bool dtype_ok = (datatype == "float" || datatype == "uint8");
  return dtype_ok && n_neighbors == R && distance == expected_dist;
}

void dispatch(const std::string& datatype,
              uint64_t n_neighbors,
              const std::string& distance,
              const std::string& filename,
              uint32_t dim,
              float alpha,
              size_t workspace_budget_bytes,
              const std::string& index_out) {

  #define TRY_DISPATCH(id, IDX, R, DAT, DIST, FUNC)                           \
    if (config_matches<FUNC>(datatype, n_neighbors, distance, R)) {           \
      std::cout << "  Config: " #id << std::endl;                             \
      construct_and_save<cfg_##id, construct_cfg_##id, DAT>(                  \
          filename, datatype, dim, alpha, workspace_budget_bytes, index_out); \
      return;                                                                  \
    }

  JASPER_FOR_EACH_CONFIG(TRY_DISPATCH)
  #undef TRY_DISPATCH

  throw std::runtime_error(
    "Unsupported config: datatype=" + datatype +
    ", n_neighbors=" + std::to_string(n_neighbors) +
    ", distance=" + distance);
}

// ── Argument parsing ───────────────────────────────────────────
int main(int argc, char** argv) {
  argparse::ArgumentParser program("create_index");

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

  program.add_argument("--index_out", "-i")
    .required()
    .help("Output index filename.");

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
  auto index_out        = program.get<std::string>("--index_out");

  std::cout << "=== create_index ===" << std::endl;
  std::cout << "  filename:         " << filename << std::endl;
  std::cout << "  datatype:         " << datatype << std::endl;
  std::cout << "  distance:         " << distance << std::endl;
  std::cout << "  n_neighbors:      " << n_neighbors << std::endl;
  std::cout << "  dim:              " << dim << std::endl;
  std::cout << "  alpha:            " << std::fixed << std::setprecision(2) << alpha << std::endl;
  std::cout << "  workspace_budget: "
            << (workspace_budget / (1024.0 * 1024.0 * 1024.0)) << " GB" << std::endl;
  std::cout << "  index_out:        " << index_out << std::endl;

  try {
    dispatch(datatype, n_neighbors, distance,
             filename, dim, alpha, workspace_budget, index_out);
    std::cout << "  Done." << std::endl;
  } catch (const std::exception& err) {
    std::cerr << "Error: " << err.what() << std::endl;
    return 1;
  }

  return 0;
}