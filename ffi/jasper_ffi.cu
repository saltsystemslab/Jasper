#include <tvm/ffi/tvm_ffi.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <thrust/pair.h>

#include "jasper/jasper.cuh"

#include <unordered_map>
#include <mutex>
#include <variant>
#include <string>

namespace jasper_ffi {

namespace ffi = tvm::ffi;

// ── Enumerate all configs ──────────────────────────────────────
// (CONFIG_ID, INDEX_T, N_NEIGHBORS, DATA_T, DISTANCE_T, DIST_FUNC)

#define JASPER_FOR_EACH_CONFIG(X)                                              \
  X(f32_r32_l2,   uint32_t, 32, float, float, jasper::distance_func::L2)      \
  X(f32_r64_l2,   uint32_t, 64, float, float, jasper::distance_func::L2)      \
  X(f32_r128_l2,  uint32_t, 128, float, float, jasper::distance_func::L2)     \
  X(f32_r32_ip,   uint32_t, 32, float, float, jasper::distance_func::INNER_PRODUCT) \
  X(f32_r64_ip,   uint32_t, 64, float, float, jasper::distance_func::INNER_PRODUCT) \
  X(u8_r32_l2,    uint32_t, 32, uint8_t, float, jasper::distance_func::L2)    \
  X(u8_r64_l2,    uint32_t, 64, uint8_t, float, jasper::distance_func::L2)

// ── Generate graph config types ────────────────────────────────
#define DECLARE_CONFIG(id, IDX, R, DAT, DIST, FUNC) \
  using cfg_##id = jasper::graph_config<IDX, R, DAT, DIST, FUNC>;

JASPER_FOR_EACH_CONFIG(DECLARE_CONFIG)
#undef DECLARE_CONFIG

// ── Generate construct config types ────────────────────────────
// (block_size=128, tile_size=4, R from graph config, L=128)
#define DECLARE_CONSTRUCT_CONFIG(id, IDX, R, DAT, DIST, FUNC) \
  using construct_cfg_##id = jasper::graph_construct_config<cfg_##id, 128, 4, R, 128>;

JASPER_FOR_EACH_CONFIG(DECLARE_CONSTRUCT_CONFIG)
#undef DECLARE_CONSTRUCT_CONFIG

// ── Graph storage using variant ────────────────────────────────
#define VARIANT_ENTRY(id, ...) jasper::graph<cfg_##id>,
using GraphVariant = std::variant<
  JASPER_FOR_EACH_CONFIG(VARIANT_ENTRY)
  std::monostate
>;
#undef VARIANT_ENTRY

static std::unordered_map<int64_t, GraphVariant> g_graphs;
static int64_t g_next_handle = 0;
static std::mutex g_mutex;

// ── Unpack kernel (shared across all configs) ──────────────────
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

// ── Per-config load/search/construct implementations ───────────

#define DEFINE_OPS(id, IDX, R, DAT, DIST, FUNC)                               \
                                                                               \
int64_t LoadGraph_##id(ffi::String path, int64_t dim) {                        \
  auto g = jasper::load_graph_from_file<cfg_##id>(                             \
      std::string(path), static_cast<uint32_t>(dim));                          \
  std::lock_guard<std::mutex> lock(g_mutex);                                   \
  int64_t handle = g_next_handle++;                                            \
  g_graphs[handle] = std::move(g);                                             \
  return handle;                                                               \
}                                                                              \
                                                                               \
int64_t ConstructGraph_##id(ffi::TensorView vectors,                           \
                            int64_t dim,                                        \
                            double alpha,                                       \
                            int64_t max_batch_size) {                           \
  uint32_t n_vectors = static_cast<uint32_t>(vectors.size(0));                 \
  uint32_t d = static_cast<uint32_t>(dim);                                     \
                                                                               \
  jasper::vector_view<DAT> vecs(                                               \
      static_cast<DAT*>(vectors.data_ptr()), d, n_vectors);                    \
                                                                               \
  jasper::graph_construct_params<construct_cfg_##id> params;                    \
  params.data_vectors   = vecs;                                                \
  params.alpha          = static_cast<float>(alpha);                           \
  params.max_batch_size = static_cast<uint32_t>(max_batch_size);               \
  params.on_host        = false;                                               \
                                                                               \
  auto g = jasper::construct_graph<construct_cfg_##id>(params);                \
                                                                               \
  std::lock_guard<std::mutex> lock(g_mutex);                                   \
  int64_t handle = g_next_handle++;                                            \
  g_graphs[handle] = std::move(g);                                             \
  return handle;                                                               \
}                                                                              \
                                                                               \
void Search_##id(int64_t handle,                                               \
                 ffi::TensorView queries,                                      \
                 ffi::TensorView out_indices,                                  \
                 ffi::TensorView out_distances,                                \
                 int64_t k, int64_t beam_width, int64_t limit) {               \
  jasper::graph<cfg_##id>* gp;                                                 \
  {                                                                            \
    std::lock_guard<std::mutex> lock(g_mutex);                                 \
    auto it = g_graphs.find(handle);                                           \
    TVM_FFI_ICHECK(it != g_graphs.end()) << "Invalid handle: " << handle;      \
    gp = &std::get<jasper::graph<cfg_##id>>(it->second);                       \
  }                                                                            \
  auto& g = *gp;                                                               \
                                                                               \
  uint32_t n_queries = static_cast<uint32_t>(queries.size(0));                 \
  uint32_t dim_ = g.vectors.dim;                                               \
                                                                               \
  jasper::vector_view<DAT> d_queries(                                          \
      static_cast<DAT*>(queries.data_ptr()), dim_, n_queries);                 \
                                                                               \
  jasper::search_params params{                                                 \
      .k          = static_cast<uint32_t>(k),                                  \
      .beam_width = static_cast<uint32_t>(beam_width),                         \
      .limit      = static_cast<uint32_t>(limit),                              \
      .get_visited = false,                                                    \
  };                                                                           \
                                                                               \
  auto result = jasper::search(g, d_queries, params);                          \
                                                                               \
  DLDevice device = queries.device();                                          \
  cudaStream_t stream = static_cast<cudaStream_t>(                             \
      TVMFFIEnvGetStream(device.device_type, device.device_id));               \
                                                                               \
  uint32_t total = n_queries * static_cast<uint32_t>(k);                       \
  uint32_t threads = 256;                                                      \
  uint32_t blocks = (total + threads - 1) / threads;                           \
                                                                               \
  unpack_results_kernel<<<blocks, threads, 0, stream>>>(                       \
      result.frontier,                                                         \
      static_cast<int32_t*>(out_indices.data_ptr()),                           \
      static_cast<float*>(out_distances.data_ptr()),                           \
      total);                                                                  \
                                                                               \
  cudaFree(result.frontier);                                                   \
}

JASPER_FOR_EACH_CONFIG(DEFINE_OPS)
#undef DEFINE_OPS

// ── Free (config-agnostic via variant visit) ───────────────────
void FreeGraph(int64_t handle) {
  std::lock_guard<std::mutex> lock(g_mutex);
  auto it = g_graphs.find(handle);
  TVM_FFI_ICHECK(it != g_graphs.end()) << "Invalid handle: " << handle;

  std::visit([](auto& g) {
    using T = std::decay_t<decltype(g)>;
    if constexpr (!std::is_same_v<T, std::monostate>) {
      if (g.edges)        cudaFree(g.edges);
      if (g.edge_counts)  cudaFree(g.edge_counts);
      if (g.vectors.data) cudaFree(g.vectors.data);
    }
  }, it->second);

  g_graphs.erase(it);
}

// ── Info (config-agnostic) ─────────────────────────────────────
int64_t GetNumVectors(int64_t handle) {
  std::lock_guard<std::mutex> lock(g_mutex);
  auto it = g_graphs.find(handle);
  TVM_FFI_ICHECK(it != g_graphs.end()) << "Invalid handle";
  return std::visit([](auto& g) -> int64_t {
    using T = std::decay_t<decltype(g)>;
    if constexpr (!std::is_same_v<T, std::monostate>)
      return static_cast<int64_t>(g.n_vectors);
    else
      return 0;
  }, it->second);
}

int64_t GetDim(int64_t handle) {
  std::lock_guard<std::mutex> lock(g_mutex);
  auto it = g_graphs.find(handle);
  TVM_FFI_ICHECK(it != g_graphs.end()) << "Invalid handle";
  return std::visit([](auto& g) -> int64_t {
    using T = std::decay_t<decltype(g)>;
    if constexpr (!std::is_same_v<T, std::monostate>)
      return static_cast<int64_t>(g.vectors.dim);
    else
      return 0;
  }, it->second);
}

// ── Export all generated functions ──────────────────────────────
#define EXPORT_OPS(id, ...)                                                    \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(jasper_load_##id,      jasper_ffi::LoadGraph_##id);      \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(jasper_construct_##id,  jasper_ffi::ConstructGraph_##id); \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(jasper_search_##id,     jasper_ffi::Search_##id);

JASPER_FOR_EACH_CONFIG(EXPORT_OPS)
#undef EXPORT_OPS

TVM_FFI_DLL_EXPORT_TYPED_FUNC(jasper_free_graph,    jasper_ffi::FreeGraph);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(jasper_get_n_vectors, jasper_ffi::GetNumVectors);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(jasper_get_dim,       jasper_ffi::GetDim);

} // namespace jasper_ffi