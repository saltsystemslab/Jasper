#include <tvm/ffi/tvm_ffi.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <thrust/pair.h>
#include <cuda_fp16.h>

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
  X(f16_r8_l2,    uint32_t, 8,  __half, float, jasper::distance_func::L2)      \
  X(f16_r16_l2,   uint32_t, 16, __half, float, jasper::distance_func::L2)      \
  X(f16_r32_l2,   uint32_t, 32, __half, float, jasper::distance_func::L2)      \
  X(f16_r64_l2,   uint32_t, 64, __half, float, jasper::distance_func::L2)      \
  X(f16_r8_ip,    uint32_t, 8,  __half, float, jasper::distance_func::INNER_PRODUCT) \
  X(f16_r16_ip,   uint32_t, 16, __half, float, jasper::distance_func::INNER_PRODUCT) \
  X(f16_r32_ip,   uint32_t, 32, __half, float, jasper::distance_func::INNER_PRODUCT) \
  X(f16_r64_ip,   uint32_t, 64, __half, float, jasper::distance_func::INNER_PRODUCT)

// ── Generate graph config types ────────────────────────────────
#define DECLARE_CONFIG(id, IDX, R, DAT, DIST, FUNC) \
  using cfg_##id = jasper::graph_config<IDX, R, DAT, DIST, FUNC>;

JASPER_FOR_EACH_CONFIG(DECLARE_CONFIG)
#undef DECLARE_CONFIG

// ── Generate construct config types ────────────────────────────
// (block_size=64, tile_size=4, R from graph config, L=64)
#define DECLARE_CONSTRUCT_CONFIG(id, IDX, R, DAT, DIST, FUNC) \
  using construct_cfg_##id = jasper::graph_construct_config<cfg_##id, 64, 4, R, 64>;

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

#define DEFINE_OPS(id, IDX, R, DAT, DIST, FUNC)                                \
                                                                               \
int64_t LoadGraph_##id(ffi::String path, int64_t dim, bool on_host) {          \
  auto g = jasper::load_graph_from_file<cfg_##id>(                             \
      std::string(path), static_cast<uint32_t>(dim), on_host);                 \
  /* rebuild the forward stable_id -> slot table from the loaded slot_to_id */ \
  if (!on_host) jasper::build_id_map<cfg_##id>(g);                              \
  std::lock_guard<std::mutex> lock(g_mutex);                                   \
  int64_t handle = g_next_handle++;                                            \
  g_graphs[handle] = std::move(g);                                             \
  return handle;                                                               \
}                                                                              \
                                                                               \
void SaveGraph_##id(int64_t handle, ffi::String path) {                        \
  jasper::graph<cfg_##id>* gp;                                                 \
  {                                                                            \
    std::lock_guard<std::mutex> lock(g_mutex);                                 \
    auto it = g_graphs.find(handle);                                           \
    TVM_FFI_ICHECK(it != g_graphs.end()) << "Invalid handle: " << handle;      \
    gp = &std::get<jasper::graph<cfg_##id>>(it->second);                       \
  }                                                                            \
  jasper::save_graph_to_file<cfg_##id>(*gp, std::string(path));                \
}                                                                              \
                                                                               \
int64_t ConstructGraph_##id(ffi::TensorView vectors,                           \
                            int64_t dim,                                       \
                            double alpha,                                      \
                            int64_t workspace_budget_bytes,                    \
                            bool on_host) {                                    \
  uint32_t n_vectors = static_cast<uint32_t>(vectors.size(0));                 \
  uint32_t d = static_cast<uint32_t>(dim);                                     \
                                                                               \
  jasper::vector_view<DAT> vecs(                                               \
      static_cast<DAT*>(vectors.data_ptr()), d, n_vectors, false);             \
                                                                               \
  jasper::graph_construct_params<construct_cfg_##id> params;                   \
  jasper::graph_construct_workspace<construct_cfg_##id> ws;                    \
  uint32_t max_batch_size = min(                                               \
    ws.max_batch_size_for_budget(workspace_budget_bytes),                      \
    n_vectors / 50                                                             \
  );                                                                           \
                                                                               \
  params.data_vectors   = vecs;                                                \
  params.alpha          = static_cast<float>(alpha);                           \
  params.max_batch_size = max_batch_size;                                      \
  params.on_host        = on_host;                                             \
                                                                               \
  auto g = jasper::construct_graph<construct_cfg_##id>(params);                \
                                                                               \
  std::lock_guard<std::mutex> lock(g_mutex);                                   \
  int64_t handle = g_next_handle++;                                            \
  g_graphs[handle] = std::move(g);                                             \
  return handle;                                                               \
}                                                                              \
                                                                               \
void GetVector_##id(int64_t handle, int64_t index,                             \
                    ffi::TensorView out) {                                     \
  jasper::graph<cfg_##id>* gp;                                                 \
  {                                                                            \
    std::lock_guard<std::mutex> lock(g_mutex);                                 \
    auto it = g_graphs.find(handle);                                           \
    TVM_FFI_ICHECK(it != g_graphs.end()) << "Invalid handle: " << handle;      \
    gp = &std::get<jasper::graph<cfg_##id>>(it->second);                       \
  }                                                                            \
  auto& g = *gp;                                                               \
  /* translate stable id -> internal slot (identity when labels are off) */    \
  IDX h_id = static_cast<IDX>(index);                                          \
  IDX *d_id, *d_slot;                                                          \
  cudaMalloc(&d_id, sizeof(IDX)); cudaMalloc(&d_slot, sizeof(IDX));            \
  cudaMemcpy(d_id, &h_id, sizeof(IDX), cudaMemcpyHostToDevice);                \
  jasper::ids_to_slots_kernel<IDX><<<1, jasper::ID_MAP_TILE>>>(               \
      jasper::id_map_of<cfg_##id>(g), d_id, d_slot, 1);                        \
  cudaDeviceSynchronize();                                                     \
  IDX idx; cudaMemcpy(&idx, d_slot, sizeof(IDX), cudaMemcpyDeviceToHost);      \
  cudaFree(d_id); cudaFree(d_slot);                                            \
  TVM_FFI_ICHECK(idx < g.n_vectors) << "Stable id " << h_id << " not found";   \
                                                                               \
  uint32_t seg_id    = jasper::graph<cfg_##id>::segment_of(idx);               \
  uint32_t local_idx = jasper::graph<cfg_##id>::local_of(idx);                 \
  uint32_t padded_dim = jasper::vector_view<DAT>::pad(g.dim);                  \
                                                                               \
  /* Copy the segment struct from device to read its device pointers */        \
  jasper::graph_segment<cfg_##id> h_seg;                                       \
  cudaMemcpy(&h_seg,                                                           \
             thrust::raw_pointer_cast(g.segments.data()) + seg_id,             \
             sizeof(h_seg), cudaMemcpyDeviceToHost);                           \
                                                                               \
  cudaMemcpy(                                                                  \
      static_cast<DAT*>(out.data_ptr()),                                       \
      h_seg.vectors.data + static_cast<size_t>(local_idx) * padded_dim,        \
      sizeof(DAT) * g.dim,                                                     \
      cudaMemcpyDeviceToDevice);                                               \
}                                                                              \
                                                                               \
void Search_##id(int64_t handle,                                               \
                 ffi::TensorView queries,                                      \
                 ffi::TensorView out_indices,                                  \
                 ffi::TensorView out_distances,                                \
                 int64_t k, int64_t beam_width, int64_t limit,                 \
                 bool print_throughput) {                                      \
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
  uint32_t dim_ = g.dim;                                                       \
                                                                               \
  jasper::vector_view<DAT> d_queries(                                          \
      static_cast<DAT*>(queries.data_ptr()), dim_, n_queries, false);          \
                                                                               \
  jasper::search_params params{                                                \
      .k          = static_cast<uint32_t>(k),                                  \
      .beam_width = static_cast<uint32_t>(beam_width),                         \
      .limit      = static_cast<uint32_t>(limit),                              \
      .get_visited = false,                                                    \
  };                                                                           \
                                                                               \
  cudaEvent_t e0, e1;                                                          \
  cudaEventCreate(&e0);                                                        \
  cudaEventCreate(&e1);                                                        \
                                                                               \
  cudaEventRecord(e0);                                                         \
  auto result = jasper::search(g, d_queries, params);                          \
  cudaEventRecord(e1);                                                         \
  cudaEventSynchronize(e1);                                                    \
                                                                               \
  float duration_ms = 0;                                                       \
  cudaEventElapsedTime(&duration_ms, e0, e1);                                  \
                                                                               \
  cudaEventDestroy(e0);                                                        \
  cudaEventDestroy(e1);                                                        \
                                                                               \
  if (print_throughput)                                                        \
    std::cout << "[beam_search] duration=" << duration_ms                      \
      << "ms, throughput=" << (n_queries * 1000.0f)/duration_ms                \
      << std::endl;                                                            \
                                                                               \
  DLDevice device = queries.device();                                          \
  cudaStream_t stream = static_cast<cudaStream_t>(                             \
      TVMFFIEnvGetStream(device.device_type, device.device_id));               \
                                                                               \
  uint32_t total = n_queries * static_cast<uint32_t>(k);                       \
  uint32_t threads = 256;                                                      \
  uint32_t blocks = (total + threads - 1) / threads;                           \
                                                                               \
  /* translate internal slots -> stable ids before returning */               \
  jasper::translate_slots_to_ids_kernel<cfg_##id>                              \
      <<<blocks, threads, 0, stream>>>(g.view(), result.frontier, total);      \
                                                                               \
  unpack_results_kernel<<<blocks, threads, 0, stream>>>(                       \
      result.frontier,                                                         \
      static_cast<int32_t*>(out_indices.data_ptr()),                           \
      static_cast<float*>(out_distances.data_ptr()),                           \
      total);                                                                  \
                                                                               \
  cudaFree(result.frontier);                                                   \
}                                                                              \
                                                                               \
/* ids: contiguous CPU int32 tensor of vertex ids to soft-delete */            \
void MarkDeleted_##id(int64_t handle, ffi::TensorView ids) {                   \
  jasper::graph<cfg_##id>* gp;                                                 \
  {                                                                            \
    std::lock_guard<std::mutex> lock(g_mutex);                                 \
    auto it = g_graphs.find(handle);                                           \
    TVM_FFI_ICHECK(it != g_graphs.end()) << "Invalid handle: " << handle;      \
    gp = &std::get<jasper::graph<cfg_##id>>(it->second);                       \
  }                                                                            \
  uint32_t n_ids = static_cast<uint32_t>(ids.size(0));                         \
  const int32_t* p = static_cast<const int32_t*>(ids.data_ptr());              \
  std::vector<IDX> host_ids(n_ids);                                            \
  for (uint32_t i = 0; i < n_ids; i++)                                         \
    host_ids[i] = static_cast<IDX>(p[i]);                                      \
  jasper::mark_deleted<construct_cfg_##id>(*gp, host_ids.data(),               \
                                           static_cast<IDX>(n_ids));           \
}                                                                              \
                                                                               \
void Consolidate_##id(int64_t handle, double alpha) {                          \
  jasper::graph<cfg_##id>* gp;                                                 \
  {                                                                            \
    std::lock_guard<std::mutex> lock(g_mutex);                                 \
    auto it = g_graphs.find(handle);                                           \
    TVM_FFI_ICHECK(it != g_graphs.end()) << "Invalid handle: " << handle;      \
    gp = &std::get<jasper::graph<cfg_##id>>(it->second);                       \
  }                                                                            \
  jasper::consolidate<construct_cfg_##id>(*gp, static_cast<float>(alpha));     \
}                                                                              \
                                                                               \
void Compact_##id(int64_t handle) {                                            \
  jasper::graph<cfg_##id>* gp;                                                 \
  {                                                                            \
    std::lock_guard<std::mutex> lock(g_mutex);                                 \
    auto it = g_graphs.find(handle);                                           \
    TVM_FFI_ICHECK(it != g_graphs.end()) << "Invalid handle: " << handle;      \
    gp = &std::get<jasper::graph<cfg_##id>>(it->second);                       \
  }                                                                            \
  jasper::compact<construct_cfg_##id>(*gp, static_cast<IDX>(0));               \
}                                                                              \
                                                                               \
/* Append a batch of vectors ([n, dim], __half, device). Returns the assigned  \
   stable ids written (int32) into out_ids[0:n]. */                            \
void Append_##id(int64_t handle, ffi::TensorView vectors, double alpha,        \
                 ffi::TensorView out_ids) {                                    \
  jasper::graph<cfg_##id>* gp;                                                 \
  {                                                                            \
    std::lock_guard<std::mutex> lock(g_mutex);                                 \
    auto it = g_graphs.find(handle);                                           \
    TVM_FFI_ICHECK(it != g_graphs.end()) << "Invalid handle: " << handle;      \
    gp = &std::get<jasper::graph<cfg_##id>>(it->second);                       \
  }                                                                            \
  uint32_t n = static_cast<uint32_t>(vectors.size(0));                         \
  jasper::vector_view<DAT> vecs(                                               \
      static_cast<DAT*>(vectors.data_ptr()), gp->dim, n, false);               \
  auto ids = jasper::append_batch<construct_cfg_##id>(                         \
      *gp, vecs, static_cast<float>(alpha));                                   \
  std::vector<int32_t> h(ids.size());                                          \
  for (size_t i = 0; i < ids.size(); i++) h[i] = static_cast<int32_t>(ids[i]); \
  cudaMemcpy(out_ids.data_ptr(), h.data(),                                     \
             h.size() * sizeof(int32_t), cudaMemcpyDefault);                   \
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
      if (g.id_map) {
        using index_t = typename T::index_t;
        jasper::id_map_destroy<index_t>(
            static_cast<jasper::id_map_t<index_t>*>(g.id_map));
        g.id_map = nullptr;
      }
      g.deallocate();
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

int64_t GetNumTombstoned(int64_t handle) {
  std::lock_guard<std::mutex> lock(g_mutex);
  auto it = g_graphs.find(handle);
  TVM_FFI_ICHECK(it != g_graphs.end()) << "Invalid handle";
  return std::visit([](auto& g) -> int64_t {
    using T = std::decay_t<decltype(g)>;
    if constexpr (!std::is_same_v<T, std::monostate>)
      return static_cast<int64_t>(g.n_deleted);
    else
      return 0;
  }, it->second);
}

// Reserve `count` fresh stable ids and write them (int32) into out_ids.
// Config-agnostic: only advances the monotonic counter.
void ReserveIds(int64_t handle, ffi::TensorView out_ids, int64_t count) {
  std::lock_guard<std::mutex> lock(g_mutex);
  auto it = g_graphs.find(handle);
  TVM_FFI_ICHECK(it != g_graphs.end()) << "Invalid handle";
  std::visit([&](auto& g) {
    using T = std::decay_t<decltype(g)>;
    if constexpr (!std::is_same_v<T, std::monostate>) {
      using index_t = typename T::index_t;
      // Same write lock append_batch uses, so the next_id bump can't race a
      // concurrent append (both advance the monotonic id counter).
      auto rwlk = g.lock_exclusive();
      index_t start = g.next_id;
      g.next_id += static_cast<index_t>(count);
      std::vector<int32_t> host(count);
      for (int64_t i = 0; i < count; i++)
        host[i] = static_cast<int32_t>(start + static_cast<index_t>(i));
      cudaMemcpy(out_ids.data_ptr(), host.data(),
                 count * sizeof(int32_t), cudaMemcpyDefault);
    }
  }, it->second);
}

int64_t GetNumLive(int64_t handle) {
  std::lock_guard<std::mutex> lock(g_mutex);
  auto it = g_graphs.find(handle);
  TVM_FFI_ICHECK(it != g_graphs.end()) << "Invalid handle";
  return std::visit([](auto& g) -> int64_t {
    using T = std::decay_t<decltype(g)>;
    if constexpr (!std::is_same_v<T, std::monostate>)
      return static_cast<int64_t>(g.n_vectors - g.n_deleted);
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
      return static_cast<int64_t>(g.dim);
    else
      return 0;
  }, it->second);
}

// ── Export all generated functions ──────────────────────────────
#define EXPORT_OPS(id, ...)                                                    \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(jasper_load_##id,      jasper_ffi::LoadGraph_##id);      \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(jasper_construct_##id,  jasper_ffi::ConstructGraph_##id); \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(jasper_search_##id,     jasper_ffi::Search_##id); \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(jasper_save_##id,       jasper_ffi::SaveGraph_##id); \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(jasper_get_vector_##id, jasper_ffi::GetVector_##id); \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(jasper_mark_deleted_##id, jasper_ffi::MarkDeleted_##id); \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(jasper_consolidate_##id,  jasper_ffi::Consolidate_##id); \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(jasper_compact_##id,      jasper_ffi::Compact_##id); \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(jasper_append_##id,       jasper_ffi::Append_##id);

JASPER_FOR_EACH_CONFIG(EXPORT_OPS)
#undef EXPORT_OPS

TVM_FFI_DLL_EXPORT_TYPED_FUNC(jasper_free_graph,    jasper_ffi::FreeGraph);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(jasper_get_n_vectors, jasper_ffi::GetNumVectors);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(jasper_get_n_tombstoned, jasper_ffi::GetNumTombstoned);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(jasper_get_n_live,    jasper_ffi::GetNumLive);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(jasper_reserve_ids,   jasper_ffi::ReserveIds);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(jasper_get_dim,       jasper_ffi::GetDim);

} // namespace jasper_ffi