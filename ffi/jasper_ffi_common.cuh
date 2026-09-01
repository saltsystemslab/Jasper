// jasper_ffi_common.cuh
//
// Shared declarations for the jasper_ffi target. This target's config matrix
// (4 plain + 16 directional graph_configs) is too expensive to compile as one
// translation unit — directional_search()/pq_search() alone each expand into
// 5 full graph_beam_search_kernel instantiations per config (one per
// MAX_SEARCH_WIDTH branch; get_visited/max_result_size are runtime kernel
// args, not template axes, see beam_search/api.cuh), on top of construct_graph
// and the LSH/PQ pipeline kernels. So the actual
// per-config op bodies (DEFINE_OPS / DEFINE_DIRECTIONAL_OPS) are only DEFINED
// here as macros; each ffi/jasper_ffi_*.cu file includes this header and
// invokes them for its own slice of the config table, so nvcc can compile
// those files as independent, parallelizable translation units (the build
// already sets -rdc=true / CUDA_SEPARABLE_COMPILATION ON for this target, so
// the resulting device code links back together normally).
#pragma once

#include <tvm/ffi/tvm_ffi.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <thrust/pair.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#include "jasper/jasper.cuh"

#include <unordered_map>
#include <mutex>
#include <variant>
#include <string>
#include <vector>

namespace jasper_ffi {

namespace ffi = tvm::ffi;

// ── Enumerate all plain configs ──────────────────────────────────
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

// ── Directional (LSH + PQ) configs ──────────────────────────────
// (CONFIG_ID, INDEX_T, N_NEIGHBORS, DATA_T, DISTANCE_T, DIST_FUNC, K_RANKS, PACKED_T)
// PACKED_T is chosen by dim bucket: uint8_t for dim<=128 (7-bit coord),
// uint16_t for dim>128.
//
// Split into 4 buckets of 4 configs each (by distance x k_ranks) so each
// bucket's ops can live in its own translation unit — see
// ffi/jasper_ffi_directional_{l2,ip}_{k4,k16}.cu.
#define JASPER_FOR_EACH_DIRECTIONAL_CONFIG_L2_K4(X)                            \
  X(f16_r32_l2_k4_d128,   uint32_t, 32, __half, float, jasper::distance_func::L2, 4, uint8_t)  \
  X(f16_r32_l2_k4_d32678, uint32_t, 32, __half, float, jasper::distance_func::L2, 4, uint16_t) \
  X(f16_r64_l2_k4_d128,   uint32_t, 64, __half, float, jasper::distance_func::L2, 4, uint8_t)  \
  X(f16_r64_l2_k4_d32678, uint32_t, 64, __half, float, jasper::distance_func::L2, 4, uint16_t)

#define JASPER_FOR_EACH_DIRECTIONAL_CONFIG_L2_K16(X)                            \
  X(f16_r32_l2_k16_d128,   uint32_t, 32, __half, float, jasper::distance_func::L2, 16, uint8_t)  \
  X(f16_r32_l2_k16_d32678, uint32_t, 32, __half, float, jasper::distance_func::L2, 16, uint16_t) \
  X(f16_r64_l2_k16_d128,   uint32_t, 64, __half, float, jasper::distance_func::L2, 16, uint8_t)  \
  X(f16_r64_l2_k16_d32678, uint32_t, 64, __half, float, jasper::distance_func::L2, 16, uint16_t)

#define JASPER_FOR_EACH_DIRECTIONAL_CONFIG_IP_K4(X)                             \
  X(f16_r32_ip_k4_d128,   uint32_t, 32, __half, float, jasper::distance_func::INNER_PRODUCT, 4, uint8_t)  \
  X(f16_r32_ip_k4_d32678, uint32_t, 32, __half, float, jasper::distance_func::INNER_PRODUCT, 4, uint16_t) \
  X(f16_r64_ip_k4_d128,   uint32_t, 64, __half, float, jasper::distance_func::INNER_PRODUCT, 4, uint8_t)  \
  X(f16_r64_ip_k4_d32678, uint32_t, 64, __half, float, jasper::distance_func::INNER_PRODUCT, 4, uint16_t)

#define JASPER_FOR_EACH_DIRECTIONAL_CONFIG_IP_K16(X)                            \
  X(f16_r32_ip_k16_d128,   uint32_t, 32, __half, float, jasper::distance_func::INNER_PRODUCT, 16, uint8_t)  \
  X(f16_r32_ip_k16_d32678, uint32_t, 32, __half, float, jasper::distance_func::INNER_PRODUCT, 16, uint16_t) \
  X(f16_r64_ip_k16_d128,   uint32_t, 64, __half, float, jasper::distance_func::INNER_PRODUCT, 16, uint8_t)  \
  X(f16_r64_ip_k16_d32678, uint32_t, 64, __half, float, jasper::distance_func::INNER_PRODUCT, 16, uint16_t)

// Full set — used only for type declarations that must span every
// directional config (cfg_##id/construct_cfg_##id aliases, GraphVariant).
// Actual op instantiation happens per-bucket in each
// ffi/jasper_ffi_directional_*.cu (see DEFINE_DIRECTIONAL_OPS below).
#define JASPER_FOR_EACH_DIRECTIONAL_CONFIG(X)  \
  JASPER_FOR_EACH_DIRECTIONAL_CONFIG_L2_K4(X)  \
  JASPER_FOR_EACH_DIRECTIONAL_CONFIG_L2_K16(X) \
  JASPER_FOR_EACH_DIRECTIONAL_CONFIG_IP_K4(X)  \
  JASPER_FOR_EACH_DIRECTIONAL_CONFIG_IP_K16(X)

#define DECLARE_DIRECTIONAL_CONFIG(id, IDX, R, DAT, DIST, FUNC, KR, PACKEDT) \
  using cfg_##id = jasper::graph_config<IDX, R, DAT, DIST, FUNC, true, KR, PACKEDT>; \
  using construct_cfg_##id = jasper::graph_construct_config<cfg_##id, 64, 4, R, 64>;

JASPER_FOR_EACH_DIRECTIONAL_CONFIG(DECLARE_DIRECTIONAL_CONFIG)
#undef DECLARE_DIRECTIONAL_CONFIG

// ── Graph storage using variant ────────────────────────────────
// Spans BOTH plain and directional configs, so plain and directional graphs
// share one handle namespace/free path. Naming every alternative type here
// is cheap (it doesn't instantiate any kernels) even though only a subset of
// their ops are compiled in any one translation unit.
#define VARIANT_ENTRY(id, ...) jasper::graph<cfg_##id>,
using GraphVariant = std::variant<
  JASPER_FOR_EACH_CONFIG(VARIANT_ENTRY)
  JASPER_FOR_EACH_DIRECTIONAL_CONFIG(VARIANT_ENTRY)
  std::monostate
>;
#undef VARIANT_ENTRY

// ── Per-handle directional state ────────────────────────────────
// Templated on the concrete graph type (GRAPH_T = jasper::graph<cfg_id>), so
// there's exactly one dir_meta_map<GRAPH_T>() per directional config — no
// variant-of-globals/codebooks needed, and FreeGraph can clean it up
// generically inside the std::visit over GraphVariant (see
// ffi/jasper_ffi_plain.cu). Being a template, this is safe to define here in
// the shared header: identical instantiations across translation units are
// merged by the linker like any other template (including the function-local
// static in dir_meta_map), so there's still exactly one map per GRAPH_T.
template <typename GRAPH_T>
struct dir_meta {
  bool has_lsh   = false;
  bool has_pq    = false;
  bool prerotate = false;
  typename GRAPH_T::data_t* d_rotation = nullptr;  // dim x dim, col-major, device; owned
  jasper::lsh_globals<GRAPH_T::k_ranks> globals{};
  jasper::pq_codebooks<GRAPH_T::pq_m, GRAPH_T::pq_k> codebooks{};  // owns d_centroids
};

template <typename GRAPH_T>
std::unordered_map<int64_t, dir_meta<GRAPH_T>>& dir_meta_map() {
  static std::unordered_map<int64_t, dir_meta<GRAPH_T>> m;
  return m;
}

// Builds a dim x dim random orthogonal rotation matrix (see
// jasper::set_rotation_matrix) and uploads it to device as __half,
// column-major — the layout jasper::rotate_data_vec expects.
inline __half* make_device_rotation_matrix(uint32_t dim, uint64_t seed) {
  std::vector<float> h_P_f(static_cast<size_t>(dim) * dim);
  std::vector<float> h_Pt_f(static_cast<size_t>(dim) * dim);  // unused, required by the API
  jasper::set_rotation_matrix(static_cast<int>(dim), h_P_f.data(), h_Pt_f.data(), seed);

  std::vector<__half> h_P(h_P_f.size());
  for (size_t i = 0; i < h_P.size(); ++i) h_P[i] = static_cast<__half>(h_P_f[i]);

  __half* d_P = nullptr;
  cudaMalloc(&d_P, sizeof(__half) * h_P.size());
  cudaMemcpy(d_P, h_P.data(), sizeof(__half) * h_P.size(), cudaMemcpyHostToDevice);
  return d_P;
}

// Rotates queries (row-major [n_queries, dim], contiguous stride == dim) into
// a freshly allocated device buffer of the same shape. Caller must cudaFree it.
inline __half* rotate_query_batch(const __half* d_queries, uint32_t n_queries,
                                  uint32_t dim, const __half* d_rotation) {
  __half* d_out = nullptr;
  cudaMalloc(&d_out, sizeof(__half) * static_cast<size_t>(n_queries) * dim);
  cublasHandle_t handle;
  cublasCreate(&handle);
  jasper::rotate_data_vec<__half>(handle, const_cast<__half*>(d_queries), d_out,
                                  n_queries, dim, const_cast<__half*>(d_rotation));
  cublasDestroy(handle);
  return d_out;
}

// ── Global handle table ─────────────────────────────────────────
// Shared by every config across every translation unit, so this must have
// exactly one definition — see ffi/jasper_ffi_plain.cu.
extern std::unordered_map<int64_t, GraphVariant> g_graphs;
extern int64_t g_next_handle;
extern std::mutex g_mutex;

// ── Unpack kernel (shared across all configs) ──────────────────
// `static` so each translation unit that includes this header gets its own
// private (internal-linkage) copy — trivially cheap to duplicate, and avoids
// a device-link "multiple definition" across the split .cu files.
static __global__ void unpack_results_kernel(
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
// Defined here as a macro only — invoked per-config in ffi/jasper_ffi_plain.cu.

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

// ── Per-config directional (LSH + PQ) implementations ───────────
// Mirrors cmd/create_lsh_index.cu's pipeline: construct (optionally
// prerotated) -> build_lsh (generate_lsh_globals + populate_edge_lsh) ->
// build_pq (generate_pq_codebooks + populate_edge_pq + compute_vector_norms)
// -> directional_search / pq_search. Each step is its own op so callers
// choose which artifacts to build (see dir_meta's has_lsh/has_pq flags).
// Defined here as a macro only — invoked per-bucket in each
// ffi/jasper_ffi_directional_*.cu.

#define DEFINE_DIRECTIONAL_OPS(id, IDX, R, DAT, DIST, FUNC, KR, PACKEDT)       \
                                                                               \
int64_t ConstructDirectionalGraph_##id(ffi::TensorView vectors,                \
                                       int64_t dim,                           \
                                       double alpha,                         \
                                       int64_t workspace_budget_bytes,        \
                                       bool on_host,                         \
                                       bool prerotate,                       \
                                       int64_t prerotate_seed) {             \
  uint32_t n_vectors = static_cast<uint32_t>(vectors.size(0));               \
  uint32_t d = static_cast<uint32_t>(dim);                                   \
                                                                               \
  jasper::vector_view<DAT> vecs(                                             \
      static_cast<DAT*>(vectors.data_ptr()), d, n_vectors, false);           \
                                                                               \
  jasper::graph_construct_params<construct_cfg_##id> params;                 \
  jasper::graph_construct_workspace<construct_cfg_##id> ws;                  \
  uint32_t max_batch_size = min(                                             \
    ws.max_batch_size_for_budget(workspace_budget_bytes),                   \
    n_vectors / 50                                                          \
  );                                                                         \
                                                                               \
  params.data_vectors   = vecs;                                              \
  params.alpha          = static_cast<float>(alpha);                        \
  params.max_batch_size = max_batch_size;                                   \
  params.on_host        = on_host;                                          \
  params.prerotate      = prerotate;                                        \
  params.prerotate_seed = static_cast<uint32_t>(prerotate_seed);            \
                                                                               \
  auto g = jasper::construct_graph<construct_cfg_##id>(params);              \
                                                                               \
  using graph_t = jasper::graph<cfg_##id>;                                   \
  dir_meta<graph_t> meta;                                                    \
  meta.prerotate = prerotate;                                               \
  if (prerotate) {                                                          \
    meta.d_rotation = make_device_rotation_matrix(                          \
        d, static_cast<uint64_t>(prerotate_seed));                         \
  }                                                                          \
                                                                               \
  std::lock_guard<std::mutex> lock(g_mutex);                                 \
  int64_t handle = g_next_handle++;                                         \
  g_graphs[handle] = std::move(g);                                          \
  dir_meta_map<graph_t>()[handle] = meta;                                   \
  return handle;                                                             \
}                                                                             \
                                                                               \
void BuildLsh_##id(int64_t handle, int64_t n_samples, int64_t seed) {        \
  using graph_t = jasper::graph<cfg_##id>;                                   \
  graph_t* gp;                                                               \
  {                                                                           \
    std::lock_guard<std::mutex> lock(g_mutex);                              \
    auto it = g_graphs.find(handle);                                        \
    TVM_FFI_ICHECK(it != g_graphs.end()) << "Invalid handle: " << handle;     \
    gp = &std::get<graph_t>(it->second);                                    \
  }                                                                          \
  auto globals = gp->generate_lsh_globals(                                  \
      static_cast<uint32_t>(n_samples), static_cast<uint64_t>(seed));       \
  gp->populate_edge_lsh();                                                  \
                                                                               \
  std::lock_guard<std::mutex> lock(g_mutex);                                 \
  auto& meta = dir_meta_map<graph_t>()[handle];                             \
  meta.has_lsh = true;                                                      \
  meta.globals = globals;                                                   \
}                                                                             \
                                                                               \
void BuildPq_##id(int64_t handle, int64_t n_train, int64_t kmeans_iter,      \
                  int64_t seed) {                                           \
  using graph_t = jasper::graph<cfg_##id>;                                   \
  graph_t* gp;                                                               \
  {                                                                           \
    std::lock_guard<std::mutex> lock(g_mutex);                              \
    auto it = g_graphs.find(handle);                                        \
    TVM_FFI_ICHECK(it != g_graphs.end()) << "Invalid handle: " << handle;     \
    gp = &std::get<graph_t>(it->second);                                    \
  }                                                                          \
  auto cb = gp->generate_pq_codebooks(                                      \
      static_cast<uint32_t>(n_train), static_cast<uint32_t>(kmeans_iter),   \
      static_cast<uint64_t>(seed));                                        \
  gp->populate_edge_pq(cb);                                                 \
  gp->compute_vector_norms();                                               \
                                                                               \
  std::lock_guard<std::mutex> lock(g_mutex);                                 \
  auto& meta = dir_meta_map<graph_t>()[handle];                             \
  if (meta.has_pq) meta.codebooks.free();                                  \
  meta.has_pq    = true;                                                    \
  meta.codebooks = cb;                                                      \
}                                                                             \
                                                                               \
void DirectionalSearch_##id(int64_t handle,                                  \
                            ffi::TensorView queries,                        \
                            ffi::TensorView out_indices,                    \
                            ffi::TensorView out_distances,                  \
                            int64_t k, int64_t beam_width, int64_t limit,   \
                            bool print_throughput) {                        \
  using graph_t = jasper::graph<cfg_##id>;                                   \
  graph_t* gp;                                                               \
  dir_meta<graph_t> meta;                                                   \
  {                                                                           \
    std::lock_guard<std::mutex> lock(g_mutex);                              \
    auto it = g_graphs.find(handle);                                        \
    TVM_FFI_ICHECK(it != g_graphs.end()) << "Invalid handle: " << handle;     \
    gp = &std::get<graph_t>(it->second);                                    \
    auto mit = dir_meta_map<graph_t>().find(handle);                        \
    TVM_FFI_ICHECK(mit != dir_meta_map<graph_t>().end())                    \
        << "Handle " << handle << " has no directional metadata";           \
    meta = mit->second;                                                     \
  }                                                                          \
  TVM_FFI_ICHECK(meta.has_lsh)                                              \
      << "Graph " << handle << " has no LSH artifacts built — "            \
      << "call build_lsh() first";                                         \
  auto& g = *gp;                                                            \
                                                                               \
  uint32_t n_queries = static_cast<uint32_t>(queries.size(0));              \
  uint32_t dim_ = g.dim;                                                    \
                                                                               \
  DAT* d_query_data = static_cast<DAT*>(queries.data_ptr());                \
  DAT* d_rotated = nullptr;                                                 \
  if (meta.prerotate) {                                                     \
    d_rotated = rotate_query_batch(d_query_data, n_queries, dim_,           \
                                   meta.d_rotation);                        \
    d_query_data = d_rotated;                                              \
  }                                                                          \
                                                                               \
  jasper::vector_view<DAT> d_queries(d_query_data, dim_, n_queries, false); \
                                                                               \
  jasper::search_params params{                                            \
      .k           = static_cast<uint32_t>(k),                             \
      .beam_width  = static_cast<uint32_t>(beam_width),                    \
      .limit       = static_cast<uint32_t>(limit),                         \
      .get_visited = false,                                                \
  };                                                                         \
                                                                               \
  cudaEvent_t e0, e1;                                                       \
  cudaEventCreate(&e0);                                                     \
  cudaEventCreate(&e1);                                                     \
  cudaEventRecord(e0);                                                      \
  auto result = jasper::directional_search(g, meta.globals, d_queries, params); \
  cudaEventRecord(e1);                                                      \
  cudaEventSynchronize(e1);                                                 \
                                                                               \
  float duration_ms = 0;                                                    \
  cudaEventElapsedTime(&duration_ms, e0, e1);                               \
  cudaEventDestroy(e0);                                                     \
  cudaEventDestroy(e1);                                                     \
                                                                               \
  if (print_throughput)                                                     \
    std::cout << "[directional_search] duration=" << duration_ms            \
      << "ms, throughput=" << (n_queries * 1000.0f)/duration_ms            \
      << std::endl;                                                         \
                                                                               \
  DLDevice device = queries.device();                                      \
  cudaStream_t stream = static_cast<cudaStream_t>(                          \
      TVMFFIEnvGetStream(device.device_type, device.device_id));           \
                                                                               \
  uint32_t total = n_queries * static_cast<uint32_t>(k);                   \
  uint32_t threads = 256;                                                   \
  uint32_t blocks = (total + threads - 1) / threads;                       \
                                                                               \
  unpack_results_kernel<<<blocks, threads, 0, stream>>>(                   \
      result.frontier,                                                      \
      static_cast<int32_t*>(out_indices.data_ptr()),                       \
      static_cast<float*>(out_distances.data_ptr()),                       \
      total);                                                               \
                                                                               \
  cudaFree(result.frontier);                                               \
  if (d_rotated) cudaFree(d_rotated);                                      \
}                                                                             \
                                                                               \
void PqSearch_##id(int64_t handle,                                          \
                   ffi::TensorView queries,                                 \
                   ffi::TensorView out_indices,                             \
                   ffi::TensorView out_distances,                          \
                   int64_t k, int64_t beam_width, int64_t limit,           \
                   bool print_throughput) {                                \
  using graph_t = jasper::graph<cfg_##id>;                                   \
  graph_t* gp;                                                               \
  dir_meta<graph_t> meta;                                                   \
  {                                                                           \
    std::lock_guard<std::mutex> lock(g_mutex);                              \
    auto it = g_graphs.find(handle);                                        \
    TVM_FFI_ICHECK(it != g_graphs.end()) << "Invalid handle: " << handle;     \
    gp = &std::get<graph_t>(it->second);                                    \
    auto mit = dir_meta_map<graph_t>().find(handle);                        \
    TVM_FFI_ICHECK(mit != dir_meta_map<graph_t>().end())                    \
        << "Handle " << handle << " has no directional metadata";           \
    meta = mit->second;                                                     \
  }                                                                          \
  TVM_FFI_ICHECK(meta.has_pq)                                               \
      << "Graph " << handle << " has no PQ artifacts built — "             \
      << "call build_pq() first";                                          \
  auto& g = *gp;                                                            \
                                                                               \
  uint32_t n_queries = static_cast<uint32_t>(queries.size(0));              \
  uint32_t dim_ = g.dim;                                                    \
                                                                               \
  DAT* d_query_data = static_cast<DAT*>(queries.data_ptr());                \
  DAT* d_rotated = nullptr;                                                 \
  if (meta.prerotate) {                                                     \
    d_rotated = rotate_query_batch(d_query_data, n_queries, dim_,           \
                                   meta.d_rotation);                        \
    d_query_data = d_rotated;                                              \
  }                                                                          \
                                                                               \
  jasper::vector_view<DAT> d_queries(d_query_data, dim_, n_queries, false); \
                                                                               \
  jasper::search_params params{                                            \
      .k           = static_cast<uint32_t>(k),                             \
      .beam_width  = static_cast<uint32_t>(beam_width),                    \
      .limit       = static_cast<uint32_t>(limit),                         \
      .get_visited = false,                                                \
  };                                                                         \
                                                                               \
  cudaEvent_t e0, e1;                                                       \
  cudaEventCreate(&e0);                                                     \
  cudaEventCreate(&e1);                                                     \
  cudaEventRecord(e0);                                                      \
  auto result = jasper::pq_search(g, meta.codebooks.view(), d_queries, params); \
  cudaEventRecord(e1);                                                      \
  cudaEventSynchronize(e1);                                                 \
                                                                               \
  float duration_ms = 0;                                                    \
  cudaEventElapsedTime(&duration_ms, e0, e1);                               \
  cudaEventDestroy(e0);                                                     \
  cudaEventDestroy(e1);                                                     \
                                                                               \
  if (print_throughput)                                                     \
    std::cout << "[pq_search] duration=" << duration_ms                     \
      << "ms, throughput=" << (n_queries * 1000.0f)/duration_ms            \
      << std::endl;                                                         \
                                                                               \
  DLDevice device = queries.device();                                      \
  cudaStream_t stream = static_cast<cudaStream_t>(                          \
      TVMFFIEnvGetStream(device.device_type, device.device_id));           \
                                                                               \
  uint32_t total = n_queries * static_cast<uint32_t>(k);                   \
  uint32_t threads = 256;                                                   \
  uint32_t blocks = (total + threads - 1) / threads;                       \
                                                                               \
  unpack_results_kernel<<<blocks, threads, 0, stream>>>(                   \
      result.frontier,                                                      \
      static_cast<int32_t*>(out_indices.data_ptr()),                       \
      static_cast<float*>(out_distances.data_ptr()),                       \
      total);                                                               \
                                                                               \
  cudaFree(result.frontier);                                               \
  if (d_rotated) cudaFree(d_rotated);                                      \
}                                                                             \
                                                                               \
void SaveDirectionalGraph_##id(int64_t handle, ffi::String path) {          \
  using graph_t = jasper::graph<cfg_##id>;                                   \
  graph_t* gp;                                                               \
  dir_meta<graph_t> meta;                                                   \
  {                                                                           \
    std::lock_guard<std::mutex> lock(g_mutex);                              \
    auto it = g_graphs.find(handle);                                        \
    TVM_FFI_ICHECK(it != g_graphs.end()) << "Invalid handle: " << handle;     \
    gp = &std::get<graph_t>(it->second);                                    \
    auto mit = dir_meta_map<graph_t>().find(handle);                        \
    TVM_FFI_ICHECK(mit != dir_meta_map<graph_t>().end())                    \
        << "Handle " << handle << " has no directional metadata";           \
    meta = mit->second;                                                     \
  }                                                                          \
  jasper::save_directional_graph_to_file<cfg_##id>(                         \
      *gp,                                                                  \
      meta.has_lsh ? &meta.globals   : nullptr,                            \
      meta.has_pq  ? &meta.codebooks : nullptr,                            \
      std::string(path));                                                   \
}                                                                             \
                                                                               \
int64_t LoadDirectionalGraph_##id(ffi::String path, int64_t dim, bool on_host, \
                                  bool prerotate, int64_t prerotate_seed) {   \
  auto bundle = jasper::load_directional_graph_from_file<cfg_##id>(          \
      std::string(path), static_cast<uint32_t>(dim), on_host);              \
                                                                               \
  using graph_t = jasper::graph<cfg_##id>;                                   \
  dir_meta<graph_t> meta;                                                   \
  meta.has_lsh   = bundle.has_lsh;                                          \
  meta.has_pq    = bundle.has_pq;                                           \
  meta.globals   = bundle.globals;                                          \
  meta.codebooks = bundle.codebooks;                                        \
  meta.prerotate = prerotate;                                              \
  if (prerotate) {                                                         \
    meta.d_rotation = make_device_rotation_matrix(                         \
        static_cast<uint32_t>(dim), static_cast<uint64_t>(prerotate_seed)); \
  }                                                                          \
                                                                               \
  std::lock_guard<std::mutex> lock(g_mutex);                                \
  int64_t handle = g_next_handle++;                                        \
  g_graphs[handle] = std::move(bundle.g);                                  \
  dir_meta_map<graph_t>()[handle] = meta;                                  \
  return handle;                                                            \
}                                                                             \
                                                                               \
void GetVectorDirectional_##id(int64_t handle, int64_t index,                \
                               ffi::TensorView out) {                       \
  using graph_t = jasper::graph<cfg_##id>;                                   \
  graph_t* gp;                                                              \
  {                                                                          \
    std::lock_guard<std::mutex> lock(g_mutex);                             \
    auto it = g_graphs.find(handle);                                       \
    TVM_FFI_ICHECK(it != g_graphs.end()) << "Invalid handle: " << handle;    \
    gp = &std::get<graph_t>(it->second);                                   \
  }                                                                         \
  auto& g = *gp;                                                           \
  uint32_t idx = static_cast<uint32_t>(index);                             \
  TVM_FFI_ICHECK(idx < g.n_vectors) << "Index " << idx << " out of range";   \
                                                                               \
  uint32_t seg_id     = graph_t::segment_of(idx);                          \
  uint32_t local_idx  = graph_t::local_of(idx);                            \
  uint32_t padded_dim = jasper::vector_view<DAT>::pad(g.dim);              \
                                                                               \
  jasper::graph_segment<cfg_##id> h_seg;                                    \
  cudaMemcpy(&h_seg,                                                        \
             thrust::raw_pointer_cast(g.segments.data()) + seg_id,          \
             sizeof(h_seg), cudaMemcpyDeviceToHost);                        \
                                                                               \
  cudaMemcpy(                                                               \
      static_cast<DAT*>(out.data_ptr()),                                   \
      h_seg.vectors.data + static_cast<size_t>(local_idx) * padded_dim,     \
      sizeof(DAT) * g.dim,                                                 \
      cudaMemcpyDeviceToDevice);                                          \
}                                                                             \
                                                                               \
/* bit0 = has_lsh, bit1 = has_pq. 0 if the handle has no directional        \
   metadata at all (shouldn't happen for a directional-config handle, but   \
   keeps this query total rather than throwing). */                        \
int64_t GetDirectionalFlags_##id(int64_t handle) {                          \
  using graph_t = jasper::graph<cfg_##id>;                                   \
  std::lock_guard<std::mutex> lock(g_mutex);                                \
  auto& dm = dir_meta_map<graph_t>();                                       \
  auto it = dm.find(handle);                                                \
  if (it == dm.end()) return 0;                                            \
  return (it->second.has_lsh ? 1 : 0) | (it->second.has_pq ? 2 : 0);        \
}

// ── Free / info (config-agnostic; defined once in ffi/jasper_ffi_plain.cu) ──
void FreeGraph(int64_t handle);
int64_t GetNumVectors(int64_t handle);
int64_t GetNumTombstoned(int64_t handle);
int64_t GetNumLive(int64_t handle);
void ReserveIds(int64_t handle, ffi::TensorView out_ids, int64_t count);
int64_t GetDim(int64_t handle);

// ── Export macros ────────────────────────────────────────────────
// Invoked per-file, for whichever configs that file actually defined ops for.
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

#define EXPORT_DIRECTIONAL_OPS(id, ...)                                                              \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(jasper_construct_directional_##id, jasper_ffi::ConstructDirectionalGraph_##id); \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(jasper_build_lsh_##id,             jasper_ffi::BuildLsh_##id);                  \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(jasper_build_pq_##id,              jasper_ffi::BuildPq_##id);                   \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(jasper_directional_search_##id,    jasper_ffi::DirectionalSearch_##id);         \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(jasper_pq_search_##id,             jasper_ffi::PqSearch_##id);                  \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(jasper_save_directional_##id,      jasper_ffi::SaveDirectionalGraph_##id);      \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(jasper_load_directional_##id,      jasper_ffi::LoadDirectionalGraph_##id);      \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(jasper_get_vector_directional_##id, jasper_ffi::GetVectorDirectional_##id);     \
  TVM_FFI_DLL_EXPORT_TYPED_FUNC(jasper_get_directional_flags_##id, jasper_ffi::GetDirectionalFlags_##id);

}  // namespace jasper_ffi
