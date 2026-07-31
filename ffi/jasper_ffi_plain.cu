// ffi/jasper_ffi_plain.cu
//
// Plain (non-directional) graph ops, plus the config-agnostic handle-table
// state and free/info ops shared by every config (plain and directional
// alike). See jasper_ffi_common.cuh for why this is split across files.
#include "jasper_ffi_common.cuh"

namespace jasper_ffi {

// ── Global handle table (single definition for the whole shared library) ──
std::unordered_map<int64_t, GraphVariant> g_graphs;
int64_t g_next_handle = 0;
std::mutex g_mutex;

JASPER_FOR_EACH_CONFIG(DEFINE_OPS)
#undef DEFINE_OPS

// ── Free (config-agnostic via variant visit) ───────────────────
void FreeGraph(int64_t handle) {
  std::lock_guard<std::mutex> lock(g_mutex);
  auto it = g_graphs.find(handle);
  TVM_FFI_ICHECK(it != g_graphs.end()) << "Invalid handle: " << handle;

  std::visit([handle](auto& g) {
    using T = std::decay_t<decltype(g)>;
    if constexpr (!std::is_same_v<T, std::monostate>) {
      if constexpr (T::use_lsh) {
        auto& dm = dir_meta_map<T>();
        auto dit = dm.find(handle);
        if (dit != dm.end()) {
          if (dit->second.has_pq) dit->second.codebooks.free();
          if (dit->second.d_rotation) cudaFree(dit->second.d_rotation);
          dm.erase(dit);
        }
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

JASPER_FOR_EACH_CONFIG(EXPORT_OPS)
#undef EXPORT_OPS

TVM_FFI_DLL_EXPORT_TYPED_FUNC(jasper_free_graph,    jasper_ffi::FreeGraph);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(jasper_get_n_vectors, jasper_ffi::GetNumVectors);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(jasper_get_dim,       jasper_ffi::GetDim);

} // namespace jasper_ffi
