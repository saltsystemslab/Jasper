#include <tvm/ffi/tvm_ffi.h>
#include <tvm/ffi/extra/c_env_api.h>
#include <thrust/pair.h>

#include "jasper/jasper.cuh"

namespace jasper_ffi {

namespace ffi = tvm::ffi;

// ── Concrete config (templates must be instantiated) ───────────
using Cfg = jasper::graph_config<
    uint32_t,                      // INDEX_T
    64,                            // N_NEIGHBORS
    float,                         // DATA_T
    float,                         // DISTANCE_T
    jasper::distance_func::L2      // distance function
>;

// ── Global state ───────────────────────────────────────────────
static jasper::graph<Cfg> g_graph;
static bool g_graph_loaded = false;

// ── Load graph ─────────────────────────────────────────────────
// jasper.load_graph(path: str, dim: int)
void LoadGraph(ffi::String path, int64_t dim) {
  g_graph = jasper::load_graph_from_file<Cfg>(
      std::string(path), static_cast<uint32_t>(dim));
  g_graph_loaded = true;
}

// ── Save graph ─────────────────────────────────────────────────
// jasper.save_graph(path: str)
void SaveGraph(ffi::String path) {
  TVM_FFI_ICHECK(g_graph_loaded) << "No graph loaded";
  jasper::save_graph_to_file<Cfg>(g_graph, std::string(path));
}


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

// ── Search ─────────────────────────────────────────────────────
// jasper.search(queries, out_indices, out_distances, k, beam_width, limit)
//
// queries:        float32 [n_queries, dim] on CUDA
// out_indices:    int32   [n_queries, k]   on CUDA
// out_distances:  float32 [n_queries, k]   on CUDA
void Search(ffi::TensorView queries,
            ffi::TensorView out_indices,
            ffi::TensorView out_distances,
            int64_t k,
            int64_t beam_width,
            int64_t limit) {
  TVM_FFI_ICHECK(g_graph_loaded) << "No graph loaded. Call load_graph first.";

  TVM_FFI_ICHECK(queries.ndim() == 2)
      << "queries must be 2D [n_queries, dim]";
  TVM_FFI_ICHECK(static_cast<uint32_t>(queries.size(1)) == g_graph.vectors.dim)
      << "query dim (" << queries.size(1) << ") must match graph dim ("
      << g_graph.vectors.dim << ")";

  DLDataType f32{kDLFloat, 32, 1};
  DLDataType i32{kDLInt, 32, 1};
  TVM_FFI_ICHECK(queries.dtype() == f32) << "queries must be float32";
  TVM_FFI_ICHECK(out_indices.dtype() == i32) << "out_indices must be int32";
  TVM_FFI_ICHECK(out_distances.dtype() == f32) << "out_distances must be float32";

  uint32_t n_queries = static_cast<uint32_t>(queries.size(0));
  uint32_t dim = g_graph.vectors.dim;

  // Wrap query tensor as vector_view (zero-copy)
  jasper::vector_view<float> d_queries(
      static_cast<float*>(queries.data_ptr()), dim, n_queries);

  jasper::SearchParams params{
      .k          = static_cast<uint32_t>(k),
      .beam_width = static_cast<uint32_t>(beam_width),
      .limit      = static_cast<uint32_t>(limit),
      .get_visited = false,
  };

  auto result = jasper::search(g_graph, d_queries, params);

  // Get stream from the calling framework
  DLDevice device = queries.device();
  cudaStream_t stream = static_cast<cudaStream_t>(
      TVMFFIEnvGetStream(device.device_type, device.device_id));

  // Unpack pairs into separate output tensors
  uint32_t total = n_queries * static_cast<uint32_t>(k);
  uint32_t threads = 256;
  uint32_t blocks = (total + threads - 1) / threads;

  unpack_results_kernel<<<blocks, threads, 0, stream>>>(
      result.frontier,
      static_cast<int32_t*>(out_indices.data_ptr()),
      static_cast<float*>(out_distances.data_ptr()),
      total);

  cudaFree(result.frontier);
}

// ── Export ──────────────────────────────────────────────────────
TVM_FFI_DLL_EXPORT_TYPED_FUNC(jasper_load_graph, jasper_ffi::LoadGraph);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(jasper_save_graph, jasper_ffi::SaveGraph);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(jasper_search,     jasper_ffi::Search);

} // namespace jasper_ffi