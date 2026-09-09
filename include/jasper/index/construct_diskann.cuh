// diskann like construction. Construct multiple small indices with duplicated
// nodes (default by a factor of 2). Then merge their neighborhoods into a large
// vector graph.
//
// This mirrors construct_scale.cuh but follows the DiskANN "merge" build:
//   1. Shard the dataset with polytope (cross-polytope) LSH so that each vector
//      is duplicated into `n_dup` overlapping shards (default 2). Nearby points
//      land in the same shard, so a shard's local Vamana graph captures useful
//      local structure.
//   2. Build one intermediate graph per shard, at the (larger) intermediate
//      degree R_int, then remap its neighbor ids from shard-local to global.
//   3. Merge (distance only): for every vector, union its neighbor lists across
//      all shards it belongs to and keep the R_final closest by the distances
//      already computed during the shard build. No vectors are needed on the
//      device for the merge, so it does not depend on the dataset fitting in GPU
//      memory (unlike the vector-based α-prune merge).
//
// Build and merge are interleaved per shard: a shard is built, its contribution
// is folded into the host-resident final edge lists, then it is freed. Shards
// therefore never accumulate in memory.
//
// The caller chooses the intermediate graph size (R_int), the final graph size
// (R_final), and the number of duplications (n_dup) by picking the INT_CFG /
// FIN_CFG construct configs and passing n_dup.
#pragma once

#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <fstream>
#include <vector>
#include <limits>
#include <algorithm>
#include <stdexcept>
#include <thread>
#include <future>
#include <atomic>
#include <tuple>
#include <deque>
#include <mutex>
#include <condition_variable>
#include <chrono>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "jasper/index/construct.cuh"
#include "jasper/index/construct_diskann_kernels.cuh"
#include "jasper/rotation/rotation.cuh"

namespace jasper {

// ── Directional (LSH/PQ) population options ─────────────────────────────────
// Which estimator artifacts the build should produce, and the sampling budgets
// for training them. Both default off: the per-edge codes roughly double the
// builder's host footprint (see docs/design/directional-diskann-population.md
// section 4), so they are opt-in.
struct diskann_directional_opts {
  bool     build_lsh   = false;
  bool     build_pq    = false;
  uint32_t lsh_samples = 32768;   // edges sampled to calibrate lsh_globals
  uint64_t lsh_seed    = 42;
  uint32_t pq_train    = 40000;   // residuals sampled to train the codebooks
  uint32_t pq_iters    = 12;      // k-means iterations
  uint64_t pq_seed     = 123;

  // Rotate the dataset in place before sharding (prerotate_dataset). Applied
  // only when an estimator is requested — a plain build gains nothing from it.
  // ⚠ The seed is NOT stored in the index; queries must be rotated with the
  // same one or they silently search an unrotated query against a rotated
  // index. See the KNOWN GAP note above DIRECTIONAL_TRAILER_MAGIC in graph.cuh.
  bool     prerotate      = true;
  uint64_t prerotate_seed = 42;

  __host__ bool any() const { return build_lsh || build_pq; }
  __host__ bool rotating() const { return any() && prerotate; }
};

// Host-resident directional payload for the merged graph, plus the estimator
// models shared by every edge.
//
// Split into a specialization so a non-directional config never instantiates
// edge_lsh_list / edge_pq_list / lsh_globals: at k_ranks == 0 those degenerate
// to zero-length arrays, which is why graph_segment only ever names them
// through a pointer. Everything below is guarded by `if constexpr (use_lsh)`
// at the use sites for the same reason.
template <typename FIN_GRAPH_CFG, bool USE_LSH = FIN_GRAPH_CFG::use_lsh>
struct diskann_directional_state {
  static constexpr bool enabled = false;
  bool has_lsh = false;
  bool has_pq  = false;
  __host__ void free_models() {}   // no-op, so callers need no if constexpr
};

template <typename FIN_GRAPH_CFG>
struct diskann_directional_state<FIN_GRAPH_CFG, true> {
  static constexpr bool enabled = true;
  using edge_lsh_t = typename FIN_GRAPH_CFG::edge_lsh_list_t;
  using edge_pq_t  = typename FIN_GRAPH_CFG::edge_pq_list_t;

  // Indexed by global id, parallel to diskann_intermediate_graph::edges.
  // Empty unless the corresponding estimator was requested.
  std::vector<edge_lsh_t> edge_lshs;
  std::vector<edge_pq_t>  edge_pqs;

  // Exact ||v||² per vector, needed by the PQ L2 estimator.
  std::vector<float> norms;

  lsh_globals<FIN_GRAPH_CFG::k_ranks>                    globals{};
  pq_codebooks<FIN_GRAPH_CFG::pq_m, FIN_GRAPH_CFG::pq_k> codebooks{};

  // Set once the arrays are allocated, i.e. "this estimator is being built".
  bool has_lsh = false;
  bool has_pq  = false;

  // Which space the stored vectors ended up in, recorded so the saver can put
  // it in the trailer and readers don't have to be told out of band.
  directional_rotation_info rotation{};

  __host__ void free_models() {
    if (has_pq) codebooks.free();
  }
};

// One shard's directional payload, on host, in shard-local row order —
// the in-flight half of the merge. Same specialization trick as above.
template <typename INT_GRAPH_CFG, bool USE_LSH = INT_GRAPH_CFG::use_lsh>
struct diskann_shard_payload {
  static constexpr bool enabled = false;
};

template <typename INT_GRAPH_CFG>
struct diskann_shard_payload<INT_GRAPH_CFG, true> {
  static constexpr bool enabled = true;
  std::vector<typename INT_GRAPH_CFG::edge_lsh_list_t> lsh;
  std::vector<typename INT_GRAPH_CFG::edge_pq_list_t>  pq;
};

// Builds the DiskANN-style merged graph. INT_CFG drives the per-shard build at
// degree R_int; FIN_CFG defines the final merged graph at degree R_final. Both
// configs must share data_t / index_t / distance_t / dist_func; only the degree
// differs. The result is held as host-resident edge lists (no vectors) and
// written out with save_to_file, which streams vectors from the dataset.
//
// When the configs enable directional storage and diskann_directional_opts asks
// for it, each shard's per-edge LSH/PQ codes are computed while the shard graph
// is still device-resident and its neighbor ids are still shard-local, then
// carried through the merge as payload alongside (id, dist). No step ever needs
// a vector from outside the resident shard.
template <typename INT_CFG, typename FIN_CFG>
struct diskann_intermediate_graph {
  using int_graph_cfg = typename INT_CFG::graph_cfg_t;
  using fin_graph_cfg = typename FIN_CFG::graph_cfg_t;

  using data_t          = typename fin_graph_cfg::data_t;
  using index_t         = typename fin_graph_cfg::index_t;
  using distance_t      = typename fin_graph_cfg::distance_t;
  using int_edge_list_t = typename int_graph_cfg::edge_list_t;
  using fin_edge_list_t = typename fin_graph_cfg::edge_list_t;
  using vector_view_t   = typename fin_graph_cfg::vector_view_t;
  using int_graph_t     = graph<int_graph_cfg>;

  // Named, never instantiated for non-directional configs (pointers/typedefs
  // only — see diskann_directional_state).
  using int_edge_lsh_t = typename int_graph_cfg::edge_lsh_list_t;
  using fin_edge_lsh_t = typename fin_graph_cfg::edge_lsh_list_t;
  using int_edge_pq_t  = typename int_graph_cfg::edge_pq_list_t;
  using fin_edge_pq_t  = typename fin_graph_cfg::edge_pq_list_t;
  using shard_payload_t = diskann_shard_payload<int_graph_cfg>;

  static constexpr uint32_t R_INT   = int_graph_cfg::n_neighbors;
  static constexpr uint32_t R_FIN   = fin_graph_cfg::n_neighbors;
  static constexpr bool     USE_LSH = fin_graph_cfg::use_lsh;

  static_assert(std::is_same<data_t, typename int_graph_cfg::data_t>::value,
                "INT_CFG and FIN_CFG must share data_t");
  static_assert(std::is_same<index_t, typename int_graph_cfg::index_t>::value,
                "INT_CFG and FIN_CFG must share index_t");
  static_assert(std::is_same<distance_t, typename int_graph_cfg::distance_t>::value,
                "INT_CFG and FIN_CFG must share distance_t");
  static_assert(int_graph_cfg::dist_func == fin_graph_cfg::dist_func,
                "INT_CFG and FIN_CFG must share dist_func");

  // The merge moves per-edge codes from an INT row to a FIN row, so the two
  // must agree on everything except the degree: same rank/subquantizer count,
  // same packed word width, same codebook shape.
  static_assert(int_graph_cfg::use_lsh == fin_graph_cfg::use_lsh,
                "INT_CFG and FIN_CFG must share use_lsh");
  static_assert(int_graph_cfg::k_ranks == fin_graph_cfg::k_ranks,
                "INT_CFG and FIN_CFG must share k_ranks");
  static_assert(std::is_same<typename int_graph_cfg::packed_t,
                             typename fin_graph_cfg::packed_t>::value,
                "INT_CFG and FIN_CFG must share packed_t");
  static_assert(int_graph_cfg::pq_m == fin_graph_cfg::pq_m,
                "INT_CFG and FIN_CFG must share pq_m");
  static_assert(int_graph_cfg::pq_k == fin_graph_cfg::pq_k,
                "INT_CFG and FIN_CFG must share pq_k");

  // Final graph, host-resident, vectors excluded. Indexed by global id.
  std::vector<fin_edge_list_t> edges;
  std::vector<uint8_t>         counts;
  index_t  n_vectors = 0;
  uint32_t dim       = 0;
  uint32_t n_parts   = 0;
  uint32_t n_dup     = 0;
  float    alpha     = 1.2f;

  // Per-edge LSH/PQ codes + the trained models. Empty for a plain build.
  diskann_directional_state<fin_graph_cfg> dir;

  // ── Global prerotation, streamed ──────────────────────────────────────────
  // Replace every vector v of `data` with Pᵀv, for one random orthogonal P
  // seeded by `seed`. In place, so the caller's buffer — the same one
  // save_to_file streams from — ends up holding the rotated vectors, and the
  // index is an index of the rotated space. Queries must be rotated with the
  // same seed.
  //
  // Why the directional path needs this: the cross-polytope estimator scores an
  // edge from its top-k_ranks residual coordinates, and its per-rank
  // calibration (lsh_globals) assumes those coordinates are interchangeable.
  // On a raw, anisotropic dataset a few dimensions dominate every residual, so
  // the top-k is nearly the same coordinate set for every edge and carries far
  // less information. This is what construct_graph's `prerotate` does for
  // create_lsh_index (construct.cuh) — but that implementation stages the whole
  // dataset on the device at once, which is exactly what this builder exists to
  // avoid, hence the chunked version here.
  //
  // Rotation is orthogonal, so L2 and inner products are preserved and the
  // graph that gets built is the same one (up to floating-point noise). It is
  // applied ONCE, before sharding, so every shard shares one space — the thing
  // per-shard rotation could never give us.
  static void prerotate_dataset(
      vector_view_t data,
      uint64_t seed,
      size_t chunk_bytes = size_t{512} * 1024 * 1024
  ) {
    static_assert(std::is_same<data_t, __half>::value,
                  "[construct_diskann] prerotate requires data_t == __half");
    const uint32_t dim        = data.dim;
    const uint32_t padded_dim = vector_view_t::pad(dim);
    const uint32_t n_vectors  = data.n_vectors;
    if (n_vectors == 0 || dim == 0) return;

    auto check = [](cudaError_t err, const char* what) {
      if (err != cudaSuccess)
        throw std::runtime_error(std::string("[construct_diskann] ") + what + ": "
                                 + cudaGetErrorString(err));
    };

    // P as float (QR needs float), then cast to __half for the GEMM.
    std::vector<float> h_P_f(static_cast<size_t>(dim) * dim);
    std::vector<float> h_Pt_f(static_cast<size_t>(dim) * dim);
    set_rotation_matrix<float>(static_cast<int>(dim), h_P_f.data(), h_Pt_f.data(), seed);
    std::vector<__half> h_P(h_P_f.size());
    for (size_t i = 0; i < h_P.size(); ++i) h_P[i] = static_cast<__half>(h_P_f[i]);

    __half* d_P = nullptr;
    check(cudaMalloc(&d_P, h_P.size() * sizeof(__half)), "cudaMalloc d_P (prerotate)");
    check(cudaMemcpy(d_P, h_P.data(), h_P.size() * sizeof(__half),
                     cudaMemcpyHostToDevice), "cudaMemcpy d_P (prerotate)");

    const size_t row_bytes = static_cast<size_t>(padded_dim) * sizeof(data_t) * 2;
    uint32_t chunk = static_cast<uint32_t>(std::max<size_t>(1, chunk_bytes / row_bytes));
    chunk = std::min(chunk, n_vectors);

    data_t* d_in  = nullptr;
    data_t* d_out = nullptr;
    const size_t chunk_bytes_alloc =
        static_cast<size_t>(chunk) * padded_dim * sizeof(data_t);
    check(cudaMalloc(&d_in,  chunk_bytes_alloc), "cudaMalloc d_in (prerotate)");
    check(cudaMalloc(&d_out, chunk_bytes_alloc), "cudaMalloc d_out (prerotate)");
    // The GEMM writes only rows [0, dim) of each column (ldc = padded_dim), so
    // zero once here and the row pad lanes stay zero for every chunk.
    check(cudaMemset(d_out, 0, chunk_bytes_alloc), "cudaMemset d_out (prerotate)");

    cublasHandle_t handle;
    cublasCreate(&handle);
    const float one = 1.0f, zero = 0.0f;

    const cudaMemcpyKind to_dev =
        data.on_host ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice;
    const cudaMemcpyKind from_dev =
        data.on_host ? cudaMemcpyDeviceToHost : cudaMemcpyDeviceToDevice;

    for (uint32_t base = 0; base < n_vectors; base += chunk) {
      const uint32_t cnt = std::min(chunk, n_vectors - base);
      data_t* rows = data.data + static_cast<size_t>(base) * padded_dim;
      const size_t bytes = static_cast<size_t>(cnt) * padded_dim * sizeof(data_t);

      check(cudaMemcpy(d_in, rows, bytes, to_dev), "cudaMemcpy chunk in (prerotate)");

      cublasStatus_t stat = cublasGemmEx(
          handle, CUBLAS_OP_T, CUBLAS_OP_N,
          dim, cnt, dim,
          &one,  d_P,  CUDA_R_16F, dim,
          d_in,        CUDA_R_16F, padded_dim,
          &zero, d_out, CUDA_R_16F, padded_dim,
          CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
      if (stat != CUBLAS_STATUS_SUCCESS) {
        cublasDestroy(handle);
        cudaFree(d_in); cudaFree(d_out); cudaFree(d_P);
        throw std::runtime_error("[construct_diskann] cublasGemmEx (prerotate) failed: "
                                 + std::to_string(stat));
      }

      check(cudaMemcpy(rows, d_out, bytes, from_dev),
            "cudaMemcpy chunk out (prerotate)");

      std::printf("\r[construct_diskann] prerotate %u / %u (%.1f%%)",
                  base + cnt, n_vectors, 100.0f * (base + cnt) / n_vectors);
      std::fflush(stdout);
    }
    std::printf("\n");

    cublasDestroy(handle);
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_P);
  }

  // ── Step 1: polytope-LSH shard assignment ─────────────────────────────────
  // Returns, per shard, the list of global ids assigned to it. Each vector is
  // placed in up to `n_dup` shards (fewer if its top coordinates collide onto
  // the same shard). Vectors are streamed to the device in chunks so this stays
  // within `chunk_bytes` regardless of dataset size.
  //
  // A random orthogonal rotation is applied to a *temporary* copy of each chunk
  // before hashing. Cross-polytope LSH is only balanced in an isotropic space:
  // raw datasets are anisotropic (a few dimensions dominate the magnitude) so
  // the argmax collapses onto a handful of coordinates, and non-negative data
  // (e.g. uint8) has a constant sign bit, which with an even n_parts leaves half
  // the shards empty. The rotation mixes coordinates and produces both signs,
  // restoring balance. It is orthogonal, so it preserves L2 distances (and thus
  // locality) and is never applied to the stored/built vectors — the final index
  // stays in the original space and queries need no rotation.
  //
  // `already_rotated` skips that hashing rotation: when the dataset has been
  // prerotated in place (prerotate_dataset above) it is already isotropic, and
  // a second random rotation would only cost a GEMM per chunk. Mean-centering
  // still applies — it removes the data's DC component, which prerotation does
  // not touch.
  static std::vector<std::vector<index_t>> assign_shards(
      vector_view_t data,
      uint32_t n_parts,
      uint32_t n_dup,
      uint64_t seed,
      size_t chunk_bytes = size_t{512} * 1024 * 1024,
      bool already_rotated = false
  ) {
    static_assert(std::is_same<data_t, __half>::value,
                  "[construct_diskann] polytope LSH sharding requires data_t == __half");
    constexpr index_t INVALID = std::numeric_limits<index_t>::max();

    const uint32_t dim        = data.dim;
    const uint32_t padded_dim = vector_view_t::pad(dim);
    const uint32_t n_vectors  = data.n_vectors;

    if (padded_dim > 65536u) {
      throw std::runtime_error(
          "[construct_diskann] polytope LSH requires padded_dim <= 65536");
    }

    const size_t row_bytes = static_cast<size_t>(padded_dim) * sizeof(data_t) * 2  // chunk + rotated
                           + static_cast<size_t>(n_dup) * sizeof(index_t);
    uint32_t chunk = static_cast<uint32_t>(std::max<size_t>(1, chunk_bytes / row_bytes));
    chunk = std::min(chunk, n_vectors);

    auto check = [](cudaError_t err, const char* what) {
      if (err != cudaSuccess)
        throw std::runtime_error(std::string("[construct_diskann] ") + what + ": "
                                 + cudaGetErrorString(err));
    };

    // Mean vector (sampled) in the original space, used to center each chunk
    // before rotation so cross-polytope LSH is not dominated by the data's DC
    // component. Sampled over up to ~2M vectors for speed.
    std::vector<__half> h_mean(dim);
    {
      const uint32_t target = std::min<uint32_t>(n_vectors, 2000000u);
      const uint32_t stride = std::max<uint32_t>(1u, n_vectors / target);
      std::vector<double> acc(dim, 0.0);
      uint32_t sampled = 0;
      for (uint32_t i = 0; i < n_vectors; i += stride) {
        const data_t* vp = data.data + static_cast<size_t>(i) * padded_dim;
        for (uint32_t d = 0; d < dim; d++) acc[d] += __half2float(vp[d]);
        sampled++;
      }
      for (uint32_t d = 0; d < dim; d++)
        h_mean[d] = static_cast<__half>(acc[d] / std::max<uint32_t>(1u, sampled));
    }
    __half* d_mean = nullptr;
    auto check_mean = [](cudaError_t err, const char* what) {
      if (err != cudaSuccess)
        throw std::runtime_error(std::string("[construct_diskann] ") + what + ": "
                                 + cudaGetErrorString(err));
    };
    check_mean(cudaMalloc(&d_mean, static_cast<size_t>(dim) * sizeof(__half)), "cudaMalloc d_mean");
    check_mean(cudaMemcpy(d_mean, h_mean.data(), static_cast<size_t>(dim) * sizeof(__half),
                          cudaMemcpyHostToDevice), "cudaMemcpy d_mean");

    // Random orthogonal rotation matrix (dim x dim), generated once. Skipped
    // when the dataset is already in a rotated space.
    __half* d_P = nullptr;
    if (!already_rotated) {
      std::vector<float> h_P_f(static_cast<size_t>(dim) * dim);
      std::vector<float> h_Pt_f(static_cast<size_t>(dim) * dim);
      set_rotation_matrix<float>(static_cast<int>(dim), h_P_f.data(), h_Pt_f.data(), seed);
      std::vector<__half> h_P(h_P_f.size());
      for (size_t i = 0; i < h_P.size(); ++i) h_P[i] = static_cast<__half>(h_P_f[i]);
      check(cudaMalloc(&d_P, h_P.size() * sizeof(__half)), "cudaMalloc d_P");
      check(cudaMemcpy(d_P, h_P.data(), h_P.size() * sizeof(__half), cudaMemcpyHostToDevice),
            "cudaMemcpy d_P");
    }

    data_t*  d_chunk    = nullptr;
    data_t*  d_rot      = nullptr;
    index_t* d_shard_of = nullptr;
    check(cudaMalloc(&d_chunk, static_cast<size_t>(chunk) * padded_dim * sizeof(data_t)),
          "cudaMalloc d_chunk");
    if (!already_rotated)
      check(cudaMalloc(&d_rot, static_cast<size_t>(chunk) * padded_dim * sizeof(data_t)),
            "cudaMalloc d_rot");
    check(cudaMalloc(&d_shard_of, static_cast<size_t>(chunk) * n_dup * sizeof(index_t)),
          "cudaMalloc d_shard_of");

    cublasHandle_t handle;
    cublasCreate(&handle);
    const float one = 1.0f, zero = 0.0f;

    // Shared memory: packed keys[padded_dim] + warp_scratch[nwarps].
    constexpr uint32_t block = 128;
    const uint32_t nwarps = block / 32;
    const size_t smem = static_cast<size_t>(padded_dim) * sizeof(uint32_t)
                      + nwarps * sizeof(uint32_t);

    std::vector<std::vector<index_t>> shard_ids(n_parts);
    std::vector<index_t> h_shard_of(static_cast<size_t>(chunk) * n_dup);

    const cudaMemcpyKind kind =
        data.on_host ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice;

    for (uint32_t base = 0; base < n_vectors; base += chunk) {
      uint32_t cnt = std::min(chunk, n_vectors - base);

      cudaMemcpy(d_chunk,
                 data.data + static_cast<size_t>(base) * padded_dim,
                 static_cast<size_t>(cnt) * padded_dim * sizeof(data_t),
                 kind);

      // Center the chunk (v -= mean) before rotating.
      {
        const size_t total = static_cast<size_t>(cnt) * dim;
        const uint32_t sblock = 256;
        const uint32_t sgrid = static_cast<uint32_t>(
            std::min<size_t>((total + sblock - 1) / sblock, 65535u));
        subtract_mean_kernel<data_t><<<sgrid, sblock>>>(d_chunk, d_mean, cnt, dim, padded_dim);
        check(cudaGetLastError(), "subtract_mean_kernel launch");
      }

      // Rotate the chunk into d_rot: d_rot = Pᵀ · d_chunk (per column/vector).
      // Leading dims are padded_dim; only the first `dim` output rows are
      // written, matching the kernel which reads coordinates [0, dim).
      const data_t* hash_src = d_chunk;
      if (!already_rotated) {
        cublasStatus_t stat = cublasGemmEx(
            handle, CUBLAS_OP_T, CUBLAS_OP_N,
            dim, cnt, dim,
            &one,  d_P,     CUDA_R_16F, dim,
            d_chunk,        CUDA_R_16F, padded_dim,
            &zero, d_rot,   CUDA_R_16F, padded_dim,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        if (stat != CUBLAS_STATUS_SUCCESS)
          throw std::runtime_error("[construct_diskann] cublasGemmEx (rotate) failed: "
                                   + std::to_string(stat));
        hash_src = d_rot;
      }

      polytope_lsh_assign_kernel<data_t, index_t><<<cnt, block, smem>>>(
          hash_src, cnt, dim, padded_dim, n_parts, n_dup, d_shard_of);
      check(cudaGetLastError(), "polytope_lsh_assign_kernel launch");
      check(cudaDeviceSynchronize(), "polytope_lsh_assign_kernel sync");

      cudaMemcpy(h_shard_of.data(), d_shard_of,
                 static_cast<size_t>(cnt) * n_dup * sizeof(index_t),
                 cudaMemcpyDeviceToHost);

      for (uint32_t i = 0; i < cnt; i++) {
        const index_t gid = static_cast<index_t>(base + i);
        for (uint32_t r = 0; r < n_dup; r++) {
          index_t s = h_shard_of[static_cast<size_t>(i) * n_dup + r];
          if (s != INVALID) shard_ids[s].push_back(gid);
        }
      }

      std::printf("\r[construct_diskann] LSH assign %u / %u (%.1f%%)",
                  base + cnt, n_vectors, 100.0f * (base + cnt) / n_vectors);
      std::fflush(stdout);
    }
    std::printf("\n");

    cublasDestroy(handle);
    cudaFree(d_mean);
    cudaFree(d_P);
    cudaFree(d_chunk);
    cudaFree(d_rot);
    cudaFree(d_shard_of);
    return shard_ids;
  }

  // ── Exact per-vector norms, streamed through the device ────────────────────
  // Returns ||v||² for every vector in `data`, in global id order. Chunked, so
  // it stays within `chunk_bytes` of device memory regardless of dataset size —
  // the same reason assign_shards streams. Only needed when PQ is built.
  static std::vector<float> compute_norms(
      vector_view_t data,
      size_t chunk_bytes = size_t{512} * 1024 * 1024
  ) {
    const uint32_t padded_dim = vector_view_t::pad(data.dim);
    const uint32_t n_vectors  = data.n_vectors;

    std::vector<float> norms(static_cast<size_t>(n_vectors));
    if (n_vectors == 0) return norms;

    auto check = [](cudaError_t err, const char* what) {
      if (err != cudaSuccess)
        throw std::runtime_error(std::string("[construct_diskann] ") + what + ": "
                                 + cudaGetErrorString(err));
    };

    const size_t row_bytes =
        static_cast<size_t>(padded_dim) * sizeof(data_t) + sizeof(float);
    uint32_t chunk = static_cast<uint32_t>(std::max<size_t>(1, chunk_bytes / row_bytes));
    chunk = std::min(chunk, n_vectors);

    data_t* d_chunk = nullptr;
    float*  d_norms = nullptr;
    check(cudaMalloc(&d_chunk, static_cast<size_t>(chunk) * padded_dim * sizeof(data_t)),
          "cudaMalloc d_chunk (norms)");
    check(cudaMalloc(&d_norms, static_cast<size_t>(chunk) * sizeof(float)),
          "cudaMalloc d_norms");

    const cudaMemcpyKind kind =
        data.on_host ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice;

    for (uint32_t base = 0; base < n_vectors; base += chunk) {
      const uint32_t cnt = std::min(chunk, n_vectors - base);
      check(cudaMemcpy(d_chunk,
                       data.data + static_cast<size_t>(base) * padded_dim,
                       static_cast<size_t>(cnt) * padded_dim * sizeof(data_t), kind),
            "cudaMemcpy chunk (norms)");
      vector_norms_kernel<data_t><<<cnt, 128>>>(d_chunk, cnt, padded_dim, d_norms);
      check(cudaGetLastError(), "vector_norms_kernel launch");
      check(cudaMemcpy(norms.data() + base, d_norms,
                       static_cast<size_t>(cnt) * sizeof(float),
                       cudaMemcpyDeviceToHost),
            "cudaMemcpy norms out");
    }

    cudaFree(d_chunk);
    cudaFree(d_norms);
    return norms;
  }

  // Pointers into one node's payload rows for a single merge_row call. Null
  // when the build isn't directional, or when that estimator wasn't requested.
  struct row_payload_t {
    fin_edge_lsh_t*       out_lsh = nullptr;
    const int_edge_lsh_t* in_lsh  = nullptr;
    fin_edge_pq_t*        out_pq  = nullptr;
    const int_edge_pq_t*  in_pq   = nullptr;
  };

  // Copy one edge slot's LSH payload between rows of different degree (INT_CFG
  // and FIN_CFG share k_ranks and packed_t, so only the row stride differs).
  template <typename DST_T, typename SRC_T>
  static inline void copy_lsh_edge(DST_T& dst, uint8_t dst_e,
                                   const SRC_T& src, uint8_t src_e) {
    constexpr uint32_t K = fin_graph_cfg::k_ranks;
    for (uint32_t r = 0; r < K; r++)
      dst.packed[static_cast<uint32_t>(dst_e) * K + r] =
          src.packed[static_cast<uint32_t>(src_e) * K + r];
    dst.mag_sq[dst_e] = src.mag_sq[src_e];
  }

  // Same for PQ. get_code/set_code apply each row's own subspace-major stride,
  // so this is stride-correct across the R_INT -> R_FIN change.
  template <typename DST_T, typename SRC_T>
  static inline void copy_pq_edge(DST_T& dst, uint8_t dst_e,
                                  const SRC_T& src, uint8_t src_e) {
    constexpr uint32_t M = fin_graph_cfg::pq_m;
    for (uint32_t j = 0; j < M; j++)
      dst.set_code(dst_e, j, src.get_code(src_e, j));
  }

  // Distance-only merge of one shard neighbor row into the final row for a node.
  // Unions the existing final neighbors with the shard's neighbors and keeps the
  // R_final smallest by stored distance, deduplicating by neighbor id (the first
  // occurrence in sorted order carries the smallest distance). No vector lookups.
  //
  // When `pay` is populated, each surviving edge's directional codes move with
  // it. The codes are a function of (x_u, x_nb) alone, so a code computed inside
  // a shard is bit-identical to what a whole-graph populate would produce for
  // that pair — the merge only permutes and truncates.
  static inline void merge_row(fin_edge_list_t& out, uint8_t& out_count,
                               const int_edge_list_t& in, uint8_t in_count,
                               const row_payload_t& pay = row_payload_t{}) {
    constexpr index_t INVALID = std::numeric_limits<index_t>::max();
    constexpr uint32_t MAXC = R_FIN + R_INT;

    index_t cid[MAXC];
    float   cd[MAXC];
    // Where each candidate came from, so its payload can follow it: 0 = the
    // existing `out` row, 1 = the incoming shard row; cslot is the source slot.
    uint8_t csrc[MAXC];
    uint8_t cslot[MAXC];
    uint32_t nc = 0;

    for (uint8_t j = 0; j < out_count && nc < MAXC; j++) {
      index_t id = out.edges[j];
      if (id == INVALID) continue;
      cid[nc]   = id;
      cd[nc]    = __bfloat162float(out.dist[j]);
      csrc[nc]  = 0;
      cslot[nc] = j;
      nc++;
    }
    for (uint8_t j = 0; j < in_count && nc < MAXC; j++) {
      index_t id = in.edges[j];
      if (id == INVALID) continue;
      cid[nc]   = id;
      cd[nc]    = __bfloat162float(in.dist[j]);
      csrc[nc]  = 1;
      cslot[nc] = j;
      nc++;
    }

    // Insertion sort by distance ascending (nc is small, <= R_FIN + R_INT).
    for (uint32_t a = 1; a < nc; a++) {
      index_t ki = cid[a];
      float   kd = cd[a];
      uint8_t ks = csrc[a];
      uint8_t kl = cslot[a];
      int32_t b  = static_cast<int32_t>(a) - 1;
      while (b >= 0 && cd[b] > kd) {
        cid[b + 1]   = cid[b];
        cd[b + 1]    = cd[b];
        csrc[b + 1]  = csrc[b];
        cslot[b + 1] = cslot[b];
        b--;
      }
      cid[b + 1]   = ki;
      cd[b + 1]    = kd;
      csrc[b + 1]  = ks;
      cslot[b + 1] = kl;
    }

    // Select up to R_FIN unique ids in ascending-distance order.
    uint8_t sel = 0;
    uint8_t sel_src[R_FIN];
    uint8_t sel_slot[R_FIN];
    for (uint32_t k = 0; k < nc && sel < R_FIN; k++) {
      index_t id = cid[k];
      bool dup = false;
      for (uint8_t t = 0; t < sel; t++) {
        if (out.edges[t] == id) { dup = true; break; }
      }
      if (dup) continue;
      out.edges[sel]  = id;
      out.dist[sel]   = __float2bfloat16(cd[k]);
      sel_src[sel]    = csrc[k];
      sel_slot[sel]   = cslot[k];
      sel++;
    }
    out_count = sel;

    // Move the payloads. Unlike the ids/dists above — which were safe to write
    // in place because they had already been copied into cid[]/cd[] — the `out`
    // payload row is simultaneously source and destination, and the sorted order
    // can place an incoming candidate at slot j while an `out` candidate still
    // needs slot j's old contents. So snapshot the destination row first.
    //
    // Slots [sel, R_FIN) need no clearing: the arrays start value-initialized
    // and sel never shrinks (every surviving `out` id is also a candidate, so
    // sel >= out_count on every call).
    if constexpr (USE_LSH) {
      if (pay.out_lsh != nullptr && pay.in_lsh != nullptr) {
        const fin_edge_lsh_t prev = *pay.out_lsh;
        for (uint8_t s = 0; s < sel; s++) {
          if (sel_src[s] == 0) copy_lsh_edge(*pay.out_lsh, s, prev, sel_slot[s]);
          else                 copy_lsh_edge(*pay.out_lsh, s, *pay.in_lsh, sel_slot[s]);
        }
      }
      if (pay.out_pq != nullptr && pay.in_pq != nullptr) {
        const fin_edge_pq_t prev = *pay.out_pq;
        for (uint8_t s = 0; s < sel; s++) {
          if (sel_src[s] == 0) copy_pq_edge(*pay.out_pq, s, prev, sel_slot[s]);
          else                 copy_pq_edge(*pay.out_pq, s, *pay.in_pq, sel_slot[s]);
        }
      }
    }
  }

  // ── Full build: assign shards, then build + merge each shard in turn ────────
  static diskann_intermediate_graph build(
      graph_construct_params<INT_CFG> params,
      uint32_t n_parts,
      uint32_t n_dup,
      size_t   workspace_budget,
      uint64_t seed = 42,
      diskann_directional_opts opts = diskann_directional_opts{}
  ) {
    if (n_parts == 0 || n_dup == 0) {
      throw std::runtime_error("[construct_diskann] n_parts and n_dup must be > 0");
    }

    vector_view_t data = params.data_vectors;
    const uint32_t dim        = data.dim;
    const uint32_t padded_dim = vector_view_t::pad(dim);
    const index_t  N          = static_cast<index_t>(data.n_vectors);

    // Reject an impossible directional request up front, before the (long)
    // shard-assignment pass.
    if constexpr (!USE_LSH) {
      if (opts.any())
        throw std::runtime_error(
            "[construct_diskann] build_lsh/build_pq require configs with "
            "use_lsh=true (graph_config's USE_LSH/K_RANKS template args)");
    } else {
      if (opts.build_pq && padded_dim % fin_graph_cfg::pq_m != 0)
        throw std::runtime_error(
            "[construct_diskann] build_pq requires padded_dim ("
            + std::to_string(padded_dim) + ") divisible by pq_m="
            + std::to_string(fin_graph_cfg::pq_m));
    }

    diskann_intermediate_graph ig;
    ig.n_vectors = N;
    ig.dim       = dim;
    ig.n_parts   = n_parts;
    ig.n_dup     = n_dup;
    ig.alpha     = params.alpha;

    using clk = std::chrono::steady_clock;
    auto secs = [](clk::time_point a, clk::time_point b) {
      return std::chrono::duration<double>(b - a).count();
    };

    // Prerotate the whole dataset ONCE, before sharding, so every shard, the
    // stored vectors and the codes all live in the same rotated space. In place,
    // so `data` (which the caller also hands to save_to_file) now holds the
    // rotated vectors — the index is an index of the rotated space, and queries
    // must be rotated with opts.prerotate_seed.
    double t_rotate = 0;
    if constexpr (USE_LSH) {
      if (opts.any()) {
        // Recorded either way: "known and not rotated" is a definitive
        // instruction to the reader, not the same as saying nothing.
        ig.dir.rotation.known     = true;
        ig.dir.rotation.prerotate = opts.rotating();
        ig.dir.rotation.seed      = opts.rotating() ? opts.prerotate_seed : 0;
      }
    }
    if (opts.rotating()) {
      std::cout << "[construct_diskann] prerotating " << N
                << " vectors in place (seed=" << opts.prerotate_seed << ")\n";
      const auto tr0 = clk::now();
      prerotate_dataset(data, opts.prerotate_seed);
      t_rotate = secs(tr0, clk::now());
    }

    std::cout << "[construct_diskann] assigning " << N
              << " vectors to " << n_parts << " shards (n_dup=" << n_dup << ")\n";
    auto t_assign0 = clk::now();
    auto shard_ids = assign_shards(data, n_parts, n_dup, seed,
                                   /*chunk_bytes=*/size_t{512} * 1024 * 1024,
                                   /*already_rotated=*/opts.rotating());
    const double t_assign = secs(t_assign0, clk::now());

    // Host-resident final edge lists (no vectors), zero-initialized, plus the
    // directional payload rows when requested. The payload arrays are the bulk
    // of this build's host memory — see the section-4 tables in
    // docs/design/directional-diskann-population.md before raising k_ranks.
    std::cout << "[construct_diskann] allocating final edge lists on host\n";
    auto t_alloc0 = clk::now();
    ig.edges.assign(static_cast<size_t>(N), fin_edge_list_t{});
    ig.counts.assign(static_cast<size_t>(N), 0);
    if constexpr (USE_LSH) {
      size_t extra = 0;
      if (opts.build_lsh) {
        ig.dir.edge_lshs.assign(static_cast<size_t>(N), fin_edge_lsh_t{});
        ig.dir.has_lsh = true;
        extra += static_cast<size_t>(N) * sizeof(fin_edge_lsh_t);
      }
      if (opts.build_pq) {
        ig.dir.edge_pqs.assign(static_cast<size_t>(N), fin_edge_pq_t{});
        ig.dir.has_pq = true;
        extra += static_cast<size_t>(N) * sizeof(fin_edge_pq_t);
      }
      if (extra > 0)
        std::printf("[construct_diskann] directional payload on host: %.1f GB "
                    "(lsh=%d %zuB/node, pq=%d %zuB/node)\n",
                    extra / (1024.0 * 1024.0 * 1024.0),
                    static_cast<int>(opts.build_lsh),
                    opts.build_lsh ? sizeof(fin_edge_lsh_t) : size_t{0},
                    static_cast<int>(opts.build_pq),
                    opts.build_pq ? sizeof(fin_edge_pq_t) : size_t{0});
    }
    const double t_alloc = secs(t_alloc0, clk::now());

    // Per-phase wall-clock accumulators (see the summary printed at the end).
    // gather runs on a background thread, so it is timed in atomic microseconds;
    // the rest run on the main thread.
    std::atomic<long long> gather_us{0};
    double t_construct = 0, t_post = 0, t_merge_wait = 0;
    double t_train = 0, t_populate = 0;

    uint32_t batch_cap =
        graph_construct_workspace<INT_CFG>::max_batch_size_for_budget(workspace_budget);
    batch_cap = std::min(batch_cap, params.max_batch_size);

    auto check = [](cudaError_t err, const char* what) {
      if (err != cudaSuccess)
        throw std::runtime_error(std::string("[construct_diskann] ") + what + ": "
                                 + cudaGetErrorString(err));
    };

    // Largest shard we can build on the device: the shard graph needs per_vec
    // bytes per slot, and we must leave room for the construction workspace
    // (~workspace_budget) plus slack. Shards exceeding this are split into
    // sub-shards so an imbalanced LSH partition can never OOM the device.
    //
    // per_vec must cover everything graph_segment::allocate actually
    // reserves — including slot_to_id and the deletion bit, which this formula
    // used to omit — plus the directional payload buffers. Those are allocated
    // lazily by the first populate_edge_* call, i.e. *after* the shard size has
    // been committed, so leaving them out here means a build that OOMs
    // mid-flight rather than picking a smaller shard.
    size_t per_vec = static_cast<size_t>(padded_dim) * sizeof(data_t)
                   + sizeof(int_edge_list_t)
                   + sizeof(uint8_t)          // edge_counts
                   + sizeof(index_t)          // slot_to_id
                   + 1;                       // 1 deletion bit, rounded up
    if constexpr (USE_LSH) {
      if (opts.build_lsh) per_vec += sizeof(int_edge_lsh_t);
      if (opts.build_pq)  per_vec += sizeof(int_edge_pq_t);
    }
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    const size_t reserve = workspace_budget + (size_t{4} << 30);  // 4 GB slack
    const size_t avail   = (free_b > reserve) ? (free_b - reserve) : (free_b / 2);

    // Budget in whole segments, not vectors: graph::allocate rounds the slot
    // count up to a multiple of vectors_per_segment and allocates every slot in
    // each segment, so sizing by exact vector count under-counts by up to one
    // full segment (1<<20 slots).
    constexpr size_t VPS     = int_graph_t::vectors_per_segment;
    const size_t     per_seg = per_vec * VPS;
    const size_t     n_seg   = avail / per_seg;
    if (n_seg == 0)
      throw std::runtime_error(
          "[construct_diskann] not enough free device memory for even one "
          + std::to_string(VPS) + "-slot shard segment (" +
          std::to_string(per_seg >> 20) + " MB needed, " +
          std::to_string(avail >> 20) + " MB available after the " +
          std::to_string(workspace_budget >> 20) + " MB workspace budget + 4 GB "
          "slack) — lower the workspace budget, or turn off the directional "
          "artifacts");
    const uint32_t max_shard = static_cast<uint32_t>(
        std::min<size_t>(n_seg * VPS, 0xFFFFFFFFull));
    std::printf("[construct_diskann] max shard build size = %u vectors "
                "(%zu segments x %zu B/slot, device free %.1f GB)\n",
                max_shard, n_seg, per_vec, free_b / (1024.0 * 1024.0 * 1024.0));

    // Allocate the shard graph and workspace ONCE, sized for the largest shard,
    // and reuse them for every (sub-)shard via construct_graph_into. This avoids
    // reallocating multi-GB graph/workspace buffers on each shard.
    int_graph_t g = int_graph_t::allocate(dim, max_shard, /*on_host=*/false);
    auto ws = graph_construct_workspace<INT_CFG>::allocate(batch_cap);

    // Enumerate the build tasks: split each oversized shard into equal sub-shards
    // that fit `max_shard`. Sorting largest-first means the one merge that cannot
    // overlap anything — the final one after the loop — lands on the smallest task.
    struct task_t { uint32_t part; size_t begin; size_t len; };
    std::vector<task_t> tasks;
    for (uint32_t p = 0; p < n_parts; p++) {
      const size_t total = shard_ids[p].size();
      if (total == 0) continue;
      const size_t n_sub    = (total + max_shard - 1) / max_shard;
      const size_t sub_size = (total + n_sub - 1) / n_sub;
      for (size_t s = 0; s < n_sub; s++) {
        const size_t begin = s * sub_size;
        const size_t len   = std::min(sub_size, total - begin);
        if (len == 0) break;
        tasks.push_back({p, begin, len});
      }
    }
    std::sort(tasks.begin(), tasks.end(),
              [](const task_t& a, const task_t& b) { return a.len > b.len; });

    // The prefetch gather and the background merge both open OpenMP teams while
    // the GPU builds; leaving both uncapped oversubscribes the cores and starves
    // the merge, which is on the critical path (merge_wait). Gather is a
    // bandwidth-bound copy that a handful of threads already saturate, and it has
    // slack (it hides behind the much longer construct), so cap it and leave the
    // rest of the cores to the merge.
#ifdef _OPENMP
    const int omp_total = omp_get_max_threads();
#else
    const int omp_total = 1;
#endif
    const int gather_nthreads = std::max(1, std::min(8, omp_total / 4));

    auto make_ids = [&](const task_t& t) {
      const auto& v = shard_ids[t.part];
      return std::vector<index_t>(v.begin() + t.begin, v.begin() + t.begin + t.len);
    };

    // Gather a task's vectors into a fresh pageable host buffer (returned; the
    // caller frees it). Runs on a background thread so it overlaps the GPU build
    // of the current task. Pageable malloc + an OpenMP copy avoids a multi-GB
    // cudaMallocHost whose page-locking cost would otherwise stall the pipeline.
    auto gather = [&](const std::vector<index_t>& ids) -> data_t* {
      const auto tg0 = clk::now();
      const uint32_t count = static_cast<uint32_t>(ids.size());
      const size_t buf_elems = static_cast<size_t>(count) * padded_dim;
      data_t* buf = static_cast<data_t*>(std::malloc(buf_elems * sizeof(data_t)));
      if (!buf) throw std::runtime_error("[construct_diskann] shard gather malloc failed");
      #pragma omp parallel for schedule(static) num_threads(gather_nthreads)
      for (uint32_t k = 0; k < count; k++) {
        std::memcpy(buf + static_cast<size_t>(k) * padded_dim,
                    data[static_cast<uint32_t>(ids[k])],
                    static_cast<size_t>(padded_dim) * sizeof(data_t));
      }
      gather_us += std::chrono::duration_cast<std::chrono::microseconds>(
                       clk::now() - tg0).count();
      return buf;
    };

    // The estimator models are shared by every edge of the final graph, so they
    // must be trained exactly once and then held fixed: codes from different
    // codebooks are not mutually comparable at search time. Train on the first
    // task, which (tasks being sorted largest-first) is the biggest shard —
    // a shard's Vamana edges are drawn from the same local-neighbour residual
    // distribution as the final graph's, so it is a representative sample.
    bool models_trained = false;
    auto train_models = [&]() {
      if constexpr (USE_LSH) {
        const auto tt0 = clk::now();
        if (opts.build_lsh) {
          std::printf("[construct_diskann] calibrating lsh_globals (%u samples)\n",
                      opts.lsh_samples);
          ig.dir.globals = g.generate_lsh_globals(opts.lsh_samples, opts.lsh_seed);
        }
        if (opts.build_pq) {
          std::printf("[construct_diskann] training PQ codebooks "
                      "(%u residuals, %u iters)\n", opts.pq_train, opts.pq_iters);
          ig.dir.codebooks =
              g.generate_pq_codebooks(opts.pq_train, opts.pq_iters, opts.pq_seed);
        }
        check(cudaDeviceSynchronize(), "train_models sync");
        t_train += secs(tt0, clk::now());
      }
    };

    // Build one task from an already-gathered buffer into the reused graph `g`,
    // returning its edges + counts (+ directional payload) on host. Frees `buf`.
    auto build_from_buf = [&](data_t* buf, const std::vector<index_t>& ids,
                              std::vector<int_edge_list_t>& se,
                              std::vector<uint8_t>& sc,
                              shard_payload_t& pay) {
      const uint32_t count = static_cast<uint32_t>(ids.size());
      const auto tb1 = clk::now();
      vector_view_t shard_vecs(buf, dim, count, /*on_host=*/true);

      graph_construct_params<INT_CFG> sp;
      sp.data_vectors   = shard_vecs;
      sp.alpha          = params.alpha;
      sp.max_batch_size = std::max<uint32_t>(1, std::min(batch_cap, count));
      sp.on_host        = false;
      sp.prerotate      = false;   // shards must share one space; no per-shard rotation

      construct_graph_into<INT_CFG>(g, ws, sp);
      std::free(buf);              // vectors are on device now; drop the gather buffer
      const auto tb2 = clk::now();
      t_construct += secs(tb1, tb2);

      // ⚠ Directional population MUST happen here: after the build, before the
      // remap below. It resolves each edge's neighbor vector via
      // g.get_vector(neighbor), which only works while the stored ids are still
      // shard-local — once remapped to global ids they are outside this shard
      // graph's index range and the reads run off the end of its segments.
      if constexpr (USE_LSH) {
        if (opts.any()) {
          if (!models_trained) { train_models(); models_trained = true; }
          const auto tp0 = clk::now();
          if (opts.build_lsh) g.populate_edge_lsh();
          if (opts.build_pq)  g.populate_edge_pq(ig.dir.codebooks);
          check(cudaGetLastError(), "populate_edge_* launch");
          check(cudaDeviceSynchronize(), "populate_edge_* sync");
          t_populate += secs(tp0, clk::now());
        }
      }
      const auto tb3 = clk::now();   // remap + copy-back are timed from here

      // Remap neighbor ids shard-local -> global on the device.
      index_t* d_idmap = nullptr;
      check(cudaMalloc(&d_idmap, static_cast<size_t>(count) * sizeof(index_t)),
            "cudaMalloc d_idmap (shard)");
      check(cudaMemcpy(d_idmap, ids.data(),
                       static_cast<size_t>(count) * sizeof(index_t),
                       cudaMemcpyHostToDevice), "cudaMemcpy d_idmap (shard)");
      constexpr uint32_t rblock = 256;
      const uint32_t rgrid = std::min<uint32_t>((count + rblock - 1) / rblock, 65535u);
      remap_shard_edges_kernel<int_graph_cfg><<<rgrid, rblock>>>(g.view(), d_idmap, count);
      check(cudaGetLastError(), "remap_shard_edges_kernel launch");
      check(cudaDeviceSynchronize(), "remap_shard_edges_kernel sync");
      cudaFree(d_idmap);

      // Copy this shard's edges + counts (+ payload) back to host. Vectors stay
      // on device; they are overwritten by the next shard's reload().
      se.resize(count);
      sc.resize(count);
      if constexpr (USE_LSH) {
        if (ig.dir.has_lsh) pay.lsh.resize(count);
        if (ig.dir.has_pq)  pay.pq.resize(count);
      }
      using seg_t = graph_segment<int_graph_cfg>;
      std::vector<seg_t> segs(g.segments.begin(), g.segments.end());
      uint32_t off = 0;
      for (uint32_t s = 0; s < g.n_segments; s++) {
        const seg_t& seg = segs[s];
        const uint32_t seg_count = static_cast<uint32_t>(seg.n_vectors);
        if (seg_count == 0) continue;
        cudaMemcpy(se.data() + off, seg.edges,
                   static_cast<size_t>(seg_count) * sizeof(int_edge_list_t),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(sc.data() + off, seg.edge_counts,
                   static_cast<size_t>(seg_count) * sizeof(uint8_t),
                   cudaMemcpyDeviceToHost);
        if constexpr (USE_LSH) {
          if (ig.dir.has_lsh)
            cudaMemcpy(pay.lsh.data() + off, seg.edge_lshs,
                       static_cast<size_t>(seg_count) * sizeof(int_edge_lsh_t),
                       cudaMemcpyDeviceToHost);
          if (ig.dir.has_pq)
            cudaMemcpy(pay.pq.data() + off, seg.edge_pqs,
                       static_cast<size_t>(seg_count) * sizeof(int_edge_pq_t),
                       cudaMemcpyDeviceToHost);
        }
        off += seg_count;
      }
      cudaDeviceSynchronize();
      t_post += secs(tb3, clk::now());
    };

    // Distance-only merges must be serialized (a global id can appear in several
    // shards, so two concurrent merges could write the same final row). Rather
    // than join the previous merge before starting the next — which blocks the
    // main/GPU thread and idles the GPU between shards — run all merges on one
    // persistent worker fed by a bounded queue. The main thread enqueues a
    // finished shard and immediately proceeds to the next GPU build; it blocks
    // only when the queue is full (MERGE_QCAP bounds host memory). Truncating to
    // R_final each step is associative in distance, so merge order is irrelevant.
    struct merge_job_t {
      std::vector<int_edge_list_t> se;
      std::vector<uint8_t>         sc;
      std::vector<index_t>         ids;
      shard_payload_t              pay;
    };
    std::deque<merge_job_t> merge_q;
    std::mutex              merge_mtx;
    std::condition_variable merge_cv_push, merge_cv_pop;
    bool                    merge_done = false;
    constexpr size_t        MERGE_QCAP = 2;  // in-flight shards; lower if host OOM

    std::thread merge_worker([&]() {
      for (;;) {
        merge_job_t job;
        {
          std::unique_lock<std::mutex> lk(merge_mtx);
          merge_cv_pop.wait(lk, [&] { return merge_done || !merge_q.empty(); });
          if (merge_q.empty()) break;  // merge_done and drained
          job = std::move(merge_q.front());
          merge_q.pop_front();
          merge_cv_push.notify_one();
        }
        const uint32_t cnt = static_cast<uint32_t>(job.ids.size());
        #pragma omp parallel for schedule(static)
        for (uint32_t i = 0; i < cnt; i++) {
          const index_t gid = job.ids[i];
          row_payload_t pay{};
          if constexpr (USE_LSH) {
            if (ig.dir.has_lsh) {
              pay.out_lsh = &ig.dir.edge_lshs[gid];
              pay.in_lsh  = &job.pay.lsh[i];
            }
            if (ig.dir.has_pq) {
              pay.out_pq = &ig.dir.edge_pqs[gid];
              pay.in_pq  = &job.pay.pq[i];
            }
          }
          merge_row(ig.edges[gid], ig.counts[gid], job.se[i], job.sc[i], pay);
        }
      }
    });

    auto enqueue_merge = [&](std::vector<int_edge_list_t>&& se,
                             std::vector<uint8_t>&& sc,
                             std::vector<index_t>&& ids,
                             shard_payload_t&& pay) {
      const auto tw0 = clk::now();
      std::unique_lock<std::mutex> lk(merge_mtx);
      merge_cv_push.wait(lk, [&] { return merge_q.size() < MERGE_QCAP; });
      t_merge_wait += secs(tw0, clk::now());
      merge_q.push_back(merge_job_t{std::move(se), std::move(sc), std::move(ids),
                                    std::move(pay)});
      merge_cv_pop.notify_one();
    };

    // Pipeline: prefetch task t+1's gather (background) while task t builds on the
    // GPU and task t-1 merges on the CPU. GPU, gather, and merge run concurrently.
    std::future<data_t*> pending_buf;
    std::vector<index_t>  pending_ids;
    if (!tasks.empty()) {
      pending_ids = make_ids(tasks[0]);
      pending_buf = std::async(std::launch::async, gather, std::cref(pending_ids));
    }

    for (size_t i = 0; i < tasks.size(); i++) {
      std::printf("[construct_diskann] task %zu / %zu (part %u, %zu vectors)\n",
                  i + 1, tasks.size(), tasks[i].part, tasks[i].len);

      data_t* buf = pending_buf.get();          // gather(i) result
      std::vector<index_t> ids = std::move(pending_ids);

      if (i + 1 < tasks.size()) {               // prefetch gather(i+1) during build(i)
        pending_ids = make_ids(tasks[i + 1]);
        pending_buf = std::async(std::launch::async, gather, std::cref(pending_ids));
      }

      std::vector<int_edge_list_t> se;
      std::vector<uint8_t>         sc;
      shard_payload_t              pay;
      build_from_buf(buf, ids, se, sc, pay);
      enqueue_merge(std::move(se), std::move(sc), std::move(ids), std::move(pay));
    }

    // Drain: signal the worker and wait for the last queued merges to finish.
    const auto t_tail0 = clk::now();
    {
      std::unique_lock<std::mutex> lk(merge_mtx);
      merge_done = true;
      merge_cv_pop.notify_all();
    }
    merge_worker.join();
    const double t_merge_tail = secs(t_tail0, clk::now());

    g.deallocate();
    ws.free();

    // Norms come last, once the shard graph and workspace have been released so
    // the chunk buffers have the device to themselves.
    double t_norms = 0;
    if constexpr (USE_LSH) {
      if (opts.build_pq) {
        std::cout << "[construct_diskann] computing exact vector norms\n";
        const auto tn0 = clk::now();
        ig.dir.norms = compute_norms(data);
        t_norms = secs(tn0, clk::now());
      }
    }

    std::printf(
        "[construct_diskann] phase totals (s): rotate=%.1f  assign=%.1f  "
        "alloc_edges=%.1f  gather=%.1f  construct=%.1f  train=%.1f  "
        "populate=%.1f  remap+copy=%.1f  merge_wait=%.1f  merge_tail=%.1f  "
        "norms=%.1f\n",
        t_rotate, t_assign, t_alloc, gather_us.load() / 1e6, t_construct,
        t_train, t_populate, t_post, t_merge_wait, t_merge_tail, t_norms);

    return ig;
  }

  __host__ double avg_degree() const {
    if (n_vectors == 0) return 0.0;
    uint64_t total = 0;
    for (index_t i = 0; i < n_vectors; i++) total += counts[i];
    return static_cast<double>(total) / static_cast<double>(n_vectors);
  }

  // Print the merged per-edge directional codes for a range of nodes. The
  // format mirrors graph::dump_edge_lsh / edge_lsh_list::print, so the codes
  // carried through the merge can be eyeballed against a query-time
  // populate_edge_lsh on the same graph without going through a file.
  __host__ void dump_edge_payload(index_t start = 0, index_t count = 5,
                                  std::ostream& out = std::cout) const {
    if constexpr (!USE_LSH) {
      out << "[dump_edge_payload] use_lsh disabled for this config\n";
    } else {
      if (!dir.has_lsh && !dir.has_pq) {
        out << "[dump_edge_payload] no directional artifacts were built\n";
        return;
      }
      const index_t end = (count == 0) ? n_vectors
                                       : std::min<index_t>(start + count, n_vectors);
      for (index_t i = start; i < end; i++) {
        const uint32_t cnt = counts[i];
        out << "[" << i << "] " << cnt << " neighbors:";
        for (uint32_t j = 0; j < cnt; j++) out << ' ' << edges[i].edges[j];
        out << '\n';
        if (dir.has_lsh) { out << "  lsh:\n"; dir.edge_lshs[i].print(out, cnt); }
        if (dir.has_pq)  { out << "  pq:\n";  dir.edge_pqs[i].print(out, cnt); }
      }
    }
  }

  // Stream the final graph to disk in the standard node-interleaved format:
  //   header: [total_file_size, n_vectors, medoid, bytes_per_node] (u64 each)
  //   node i: [vector: dim*data_t][edge_count: u8][edges: fin_edge_list_t]
  // Vectors are read from `data` (host) so we never hold a second copy.
  __host__ void save_to_file(vector_view_t data, const std::string& output_fname) const {
    const uint32_t padded_dim = vector_view_t::pad(dim);

    std::ofstream outFile(output_fname, std::ios::binary);
    if (!outFile.is_open())
      throw std::runtime_error("Failed to open " + output_fname + " for graph save");

    uint64_t big_n_vectors  = static_cast<uint64_t>(n_vectors);
    uint64_t medoid         = 0;
    uint64_t bytes_per_node = sizeof(data_t) * dim + sizeof(uint8_t) + sizeof(fin_edge_list_t);
    uint64_t total_file_size = 4 * sizeof(uint64_t) + big_n_vectors * bytes_per_node;

    outFile.write(reinterpret_cast<const char*>(&total_file_size), sizeof(uint64_t));
    outFile.write(reinterpret_cast<const char*>(&big_n_vectors),   sizeof(uint64_t));
    outFile.write(reinterpret_cast<const char*>(&medoid),          sizeof(uint64_t));
    outFile.write(reinterpret_cast<const char*>(&bytes_per_node),  sizeof(uint64_t));

    for (index_t i = 0; i < n_vectors; i++) {
      outFile.write(reinterpret_cast<const char*>(data.data + static_cast<size_t>(i) * padded_dim),
                    sizeof(data_t) * dim);
      outFile.write(reinterpret_cast<const char*>(&counts[i]), sizeof(uint8_t));
      outFile.write(reinterpret_cast<const char*>(&edges[i]),  sizeof(fin_edge_list_t));
    }
    outFile.close();

    std::cout << "Saved graph to " << output_fname << "\n"
              << "  n_vectors       : " << big_n_vectors << "\n"
              << "  medoid          : " << medoid << "\n"
              << "  dim             : " << dim << "\n";
  }

  // Save the base section (exactly what save_to_file writes) followed by the
  // directional trailer, in the byte order load_directional_graph_from_file
  // expects — see the format notes above DIRECTIONAL_TRAILER_MAGIC in graph.cuh.
  //
  // The per-block writes there are per graph segment; here the arrays are flat
  // in global id order, and segments are consecutive vectors_per_segment-sized
  // chunks of that same order, so writing the flat array reproduces the same
  // bytes. Chunked only to bound the per-call write size and give progress.
  __host__ void save_directional_to_file(vector_view_t data,
                                         const std::string& output_fname) const {
    if constexpr (!USE_LSH) {
      throw std::runtime_error(
          "[construct_diskann] save_directional_to_file requires a config with "
          "use_lsh=true; use save_to_file for a plain index");
    } else {
      save_to_file(data, output_fname);
      if (n_vectors == 0) return;

      // Same guard as save_directional_graph_to_file: the reader finds the
      // trailer by recomputing where the base section ends, so verify the two
      // agree before appending something it would never see.
      {
        const uint64_t expected =
            directional_trailer_offset<fin_graph_cfg>(output_fname, dim);
        std::ifstream chk(output_fname, std::ios::binary | std::ios::ate);
        const uint64_t actual = static_cast<uint64_t>(chk.tellg());
        if (actual != expected)
          throw std::runtime_error(
              "[construct_diskann] base section is " + std::to_string(actual) +
              " bytes but the header implies " + std::to_string(expected) +
              " — base_section_bytes() is out of sync with save_to_file");
      }

      const bool has_lsh      = dir.has_lsh && !dir.edge_lshs.empty();
      const bool has_pq       = dir.has_pq  && !dir.edge_pqs.empty();
      const bool has_norms    = !dir.norms.empty();
      const bool has_rotation = dir.rotation.known;
      if (!has_lsh && !has_pq && !has_norms && !has_rotation)
        return;   // base-only, nothing to add

      std::ofstream out(output_fname, std::ios::binary | std::ios::app);
      if (!out.is_open())
        throw std::runtime_error("[construct_diskann] cannot reopen " + output_fname);

      out.write(reinterpret_cast<const char*>(&DIRECTIONAL_TRAILER_MAGIC),   sizeof(uint32_t));
      out.write(reinterpret_cast<const char*>(&DIRECTIONAL_TRAILER_VERSION), sizeof(uint32_t));
      const uint8_t flags = (has_lsh ? 1u : 0u) | (has_pq ? 2u : 0u)
                          | (has_norms ? 4u : 0u) | (has_rotation ? 8u : 0u);
      out.write(reinterpret_cast<const char*>(&flags), sizeof(uint8_t));

      // Write `rows` array entries in vectors_per_segment-sized chunks.
      constexpr size_t VPS = fin_graph_cfg::vectors_per_segment;
      auto write_rows = [&](const void* base, size_t elem_size, const char* what) {
        const char* p = static_cast<const char*>(base);
        size_t done = 0;
        while (done < static_cast<size_t>(n_vectors)) {
          const size_t rows = std::min<size_t>(VPS, n_vectors - done);
          out.write(p + done * elem_size,
                    static_cast<std::streamsize>(rows * elem_size));
          if (!out)
            throw std::runtime_error(std::string("[construct_diskann] short write of ")
                                     + what + " to " + output_fname);
          done += rows;
        }
      };

      if (has_lsh) {
        const uint8_t k_ranks_u8     = fin_graph_cfg::k_ranks;
        const uint8_t packed_size_u8 =
            static_cast<uint8_t>(sizeof(typename fin_graph_cfg::packed_t));
        out.write(reinterpret_cast<const char*>(&k_ranks_u8),     sizeof(uint8_t));
        out.write(reinterpret_cast<const char*>(&packed_size_u8), sizeof(uint8_t));
        out.write(reinterpret_cast<const char*>(&dir.globals),
                  sizeof(lsh_globals<fin_graph_cfg::k_ranks>));
        write_rows(dir.edge_lshs.data(), sizeof(fin_edge_lsh_t), "edge_lshs");
      }

      if (has_pq) {
        const uint32_t m_u32 = fin_graph_cfg::pq_m;
        const uint32_t k_u32 = fin_graph_cfg::pq_k;
        const uint32_t dsub  = dir.codebooks.dsub;
        out.write(reinterpret_cast<const char*>(&m_u32), sizeof(uint32_t));
        out.write(reinterpret_cast<const char*>(&k_u32), sizeof(uint32_t));
        out.write(reinterpret_cast<const char*>(&dsub),  sizeof(uint32_t));

        const size_t centroid_count = dir.codebooks.centroids_count();
        std::vector<float> h_centroids(centroid_count);
        if (cudaMemcpy(h_centroids.data(), dir.codebooks.d_centroids,
                       centroid_count * sizeof(float),
                       cudaMemcpyDeviceToHost) != cudaSuccess)
          throw std::runtime_error("[construct_diskann] cudaMemcpy pq centroids out");
        out.write(reinterpret_cast<const char*>(h_centroids.data()),
                  static_cast<std::streamsize>(centroid_count * sizeof(float)));

        write_rows(dir.edge_pqs.data(), sizeof(fin_edge_pq_t), "edge_pqs");
      }

      if (has_norms)
        write_rows(dir.norms.data(), sizeof(float), "vector norms");

      // Last block, so a reader that doesn't know flag bit 3 never reads into it.
      if (has_rotation) {
        const uint8_t rot = dir.rotation.prerotate ? 1u : 0u;
        out.write(reinterpret_cast<const char*>(&rot), sizeof(uint8_t));
        out.write(reinterpret_cast<const char*>(&dir.rotation.seed), sizeof(uint64_t));
      }

      out.close();

      std::cout << "Appended directional trailer to " << output_fname
                << " (lsh=" << has_lsh << ", pq=" << has_pq
                << ", norms=" << has_norms;
      if (has_rotation)
        std::cout << ", prerotate=" << dir.rotation.prerotate
                  << " seed=" << dir.rotation.seed;
      std::cout << ")\n";
    }
  }
};

}  // namespace jasper
