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

// Builds the DiskANN-style merged graph. INT_CFG drives the per-shard build at
// degree R_int; FIN_CFG defines the final merged graph at degree R_final. Both
// configs must share data_t / index_t / distance_t / dist_func; only the degree
// differs. The result is held as host-resident edge lists (no vectors) and
// written out with save_to_file, which streams vectors from the dataset.
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

  static constexpr uint32_t R_INT = int_graph_cfg::n_neighbors;
  static constexpr uint32_t R_FIN = fin_graph_cfg::n_neighbors;

  static_assert(std::is_same<data_t, typename int_graph_cfg::data_t>::value,
                "INT_CFG and FIN_CFG must share data_t");
  static_assert(std::is_same<index_t, typename int_graph_cfg::index_t>::value,
                "INT_CFG and FIN_CFG must share index_t");
  static_assert(std::is_same<distance_t, typename int_graph_cfg::distance_t>::value,
                "INT_CFG and FIN_CFG must share distance_t");
  static_assert(int_graph_cfg::dist_func == fin_graph_cfg::dist_func,
                "INT_CFG and FIN_CFG must share dist_func");

  // Final graph, host-resident, vectors excluded. Indexed by global id.
  std::vector<fin_edge_list_t> edges;
  std::vector<uint8_t>         counts;
  index_t  n_vectors = 0;
  uint32_t dim       = 0;
  uint32_t n_parts   = 0;
  uint32_t n_dup     = 0;
  float    alpha     = 1.2f;

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
  static std::vector<std::vector<index_t>> assign_shards(
      vector_view_t data,
      uint32_t n_parts,
      uint32_t n_dup,
      uint64_t seed,
      size_t chunk_bytes = size_t{512} * 1024 * 1024
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

    // Random orthogonal rotation matrix (dim x dim), generated once.
    std::vector<float> h_P_f(static_cast<size_t>(dim) * dim);
    std::vector<float> h_Pt_f(static_cast<size_t>(dim) * dim);
    set_rotation_matrix<float>(static_cast<int>(dim), h_P_f.data(), h_Pt_f.data(), seed);
    std::vector<__half> h_P(h_P_f.size());
    for (size_t i = 0; i < h_P.size(); ++i) h_P[i] = static_cast<__half>(h_P_f[i]);
    __half* d_P = nullptr;
    check(cudaMalloc(&d_P, h_P.size() * sizeof(__half)), "cudaMalloc d_P");
    check(cudaMemcpy(d_P, h_P.data(), h_P.size() * sizeof(__half), cudaMemcpyHostToDevice),
          "cudaMemcpy d_P");

    data_t*  d_chunk    = nullptr;
    data_t*  d_rot      = nullptr;
    index_t* d_shard_of = nullptr;
    check(cudaMalloc(&d_chunk, static_cast<size_t>(chunk) * padded_dim * sizeof(data_t)),
          "cudaMalloc d_chunk");
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

      polytope_lsh_assign_kernel<data_t, index_t><<<cnt, block, smem>>>(
          d_rot, cnt, dim, padded_dim, n_parts, n_dup, d_shard_of);
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

  // Distance-only merge of one shard neighbor row into the final row for a node.
  // Unions the existing final neighbors with the shard's neighbors and keeps the
  // R_final smallest by stored distance, deduplicating by neighbor id (the first
  // occurrence in sorted order carries the smallest distance). No vector lookups.
  static inline void merge_row(fin_edge_list_t& out, uint8_t& out_count,
                               const int_edge_list_t& in, uint8_t in_count) {
    constexpr index_t INVALID = std::numeric_limits<index_t>::max();
    constexpr uint32_t MAXC = R_FIN + R_INT;

    index_t cid[MAXC];
    float   cd[MAXC];
    uint32_t nc = 0;

    for (uint8_t j = 0; j < out_count && nc < MAXC; j++) {
      index_t id = out.edges[j];
      if (id == INVALID) continue;
      cid[nc] = id;
      cd[nc]  = __bfloat162float(out.dist[j]);
      nc++;
    }
    for (uint8_t j = 0; j < in_count && nc < MAXC; j++) {
      index_t id = in.edges[j];
      if (id == INVALID) continue;
      cid[nc] = id;
      cd[nc]  = __bfloat162float(in.dist[j]);
      nc++;
    }

    // Insertion sort by distance ascending (nc is small, <= R_FIN + R_INT).
    for (uint32_t a = 1; a < nc; a++) {
      index_t ki = cid[a];
      float   kd = cd[a];
      int32_t b  = static_cast<int32_t>(a) - 1;
      while (b >= 0 && cd[b] > kd) {
        cid[b + 1] = cid[b];
        cd[b + 1]  = cd[b];
        b--;
      }
      cid[b + 1] = ki;
      cd[b + 1]  = kd;
    }

    // Select up to R_FIN unique ids in ascending-distance order.
    uint8_t sel = 0;
    for (uint32_t k = 0; k < nc && sel < R_FIN; k++) {
      index_t id = cid[k];
      bool dup = false;
      for (uint8_t t = 0; t < sel; t++) {
        if (out.edges[t] == id) { dup = true; break; }
      }
      if (dup) continue;
      out.edges[sel] = id;
      out.dist[sel]  = __float2bfloat16(cd[k]);
      sel++;
    }
    out_count = sel;
  }

  // ── Full build: assign shards, then build + merge each shard in turn ────────
  static diskann_intermediate_graph build(
      graph_construct_params<INT_CFG> params,
      uint32_t n_parts,
      uint32_t n_dup,
      size_t   workspace_budget,
      uint64_t seed = 42
  ) {
    if (n_parts == 0 || n_dup == 0) {
      throw std::runtime_error("[construct_diskann] n_parts and n_dup must be > 0");
    }

    vector_view_t data = params.data_vectors;
    const uint32_t dim        = data.dim;
    const uint32_t padded_dim = vector_view_t::pad(dim);
    const index_t  N          = static_cast<index_t>(data.n_vectors);

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

    std::cout << "[construct_diskann] assigning " << N
              << " vectors to " << n_parts << " shards (n_dup=" << n_dup << ")\n";
    auto t_assign0 = clk::now();
    auto shard_ids = assign_shards(data, n_parts, n_dup, seed);
    const double t_assign = secs(t_assign0, clk::now());

    // Host-resident final edge lists (no vectors), zero-initialized.
    std::cout << "[construct_diskann] allocating final edge lists on host\n";
    auto t_alloc0 = clk::now();
    ig.edges.assign(static_cast<size_t>(N), fin_edge_list_t{});
    ig.counts.assign(static_cast<size_t>(N), 0);
    const double t_alloc = secs(t_alloc0, clk::now());

    // Per-phase wall-clock accumulators (see the summary printed at the end).
    // gather runs on a background thread, so it is timed in atomic microseconds;
    // the rest run on the main thread.
    std::atomic<long long> gather_us{0};
    double t_construct = 0, t_post = 0, t_merge_wait = 0;

    uint32_t batch_cap =
        graph_construct_workspace<INT_CFG>::max_batch_size_for_budget(workspace_budget);
    batch_cap = std::min(batch_cap, params.max_batch_size);

    auto check = [](cudaError_t err, const char* what) {
      if (err != cudaSuccess)
        throw std::runtime_error(std::string("[construct_diskann] ") + what + ": "
                                 + cudaGetErrorString(err));
    };

    // Largest shard we can build on the device: the shard graph needs
    // per_vec bytes per vector, and we must leave room for the construction
    // workspace (~workspace_budget) plus slack. Shards exceeding this are split
    // into sub-shards so an imbalanced LSH partition can never OOM the device.
    const size_t per_vec = static_cast<size_t>(padded_dim) * sizeof(data_t)
                         + sizeof(int_edge_list_t) + sizeof(uint8_t);
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    const size_t reserve = workspace_budget + (size_t{4} << 30);  // 4 GB slack
    const size_t avail   = (free_b > reserve) ? (free_b - reserve) : (free_b / 2);
    const uint32_t max_shard = static_cast<uint32_t>(
        std::min<size_t>(std::max<size_t>(avail / per_vec, 1), 0xFFFFFFFFull));
    std::printf("[construct_diskann] max shard build size = %u vectors "
                "(device free %.1f GB)\n",
                max_shard, free_b / (1024.0 * 1024.0 * 1024.0));

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

    // Build one task from an already-gathered buffer into the reused graph `g`,
    // returning its edges + counts on host. Frees `buf`.
    auto build_from_buf = [&](data_t* buf, const std::vector<index_t>& ids,
                              std::vector<int_edge_list_t>& se,
                              std::vector<uint8_t>& sc) {
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

      // Copy this shard's edges + counts back to host (vectors stay on device).
      se.resize(count);
      sc.resize(count);
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
        off += seg_count;
      }
      cudaDeviceSynchronize();
      t_post += secs(tb2, clk::now());
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
          merge_row(ig.edges[gid], ig.counts[gid], job.se[i], job.sc[i]);
        }
      }
    });

    auto enqueue_merge = [&](std::vector<int_edge_list_t>&& se,
                             std::vector<uint8_t>&& sc,
                             std::vector<index_t>&& ids) {
      const auto tw0 = clk::now();
      std::unique_lock<std::mutex> lk(merge_mtx);
      merge_cv_push.wait(lk, [&] { return merge_q.size() < MERGE_QCAP; });
      t_merge_wait += secs(tw0, clk::now());
      merge_q.push_back(merge_job_t{std::move(se), std::move(sc), std::move(ids)});
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
      build_from_buf(buf, ids, se, sc);
      enqueue_merge(std::move(se), std::move(sc), std::move(ids));
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

    std::printf(
        "[construct_diskann] phase totals (s): assign=%.1f  alloc_edges=%.1f  "
        "gather=%.1f  construct=%.1f  remap+copy=%.1f  merge_wait=%.1f  "
        "merge_tail=%.1f\n",
        t_assign, t_alloc, gather_us.load() / 1e6, t_construct, t_post,
        t_merge_wait, t_merge_tail);

    return ig;
  }

  __host__ double avg_degree() const {
    if (n_vectors == 0) return 0.0;
    uint64_t total = 0;
    for (index_t i = 0; i < n_vectors; i++) total += counts[i];
    return static_cast<double>(total) / static_cast<double>(n_vectors);
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
};

}  // namespace jasper
