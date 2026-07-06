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

    std::cout << "[construct_diskann] assigning " << N
              << " vectors to " << n_parts << " shards (n_dup=" << n_dup << ")\n";
    auto shard_ids = assign_shards(data, n_parts, n_dup, seed);

    // Host-resident final edge lists (no vectors), zero-initialized.
    std::cout << "[construct_diskann] allocating final edge lists on host\n";
    ig.edges.assign(static_cast<size_t>(N), fin_edge_list_t{});
    ig.counts.assign(static_cast<size_t>(N), 0);

    uint32_t batch_cap =
        graph_construct_workspace<INT_CFG>::max_batch_size_for_budget(workspace_budget);
    batch_cap = std::min(batch_cap, params.max_batch_size);

    auto check = [](cudaError_t err, const char* what) {
      if (err != cudaSuccess)
        throw std::runtime_error(std::string("[construct_diskann] ") + what + ": "
                                 + cudaGetErrorString(err));
    };

    for (uint32_t p = 0; p < n_parts; p++) {
      std::vector<index_t>& ids = shard_ids[p];
      const uint32_t count = static_cast<uint32_t>(ids.size());
      std::printf("[construct_diskann] shard %u / %u (%u vectors)\n",
                  p + 1, n_parts, count);
      if (count == 0) continue;

      // Gather this shard's vectors into a contiguous pageable host buffer.
      // Pageable malloc + an OpenMP copy avoids a multi-GB cudaMallocHost, whose
      // page-locking cost otherwise stalls the transition between shards.
      const size_t buf_elems = static_cast<size_t>(count) * padded_dim;
      data_t* buf = static_cast<data_t*>(std::malloc(buf_elems * sizeof(data_t)));
      if (!buf) throw std::runtime_error("[construct_diskann] shard gather malloc failed");
      #pragma omp parallel for schedule(static)
      for (uint32_t k = 0; k < count; k++) {
        std::memcpy(buf + static_cast<size_t>(k) * padded_dim,
                    data[static_cast<uint32_t>(ids[k])],
                    static_cast<size_t>(padded_dim) * sizeof(data_t));
      }
      vector_view_t shard_vecs(buf, dim, count, /*on_host=*/true);

      // Build the intermediate Vamana graph on device at degree R_int.
      graph_construct_params<INT_CFG> sp;
      sp.data_vectors   = shard_vecs;
      sp.alpha          = params.alpha;
      sp.max_batch_size = std::max<uint32_t>(1, std::min(batch_cap, count));
      sp.on_host        = false;   // build on device
      sp.prerotate      = false;   // shards must share one space; no per-shard rotation

      int_graph_t g = construct_graph<INT_CFG>(sp);
      std::free(buf);              // vectors are on device now; drop the gather buffer

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
      std::vector<int_edge_list_t> se(count);
      std::vector<uint8_t>         sc(count);
      using seg_t = graph_segment<int_graph_cfg>;
      std::vector<seg_t> segs(g.segments.begin(), g.segments.end());
      uint32_t off = 0;
      for (uint32_t s = 0; s < g.n_segments; s++) {
        const seg_t& seg = segs[s];
        const uint32_t seg_count = static_cast<uint32_t>(seg.n_vectors);
        cudaMemcpy(se.data() + off, seg.edges,
                   static_cast<size_t>(seg_count) * sizeof(int_edge_list_t),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(sc.data() + off, seg.edge_counts,
                   static_cast<size_t>(seg_count) * sizeof(uint8_t),
                   cudaMemcpyDeviceToHost);
        off += seg_count;
      }
      cudaDeviceSynchronize();
      g.deallocate();

      // Distance-only merge into the final edge lists. Each global id is unique
      // within a shard, so nodes can be merged in parallel without races.
      #pragma omp parallel for schedule(static)
      for (uint32_t i = 0; i < count; i++) {
        const index_t gid = ids[i];
        merge_row(ig.edges[gid], ig.counts[gid], se[i], sc[i]);
      }

      // Release this shard's storage before moving to the next.
      ids.clear();  ids.shrink_to_fit();
      se.clear();   se.shrink_to_fit();
      sc.clear();   sc.shrink_to_fit();

      std::printf("[construct_diskann] merged shard %u / %u\n", p + 1, n_parts);
    }

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
