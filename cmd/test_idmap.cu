// Minimal WarpSpeed integration probe: create the id_map table, insert a few
// (stable_id -> slot) pairs from a kernel, look them up, and verify. This
// de-risks the CPM/include/compile/runtime integration before wiring the table
// into the graph.

#include <cstdint>
#include <cstdio>
#include <vector>
#include <cooperative_groups.h>

#include "jasper/index/id_map.cuh"

namespace cg = cooperative_groups;
using INDEX_T = uint32_t;
using table_t = jasper::id_map_t<INDEX_T>;
static constexpr uint32_t TILE = 8;

__global__ void probe_insert(table_t* table, const INDEX_T* ids,
                             const INDEX_T* slots, uint32_t n) {
  auto tile = cg::tiled_partition<TILE>(cg::this_thread_block());
  uint32_t i = blockIdx.x;
  if (i >= n) return;
  table->upsert_replace(tile, jasper::id_to_key(ids[i]), slots[i]);
}

__global__ void probe_find(table_t* table, const INDEX_T* ids, INDEX_T* out,
                           uint32_t n) {
  auto tile = cg::tiled_partition<TILE>(cg::this_thread_block());
  uint32_t i = blockIdx.x;
  if (i >= n) return;
  INDEX_T v = 0;
  bool ok = table->find_with_reference(tile, jasper::id_to_key(ids[i]), v);
  if (tile.thread_rank() == 0) out[i] = ok ? v : jasper::id_map_miss<INDEX_T>();
}

#define CK(call) do { cudaError_t e=(call); if(e!=cudaSuccess){ \
  printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
  return 1; } } while(0)

int main() {
  table_t* table = jasper::id_map_create<INDEX_T>(/*capacity=*/4096);

  std::vector<INDEX_T> ids   = {0, 1, 2, 7, 1000000};
  std::vector<INDEX_T> slots = {10, 11, 12, 17, 99};
  uint32_t n = ids.size();

  INDEX_T *d_ids, *d_slots, *d_out;
  CK(cudaMalloc(&d_ids, n * sizeof(INDEX_T)));
  CK(cudaMalloc(&d_slots, n * sizeof(INDEX_T)));
  CK(cudaMalloc(&d_out, (n + 1) * sizeof(INDEX_T)));
  CK(cudaMemcpy(d_ids, ids.data(), n * sizeof(INDEX_T), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_slots, slots.data(), n * sizeof(INDEX_T), cudaMemcpyHostToDevice));

  probe_insert<<<n, TILE>>>(table, d_ids, d_slots, n);
  CK(cudaDeviceSynchronize());

  // Look up the inserted ids plus one absent id (42).
  std::vector<INDEX_T> q = {0, 1, 2, 7, 1000000, 42};
  uint32_t nq = q.size();
  INDEX_T* d_q;
  CK(cudaMalloc(&d_q, nq * sizeof(INDEX_T)));
  CK(cudaMemcpy(d_q, q.data(), nq * sizeof(INDEX_T), cudaMemcpyHostToDevice));
  probe_find<<<nq, TILE>>>(table, d_q, d_out, nq);
  CK(cudaDeviceSynchronize());

  std::vector<INDEX_T> out(nq);
  CK(cudaMemcpy(out.data(), d_out, nq * sizeof(INDEX_T), cudaMemcpyDeviceToHost));

  INDEX_T expect[] = {10, 11, 12, 17, 99, jasper::id_map_miss<INDEX_T>()};
  int fails = 0;
  for (uint32_t i = 0; i < nq; i++) {
    bool ok = (out[i] == expect[i]);
    printf("  id=%u -> slot=%u (expect %u) %s\n", q[i], out[i], expect[i],
           ok ? "OK" : "FAIL");
    if (!ok) fails++;
  }
  jasper::id_map_destroy<INDEX_T>(table);
  printf("===== id_map probe %s (%d failures) =====\n", fails ? "FAIL" : "PASS", fails);
  return fails ? 1 : 0;
}
