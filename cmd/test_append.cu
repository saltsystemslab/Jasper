// test_append — validate live batch append + stable ids.
//   1. construct an index from base[0:N]
//   2. append base[N:N+M] as a live batch -> reserved ids must be [N, N+M)
//   3. query with those exact vectors -> each must find its own appended id
//      (recall@1 ~ 1.0), proving edges were wired in and ids are correct.

#include <argparse/argparse.hpp>
#include <thrust/pair.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <iostream>
#include <vector>

#include "jasper/jasper.cuh"

#define CUDA_CHECK(call)                                                       \
  do { cudaError_t e=(call); if(e!=cudaSuccess){                               \
    std::cerr<<"CUDA "<<cudaGetErrorString(e)<<" @"<<__FILE__<<":"<<__LINE__   \
             <<"\n"; throw std::runtime_error("cuda"); } } while(0)

using cfg = jasper::graph_config<uint32_t, 64, __half, float, jasper::distance_func::L2>;
using construct_cfg = jasper::graph_construct_config<cfg, 64, 4, 64, 64>;
using INDEX_T = uint32_t;

__global__ void unpack_ids(const thrust::pair<uint32_t, float>* p,
                           int32_t* out, uint32_t total) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < total) out[i] = static_cast<int32_t>(p[i].first);
}

int main(int argc, char** argv) {
  argparse::ArgumentParser program("test_append");
  program.add_argument("--base", "-b").required();
  program.add_argument("--datatype", "-t").default_value(std::string{"uint8"}).choices("uint8", "float");
  program.add_argument("--dim").required().scan<'u', uint32_t>();
  program.add_argument("--n_construct", "-n").default_value(uint64_t{1000000}).scan<'u', uint64_t>();
  program.add_argument("--n_append", "-a").default_value(uint64_t{100000}).scan<'u', uint64_t>();
  program.add_argument("--rounds", "-r").default_value(uint32_t{1}).scan<'u', uint32_t>();
  program.add_argument("--k", "-k").default_value(uint32_t{10}).scan<'u', uint32_t>();
  program.add_argument("--beam_width").default_value(uint32_t{64}).scan<'u', uint32_t>();
  try { program.parse_args(argc, argv); }
  catch (const std::exception& e) { std::cerr << e.what() << "\n" << program; return 1; }

  auto base_path = program.get<std::string>("--base");
  auto datatype  = program.get<std::string>("--datatype");
  auto dim       = program.get<uint32_t>("--dim");
  auto N         = (uint32_t)program.get<uint64_t>("--n_construct");
  auto M         = (uint32_t)program.get<uint64_t>("--n_append");
  auto rounds    = program.get<uint32_t>("--rounds");
  auto k         = program.get<uint32_t>("--k");
  auto bw        = program.get<uint32_t>("--beam_width");

  std::cout << "=== test_append === N(construct)=" << N << " M(append)=" << M
            << " dim=" << dim << " k=" << k << " bw=" << bw << std::endl;

  // Load base vectors (host) and move to device.
  jasper::vector_view<__half> h_base =
      (datatype == "float")
          ? jasper::load_vectors_from_file_cast<__half, float>(base_path)
          : jasper::load_vectors_from_file_cast<__half, uint8_t>(base_path);
  if (!h_base.data) { std::cerr << "failed to load base\n"; return 1; }
  if (h_base.dim != dim) { std::cerr << "dim mismatch\n"; return 1; }
  if (h_base.n_vectors < (uint64_t)N + (uint64_t)M * rounds) {
    std::cerr << "base too small for N + M*rounds\n"; return 1;
  }
  auto d_base = h_base.to_device();
  cudaFreeHost(h_base.data);

  int failures = 0;

  // 1. construct from base[0:N]
  jasper::graph_construct_params<construct_cfg> params;
  params.data_vectors   = d_base.subview(0, N);
  params.alpha          = 1.2f;
  params.max_batch_size = std::min<uint32_t>(N / 50, 200000);
  params.on_host        = false;
  auto g = jasper::construct_graph<construct_cfg>(params);
  std::cout << "[construct] n_vectors=" << g.n_vectors
            << " next_id=" << g.next_id << std::endl;

  // 2. append base[N + r*M : N + (r+1)*M] over `rounds` rounds. Each round's
  //    ids must continue the monotonic sequence; capacity resizes transparently.
  for (uint32_t r = 0; r < rounds; r++) {
    uint32_t off = N + r * M;
    auto ids = jasper::append_batch<construct_cfg>(g, d_base.subview(off, M), 1.2f);
    std::cout << "[append r" << r << "] ids [" << ids.front() << ", " << ids.back()
              << "]  n_vectors=" << g.n_vectors << " next_id=" << g.next_id
              << " table_cap=" << g.id_map_capacity << std::endl;
    if (ids.size() != M)          { std::cerr << "FAIL: id count\n"; failures++; }
    if (ids.front() != off)       { std::cerr << "FAIL: first id != " << off << "\n"; failures++; }
    if (ids.back() != off + M - 1){ std::cerr << "FAIL: last id\n"; failures++; }
    if (g.n_vectors != off + M)   { std::cerr << "FAIL: n_vectors\n"; failures++; }
  }
  uint32_t total_appended = M * rounds;

  // 3. query with all appended vectors; each should find its own id.
  jasper::vector_view<__half> q = d_base.subview(N, total_appended);
  jasper::search_params sp{.k = k, .beam_width = bw, .limit = bw * 2, .get_visited = false};
  auto res = jasper::search(g, q, sp);

  uint32_t total = total_appended * k;
  jasper::translate_slots_to_ids_kernel<cfg><<<(total + 255) / 256, 256>>>(
      g.view(), res.frontier, total);
  int32_t* d_out; CUDA_CHECK(cudaMalloc(&d_out, total * sizeof(int32_t)));
  unpack_ids<<<(total + 255) / 256, 256>>>(res.frontier, d_out, total);
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<int32_t> out(total);
  CUDA_CHECK(cudaMemcpy(out.data(), d_out, total * sizeof(int32_t), cudaMemcpyDeviceToHost));

  uint64_t self_top1 = 0, self_topk = 0;
  for (uint32_t i = 0; i < total_appended; i++) {
    int32_t want = (int32_t)(N + i);
    if (out[i * k + 0] == want) self_top1++;
    for (uint32_t j = 0; j < k; j++) if (out[i * k + j] == want) { self_topk++; break; }
  }
  double r1 = (double)self_top1 / total_appended, rk = (double)self_topk / total_appended;
  std::cout << "[query] appended-vector self-recall@1=" << r1
            << " self-recall@" << k << "=" << rk << std::endl;
  // Exact-match queries should almost always return their own appended id.
  if (rk < 0.99) { std::cerr << "FAIL: appended vectors not findable\n"; failures++; }

  cudaFree(d_out); cudaFree(res.frontier); cudaFree(d_base.data); g.deallocate();
  std::cout << "===== test_append " << (failures ? "FAIL" : "PASS")
            << " (" << failures << " failures) =====" << std::endl;
  return failures ? 1 : 0;
}
