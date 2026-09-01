// bench_all — time construction, query (QPS + recall), and the deletion ops
// (mark_deleted / consolidate / compact) plus a live append, for the CURRENT
// (stable-id + deletion) build. Construction + query are directly comparable to
// the original via create_index / run_query.

#include <argparse/argparse.hpp>
#include <thrust/pair.h>
#include <cuda_fp16.h>

#include <chrono>
#include <cstdint>
#include <fstream>
#include <functional>
#include <iostream>
#include <random>
#include <set>
#include <unordered_set>
#include <vector>

#include "jasper/jasper.cuh"

#define CK(call) do { cudaError_t e=(call); if(e!=cudaSuccess){ \
  std::cerr<<"CUDA "<<cudaGetErrorString(e)<<" @"<<__LINE__<<"\n"; throw std::runtime_error("cuda"); } } while(0)

using cfg = jasper::graph_config<uint32_t, 64, __half, float, jasper::distance_func::L2>;
using construct_cfg = jasper::graph_construct_config<cfg, 64, 4, 64, 64>;

__global__ void unpack_ids(const thrust::pair<uint32_t,float>* p, int32_t* out, uint32_t n) {
  uint32_t i = blockIdx.x*blockDim.x+threadIdx.x; if (i<n) out[i]=(int32_t)p[i].first;
}

struct GT { uint32_t nq=0,k=0; std::vector<int32_t> ids; };
static GT read_gt(const std::string& path, uint32_t k) {
  std::ifstream f(path, std::ios::binary); GT g; if(!f) return g;
  uint32_t nq,gk; f.read((char*)&nq,4); f.read((char*)&gk,4);
  std::vector<uint32_t> all((size_t)nq*gk); f.read((char*)all.data(),(size_t)nq*gk*4);
  g.nq=nq; g.k=k; g.ids.resize((size_t)nq*k);
  for(uint32_t q=0;q<nq;q++) for(uint32_t j=0;j<k;j++) g.ids[q*k+j]=(int32_t)all[q*gk+j];
  return g;
}

// timed search -> (host indices [nq*k], elapsed ms)
static std::vector<int32_t> timed_search(jasper::graph<cfg>& g, __half* dq, uint32_t nq,
                                         uint32_t k, uint32_t bw, float& ms) {
  jasper::vector_view<__half> qv(dq, g.dim, nq, false);
  jasper::search_params sp{.k=k,.beam_width=bw,.limit=bw*2,.get_visited=false};
  cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
  cudaEventRecord(e0);
  auto res = jasper::search(g, qv, sp);
  cudaEventRecord(e1); cudaEventSynchronize(e1);
  ms=0; cudaEventElapsedTime(&ms,e0,e1); cudaEventDestroy(e0); cudaEventDestroy(e1);
  uint32_t total=nq*k;
  jasper::translate_slots_to_ids_kernel<cfg><<<(total+255)/256,256>>>(g.view(),res.frontier,total);
  int32_t* d; CK(cudaMalloc(&d,total*sizeof(int32_t)));
  unpack_ids<<<(total+255)/256,256>>>(res.frontier,d,total); CK(cudaDeviceSynchronize());
  std::vector<int32_t> h(total); CK(cudaMemcpy(h.data(),d,total*sizeof(int32_t),cudaMemcpyDeviceToHost));
  cudaFree(d); cudaFree(res.frontier); return h;
}

static double recall(const GT& gt, const std::vector<int32_t>& r, uint32_t k, uint32_t nq,
                     const std::unordered_set<uint32_t>& del) {
  if(gt.nq==0) return -1; uint64_t hit=0,tot=0;
  for(uint32_t q=0;q<nq;q++){ std::set<int32_t> got(r.data()+q*k,r.data()+q*k+k);
    for(uint32_t j=0;j<k;j++){ int32_t id=gt.ids[q*k+j]; if(del.count((uint32_t)id)) continue; tot++; if(got.count(id)) hit++; } }
  return tot? (double)hit/tot : 0;
}
static uint64_t viols(const std::vector<int32_t>& r, const std::unordered_set<uint32_t>& del){
  uint64_t v=0; for(int32_t id:r) if(id>=0&&del.count((uint32_t)id)) v++; return v; }

int main(int argc, char** argv){
  argparse::ArgumentParser p("bench_all");
  p.add_description(
      "Build a graph from --base, run queries from --queries, then benchmark "
      "delete + append.\nVector files are binary: [int32 n][int32 dim][n*dim "
      "elements], elements typed per --datatype.");
  p.add_argument("--base","-b").required()
      .help("(required) base vectors file to build the index from");
  p.add_argument("--queries","-q").required()
      .help("(required) query vectors file (same binary format as --base)");
  p.add_argument("--gt","-g").default_value(std::string{})
      .help("ground-truth file for recall; omit to skip recall reporting");
  p.add_argument("--datatype","-t").default_value(std::string{"uint8"}).choices("uint8","float")
      .help("element type of the vector files: 'uint8' or 'float' (default: uint8)");
  p.add_argument("--dim").required().scan<'u',uint32_t>()
      .help("(required) vector dimension");
  p.add_argument("--k","-k").default_value(uint32_t{10}).scan<'u',uint32_t>()
      .help("number of nearest neighbors to return (default: 10)");
  p.add_argument("--beam_width").default_value(uint32_t{64}).scan<'u',uint32_t>()
      .help("search beam width (default: 64)");
  p.add_argument("--delete_fraction","-f").default_value(0.05).scan<'g',double>()
      .help("fraction of vectors to delete in the delete benchmark (default: 0.05)");
  p.add_argument("--n_append","-a").default_value(uint64_t{100000}).scan<'u',uint64_t>()
      .help("number of vectors to append in the append benchmark (default: 100000)");
  try {
    p.parse_args(argc, argv);
  } catch (const std::exception& e) {
    // Clear, actionable error: what went wrong + full usage/options.
    std::cerr << "bench_all: error: " << e.what() << "\n\n" << p;
    return 1;
  }

  auto base_path=p.get<std::string>("--base"); auto q_path=p.get<std::string>("--queries");
  auto gt_path=p.get<std::string>("--gt"); auto dt=p.get<std::string>("--datatype");
  auto dim=p.get<uint32_t>("--dim"); auto k=p.get<uint32_t>("--k"); auto bw=p.get<uint32_t>("--beam_width");
  auto df=p.get<double>("--delete_fraction"); auto na=(uint32_t)p.get<uint64_t>("--n_append");

  auto h_base = (dt=="float") ? jasper::load_vectors_from_file_cast<__half,float>(base_path)
                              : jasper::load_vectors_from_file_cast<__half,uint8_t>(base_path);
  auto d_base = h_base.to_device(); cudaFreeHost(h_base.data);
  auto h_q = (dt=="float") ? jasper::load_vectors_from_file_cast<__half,float>(q_path)
                           : jasper::load_vectors_from_file_cast<__half,uint8_t>(q_path);
  uint32_t nq=h_q.n_vectors; auto d_qv=h_q.to_device(); cudaFreeHost(h_q.data); __half* dq=d_qv.data;
  GT gt; if(!gt_path.empty()) gt=read_gt(gt_path,k);
  std::unordered_set<uint32_t> none;

  auto wall=[](std::function<void()> fn){ auto t0=std::chrono::steady_clock::now(); fn();
    return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count(); };

  std::printf("\n##### bench_all: n_base=%u dim=%u df=%.2f append=%u #####\n",
              d_base.n_vectors, dim, df, na);

  // INSERT (construction)
  jasper::graph_construct_params<construct_cfg> cp;
  cp.data_vectors=d_base; cp.alpha=1.2f; cp.on_host=false;
  cp.max_batch_size=std::min<uint32_t>(d_base.n_vectors/50,200000);
  jasper::graph<cfg> g;
  double build_ms = wall([&]{ g = jasper::construct_graph<construct_cfg>(cp); });
  std::printf("[INSERT]  construct %u vectors: %.1f ms  (%.0f vec/s)\n",
              g.n_vectors, build_ms, g.n_vectors*1000.0/build_ms);

  // QUERY (baseline)
  float qms; auto r0=timed_search(g,dq,nq,k,bw,qms);
  std::printf("[QUERY ]  bw=%u: %.2f ms, %.0f QPS, recall@%u=%.4f\n",
              bw, qms, nq*1000.0/qms, k, recall(gt,r0,k,nq,none));

  // DELETION
  uint64_t ndel=(uint64_t)(df*g.n_vectors);
  std::mt19937_64 rng(1); std::unordered_set<uint32_t> del;
  { std::uniform_int_distribution<uint32_t> u(0,g.n_vectors-1); while(del.size()<ndel) del.insert(u(rng)); }
  std::vector<uint32_t> del_ids(del.begin(),del.end());
  double del_ms = wall([&]{ jasper::mark_deleted<construct_cfg>(g, del_ids.data(), (uint32_t)del_ids.size()); });
  float qms2; auto r1=timed_search(g,dq,nq,k,bw,qms2);
  std::printf("[DELETE]  mark_deleted %llu ids: %.1f ms | query %.0f QPS live_recall=%.4f viol=%llu\n",
              (unsigned long long)ndel, del_ms, nq*1000.0/qms2, recall(gt,r1,k,nq,del),
              (unsigned long long)viols(r1,del));
  double cons_ms = wall([&]{ jasper::consolidate<construct_cfg>(g,1.2f); });
  std::printf("[DELETE]  consolidate: %.1f ms\n", cons_ms);
#if JASPER_ENABLE_COMPACT
  uint32_t before=g.n_vectors;
  double comp_ms = wall([&]{ jasper::compact<construct_cfg>(g); });
  std::printf("[DELETE]  compact %u->%u: %.1f ms\n", before, g.n_vectors, comp_ms);
#else
  std::printf("[DELETE]  compact: disabled (mark + consolidate only)\n");
#endif

  // INSERT (live append)
  na=std::min<uint32_t>(na, d_base.n_vectors);
  double app_ms = wall([&]{ (void)jasper::append_batch<construct_cfg>(g, d_base.subview(0,na), 1.2f); });
  std::printf("[APPEND]  append %u vectors: %.1f ms  (%.0f vec/s)\n",
              na, app_ms, na*1000.0/app_ms);

  cudaFree(d_base.data); cudaFree(dq); g.deallocate();
  std::printf("##### done #####\n");
  return 0;
}
