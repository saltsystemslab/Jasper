# Quick Start

## Install

Jasper's Python bindings wrap a CUDA C++ core via [tvm-ffi](https://pypi.org/project/apache-tvm-ffi/). You'll need a CUDA toolchain (nvcc + CMake) and an NVIDIA GPU to build it.

```bash
# Install tvm_ffi
pip install apache-tvm-ffi

# Build the FFI shared library
# CMAKE_CUDA_ARCHITECTURES defaults to 120; override for your target GPU
cmake -B build -DJASPER_BUILD_FFI=ON -DJASPER_BUILD_CMD=ON -DCMAKE_CUDA_ARCHITECTURES="90;120"
cmake --build build -j
cmake --install build
pip install -e python/
```

Run the test suite:

```bash
cmake -B build -DJASPER_BUILD_TESTS=ON
cmake --build build
cd build && ctest --output-on-failure
```

Runnable end-to-end scripts live under [`example/`](https://github.com/zikunw/jasper/tree/main/example): `graph.py` builds and searches in one pass, `query.py` loads a saved index and sweeps beam widths, and `groundtruth.py` generates exact k-NN ground truth for recall evaluation.

## Build a graph and search

```python
import torch
import jasper

vectors = torch.randn(1_000_000, 128, dtype=torch.float16, device="cuda")
# vectors: torch.Tensor, shape [n_vectors, dim], dtype=torch.float16, CUDA

g = jasper.Graph.build(
    vectors,          # torch.Tensor  [n_vectors, dim], dtype=torch.float16
    n_neighbors=64,    # int    — max neighbors per node (R)
    distance="l2",     # str | DistanceFunc — "l2" or "ip"
    alpha=1.2,          # float  — pruning factor (1.0=strict, >1.0=longer hops)
)
# g: jasper.Graph

queries = torch.randn(100, 128, dtype=torch.float16, device="cuda")
# queries: torch.Tensor, shape [n_queries, dim], dtype=torch.float16, CUDA

indices, distances = g.search(
    queries,       # torch.Tensor  [n_queries, dim], dtype=torch.float16, CUDA
    k=10,           # int    — number of neighbors to return
    beam_width=64,   # int    — search beam width
)
# indices:   torch.Tensor, int32,   shape [n_queries, k]
# distances: torch.Tensor, float32, shape [n_queries, k]
```

## Save and load a graph

```python
g.save("sift1m.graph")

g = jasper.Graph.load(
    "sift1m.graph",  # str  — path to graph binary file
    dim=128,          # int  — vector dimensionality
    n_neighbors=32,    # int  — max neighbors per node (R), must match file
)
```

## Insert and delete vectors

Jasper supports updating a live graph without a full rebuild.

```python
# Insert: wires new vectors into the existing graph and assigns stable ids
new_vectors = torch.randn(10_000, 128, dtype=torch.float16, device="cuda")
ids = g.append(new_vectors, alpha=1.2)
# ids: torch.Tensor, int32, shape [n] — assigned stable ids, in input order

# Delete: soft-delete hides vectors immediately; consolidate/compact repair the graph
g.mark_deleted(ids)          # hidden from search right away
g.consolidate(alpha=1.2)     # repairs edges, clears tombstones
g.compact()                  # reclaims space, preserves stable ids
```

Stable ids are the ids returned by `search()`; they're never reused and are unchanged by `consolidate`/`compact`.

## Directional search (LSH / PQ estimators)

A **directional graph** stores extra per-edge estimator artifacts alongside the plain graph, enabling beam search scored by a cheaper proxy distance instead of the exact one.

```python
# LSH estimator
g = jasper.Graph.build(
    vectors, n_neighbors=64, distance="l2", alpha=1.2,
    build_lsh=True,   # populate LSH edges, enables directional_search()
    k_ranks=4,         # LSH rank count (4 or 16)
)
indices, distances = g.directional_search(queries, k=10, beam_width=64)

# PQ estimator
g = jasper.Graph.build(
    vectors, n_neighbors=64, distance="l2", alpha=1.2,
    build_pq=True,    # populate PQ edges + exact norms, enables pq_search()
    k_ranks=4,         # PQ subquantizer count (4 or 16)
)
indices, distances = g.pq_search(queries, k=10, beam_width=64)
```

`build_lsh` and `build_pq` can be enabled together on the same graph and queried independently. A graph built with either flag is *directional*: plain `g.search()` raises — use `directional_search()` / `pq_search()` instead.

## Evaluate recall

```python
gt_ids, gt_dists = jasper.read_groundtruth("sift1m_groundtruth.bin", k=10)

recall = jasper.get_recall(gt_ids, indices.cpu(), k=10, n_queries=indices.shape[0])
# recall: float — value in [0, 1]
```

See the [Python API reference](api.md) for the full set of methods, including reading raw vector files, generating your own ground truth, and inspecting graph/deletion state.
