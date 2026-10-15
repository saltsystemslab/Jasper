# Directional Beam Search

## Compilation

Compile library
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

## Slurm benchmark examples

Slurm scripts for benchmarks found inside the paper is stored in `sbatch/`.

## Python Usage

All Python APIs can be found under the `example/` folder, where we provides python scripts to construct and query vector index in jasper. See `example/graph.py` for building and searching in one pass, `example/query.py` for loading a saved index and running queries, and `example/groundtruth.py` for generating exact k-NN ground truth to measure recall.

### Directional search (LSH / PQ estimators)

A **directional graph** stores extra per-edge estimator artifacts alongside
the plain graph, enabling beam search scored by a cheaper proxy distance
instead of the exact one. Build with `build_lsh=True` for a cross-polytope
LSH estimator (`directional_search()`), or `build_pq=True` for a
Product-Quantization ADC estimator (`pq_search()`). Both can also be
enabled together on the same graph and queried independently.

#### LSH estimator

```python
g = jasper.Graph.build(
    vectors,             # torch.Tensor  [n_vectors, dim], dtype=torch.float16
    n_neighbors=64,       # int    — max neighbors per node (R)
    distance="l2",         # str | DistanceFunc — "l2" or "ip"
    alpha=1.2,               # float  — pruning factor (1.0=strict, >1.0=longer hops)
    build_lsh=True,           # bool — populate LSH edges, enables directional_search()
    k_ranks=4,                  # int — LSH rank count (4 or 16)
    prerotate=None,               # bool | None — rotate vectors/queries by a random
                                   #   orthogonal matrix; defaults to True since it's
                                   #   needed for LSH estimator quality
    lsh_samples=32768,             # int — edge samples used to calibrate LSH globals
)
# g: jasper.Graph, is_directional=True
# NOTE: g.search() raises on a directional graph — use directional_search() instead.

indices, distances = g.directional_search(
    queries,              # torch.Tensor  [n_queries, dim], dtype=torch.float16, CUDA
    k=10,                  # int    — number of neighbors to return
    beam_width=64,          # int    — search beam width
)
# indices:   torch.Tensor, int32,   shape [n_queries, k]
# distances: torch.Tensor, float32, shape [n_queries, k]
# Requires build_lsh=True at build() time (or LSH artifacts present at load()).
```

#### PQ estimator

```python
g = jasper.Graph.build(
    vectors,             # torch.Tensor  [n_vectors, dim], dtype=torch.float16
    n_neighbors=64,       # int    — max neighbors per node (R)
    distance="l2",         # str | DistanceFunc — "l2" or "ip"
    alpha=1.2,               # float  — pruning factor (1.0=strict, >1.0=longer hops)
    build_pq=True,            # bool — populate PQ edges + exact norms, enables pq_search()
    k_ranks=4,                  # int — PQ subquantizer count (4 or 16)
    prerotate=None,               # bool | None — rotate vectors/queries by a random
                                   #   orthogonal matrix; defaults to True since it's
                                   #   needed for PQ estimator quality
    pq_train=40000,                # int — residual samples used to train PQ codebooks
    pq_kmeans_iter=12,               # int — k-means iterations for PQ codebook training
)
# g: jasper.Graph, is_directional=True
# NOTE: g.search() raises on a directional graph — use pq_search() instead.

indices, distances = g.pq_search(
    queries,              # torch.Tensor  [n_queries, dim], dtype=torch.float16, CUDA
    k=10,                  # int    — number of neighbors to return
    beam_width=64,          # int    — search beam width
)
# indices:   torch.Tensor, int32,   shape [n_queries, k]
# distances: torch.Tensor, float32, shape [n_queries, k]
# Requires build_pq=True at build() time (or PQ artifacts present at load()).
```

### Save a graph

```python
g.save(
    "sift1m.graph",  # str — destination file path
)
# returns: None
```

