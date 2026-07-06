> [!IMPORTANT]
> This repository is currently under development. Not all APIs are implemented.

# Jasper: Fast and scalable GPU-native ANNS index

Jasper is a GPU-native approximate nearest neighbor search (ANNS) index designed for speed and scalability. Drawing on the Vamana graph index and RaBitQ quantization, Jasper delivers state-of-the-art construction throughput and query performance entirely on the GPU.

The core algorithms are based on our research paper: [GPU-Accelerated ANNS: Quantized for Speed, Built for Change](https://arxiv.org/abs/2601.07048). This repository extends the [original experiment code](https://github.com/saltsystemslab/JasperGPUANNS) into a reusable header-only CUDA library with Python bindings via [TVM FFI](https://github.com/apache/tvm-ffi).


## Why Jasper?

- **Fast construction and search.** Jasper matches or exceeds state-of-the-art GPU-based ANNS libraries in both index build throughput and query latency.
- **Index updates.** Jasper supports inserting vectors without rebuilding the entire index.
- **RaBitQ quantization.** Jasper uses RaBitQ for vector quantization, which achieves higher performance than traditional product quantization and maps naturally to GPU computations.

## Compilation

Compile library
```bash
# Install tvm_ffi
pip install apache-tvm-ffi

# Build the FFI shared library
cmake -B build -DJASPER_BUILD_FFI=ON -DJASPER_BUILD_CMD=ON
cmake --build build -j
cmake --install build
pip install -e python/
```

Run tests
```bash
cmake -B build -DJASPER_BUILD_TESTS=ON
cmake --build build
cd build && ctest --output-on-failure
```

## Python Usage

All Python APIs can be found under the `example/` folder, where we provides python scripts to construct and query vector index in jasper. See `example/graph.py` for building and searching in one pass, `example/query.py` for loading a saved index and running queries, and `example/groundtruth.py` for generating exact k-NN ground truth to measure recall.

### Import

```python
import jasper
```

### Build a graph and search

```python
import torch
import jasper

vectors = torch.randn(1_000_000, 128, dtype=torch.float16, device="cuda")
# vectors: torch.Tensor, shape [n_vectors, dim], dtype=torch.float16, CUDA

g = jasper.Graph.build(
    vectors,             # torch.Tensor  [n_vectors, dim], dtype=torch.float16
    n_neighbors=64,       # int    — max neighbors per node (R)
    distance="l2",        # str | DistanceFunc — "l2" or "ip"
    alpha=1.2,             # float  — pruning factor (1.0=strict, >1.0=longer hops)
)
# g: jasper.Graph

queries = torch.randn(100, 128, dtype=torch.float16, device="cuda")
# queries: torch.Tensor, shape [n_queries, dim], dtype=torch.float16, CUDA

indices, distances = g.search(
    queries,              # torch.Tensor  [n_queries, dim], dtype=torch.float16, CUDA
    k=10,                  # int    — number of neighbors to return
    beam_width=64,          # int    — search beam width
)
# indices:   torch.Tensor, int32,   shape [n_queries, k]
# distances: torch.Tensor, float32, shape [n_queries, k]
```

### Load a graph from file and search

```python
g = jasper.Graph.load(
    "sift1m.graph",       # str  — path to graph binary file
    dim=128,               # int  — vector dimensionality
    n_neighbors=32,         # int  — max neighbors per node (R), must match file
)
# g: jasper.Graph

indices, distances = g.search(
    queries,   # torch.Tensor, [n_queries, dim], dtype=torch.float16, CUDA
    k=10,        # int — number of neighbors to return
)
# indices:   torch.Tensor, int32,   shape [n_queries, k]
# distances: torch.Tensor, float32, shape [n_queries, k]
```

### Save a graph

```python
g.save(
    "sift1m.graph",  # str — destination file path
)
# returns: None
```

### Read vectors from a binary file

```python
vectors = jasper.read_bin(
    "sift1m_base.fvecs.bin",  # str — path to binary vector file
    dtype="f32",                # str — on-disk dtype, "f32" or "u8"
).to("cuda")
# vectors: torch.Tensor, dtype=torch.float16, shape [n_vectors, dim], CUDA
```

### Evaluate recall against ground truth

```python
gt_ids, gt_dists = jasper.read_groundtruth(
    "sift1m_groundtruth.bin",  # str — path to ground truth binary file
    k=10,                        # int — number of neighbors to load
)
# gt_ids:   torch.Tensor, int32,   shape [n_queries, k]
# gt_dists: torch.Tensor, float32, shape [n_queries, k]

recall = jasper.get_recall(
    gt_ids,               # torch.Tensor / indexable — ground truth ids, shape [n_queries, k]
    indices.cpu(),          # torch.Tensor / indexable — search result ids, shape [n_queries, k]
    k=10,                     # int — number of neighbors considered per query
    n_queries=indices.shape[0],  # int — number of queries
)
# recall: float — value in [0, 1]
```

### Generate your own ground truth

```python
gt_ids, gt_dists = jasper.generate_groundtruth(
    vectors,        # torch.Tensor, CPU, [n, dim], dtype=float32 or float16
    queries,          # torch.Tensor, CPU, [nq, dim], dtype=float32 or float16
    k=100,              # int — number of neighbors to compute
    distance="l2",        # str — "l2" or "ip"
)
# gt_ids:   torch.Tensor, int32,   shape [nq, k], CPU
# gt_dists: torch.Tensor, float32, shape [nq, k], CPU

jasper.save_groundtruth(
    "groundtruth.bin",  # str — destination file path
    gt_ids,               # torch.Tensor, int32/int, shape [nq, k]
    gt_dists,               # torch.Tensor, float32,   shape [nq, k]
)
# returns: None
```

### Inspect a graph

```python
print(g)              # g: jasper.Graph -> str, e.g. Graph(n_vectors=1000000, dim=128, R=64, dtype=f16, dist=l2)
print(g.n_vectors, g.dim, g.n_neighbors)
# g.n_vectors:   int
# g.dim:         int
# g.n_neighbors: int

vec = g.get_vector(
    42,  # int — zero-based vector index
)
# vec: torch.Tensor, dtype=torch.float16, shape [dim], CUDA
```

### Free a graph

```python
g.free()
# returns: None
```

## Citation

```
@misc{mccoy2026gpuacceleratedannsquantizedspeed,
      title={GPU-Accelerated ANNS: Quantized for Speed, Built for Change}, 
      author={Hunter McCoy and Zikun Wang and Prashant Pandey},
      year={2026},
      eprint={2601.07048},
      archivePrefix={arXiv},
      primaryClass={cs.DB},
      url={https://arxiv.org/abs/2601.07048}, 
}
```
