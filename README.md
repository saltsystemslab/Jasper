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

## API Examples

All Python APIs can be found under the `example/` folder, where we provides python scripts to construct and query vector index in jasper. 

The snippet below shows a minimal workflow: build an index from a set of vectors, then run a top-k search for a batch of queries.

```python
import jasper

# Load base vectors and query vectors from .bin files.
vectors = jasper.read_bin("vectors.bin", "f32")
queries = jasper.read_bin("queries.bin", "f32").to(device="cuda")

# Build the graph index on the GPU.
g = jasper.Graph.build(
    vectors,
    n_neighbors=64,       # graph degree
    distance="l2",        # "l2" or "ip"
    alpha=1.2,            # pruning parameter
    workspace_budget="10GB",
)

# (Optional) persist the index and reload it later.
g.save("index.bin")
# g = jasper.Graph.load("index.bin", dim=128, n_neighbors=64,
#                       data_type="f16", distance="l2")

# Query the index: returns neighbor indices and their distances.
indices, distances = g.search(queries, k=10, beam_width=64)
print(indices, distances)

g.free()
```

See `example/graph.py` for building and searching in one pass, `example/query.py` for loading a saved index and running queries, and `example/groundtruth.py` for generating exact k-NN ground truth to measure recall.

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
