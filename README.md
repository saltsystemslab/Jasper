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

## Benchmarks

Updated automatically by neu-ci after each nightly cluster run.

### QPS vs. recall

<!--neu-ci:qps_recall start-->
![qps_recall](https://raw.githubusercontent.com/saltsystemslab/jasperpy/ci-results/qps_recall.png)

_updated 2026-07-08 15:37 UTC_
<!--neu-ci:qps_recall end-->

### Latest results

<!--neu-ci:results start-->
| dataset | insert_vps | query_qps | recall | delete_ms | live_recall | viol | consolidate_ms | append_vps |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bigann10M | 548969.0 | 2033724.0 | 0.9569 | 1.5 | 0.957 | 0 | 3233.2 | 257150.0 |
| deep10M | 649083.0 | 2322609.0 | 0.9491 | 1.7 | 0.9495 | 0 | 2810.2 | 294327.0 |
| gist | 170374.0 | 272217.0 | 0.8474 | 1.0 | 0.847 | 0 | 918.8 | 161264.0 |
| openai | 50709.0 | 139181.0 | 0.8937 | 0.9 | 0.8937 | 0 | 13954.5 | 35116.0 |

_updated 2026-07-08 15:37 UTC_
<!--neu-ci:results end-->

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