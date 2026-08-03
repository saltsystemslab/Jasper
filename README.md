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
# CMAKE_CUDA_ARCHITECTURES defaults to 120; override for your target GPU
cmake -B build -DJASPER_BUILD_FFI=ON -DJASPER_BUILD_CMD=ON -DCMAKE_CUDA_ARCHITECTURES="90;120"
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