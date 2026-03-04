#pragma once

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>

#define CUDA_CHECK(call)                                                  \
  do {                                                                    \
    cudaError_t err = (call);                                             \
    if (err != cudaSuccess) {                                             \
      FAIL() << #call << " failed: " << cudaGetErrorString(err)           \
             << " at " << __FILE__ << ":" << __LINE__;                    \
    }                                                                     \
  } while (0)

// RAII device buffer for tests
template <typename T>
struct DeviceBuf {
  T* ptr = nullptr;
  size_t count = 0;

  DeviceBuf() = default;
  explicit DeviceBuf(size_t n) : count(n) {
    cudaMalloc(&ptr, sizeof(T) * n);
    cudaMemset(ptr, 0, sizeof(T) * n);
  }
  DeviceBuf(const std::vector<T>& h) : count(h.size()) {
    cudaMalloc(&ptr, sizeof(T) * count);
    cudaMemcpy(ptr, h.data(), sizeof(T) * count, cudaMemcpyHostToDevice);
  }
  std::vector<T> to_host() const {
    std::vector<T> h(count);
    cudaMemcpy(h.data(), ptr, sizeof(T) * count, cudaMemcpyDeviceToHost);
    return h;
  }
  ~DeviceBuf() { if (ptr) cudaFree(ptr); }
  DeviceBuf(const DeviceBuf&) = delete;
  DeviceBuf& operator=(const DeviceBuf&) = delete;
  DeviceBuf(DeviceBuf&& o) noexcept : ptr(o.ptr), count(o.count) { o.ptr = nullptr; }
};