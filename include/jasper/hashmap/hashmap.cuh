#pragma once

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cooperative_groups/scan.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <fstream>
#include <iostream>
#include <vector>

#include "assert.h"
#include "stdio.h"

namespace cg = cooperative_groups;

namespace jasper {
// a shared memory version of hashtable
// for beam search's visited vectors.
namespace hashmap {

// default key is with all 1s.
constexpr uint32_t default_key = ~static_cast<uint32_t>(0);

__forceinline__ __device__ uint32_t hash(uint32_t key, uint32_t size) {
  return key * (key + 3) % size;
}

__device__ uint32_t get_size(const uint32_t bitlen) { return 1U << bitlen; }

__host__ uint32_t get_size_host(const uint32_t bitlen) { return 1U << bitlen; }

__device__ void init(uint32_t* table, uint32_t bitlen) {
  for (unsigned i = threadIdx.x; i < get_size(bitlen); i += blockDim.x) {
    table[i] = default_key;
  }
}

__device__ uint32_t insert(uint32_t* table, const uint32_t bitlen,
                           const uint32_t key) {
  const uint32_t size = get_size(bitlen);
  const uint32_t bit_mask = get_size(bitlen) - 1;
  uint32_t index = hash(key, size);

  for (unsigned i = 0; i < size; i++) {
    const uint32_t old = table[index];
    if (old == default_key) {
      table[index] = key;
      return 1;  // inserted
    } else if (old == key) {
      return 0;  // duplicate
    }
    index = (index + 1) & bit_mask;  // linear probe
  }
  return 0;  // table full
}

__device__ void insert_fast(uint32_t* table, const uint32_t bitlen,
                            const uint32_t key) {
  const uint32_t size = get_size(bitlen);
  const uint32_t bit_mask = get_size(bitlen) - 1;

  for (unsigned i = threadIdx.x; i < get_size(bitlen); i += blockDim.x) {
    bool prev_empty = false;
    if (i != 0) {
      prev_empty = (table[i - 1] == default_key);
    }

    bool next_empty = (table[i] == default_key);

    if (!prev_empty && next_e mpty) {
      table[i] = key;
    }
  }

  return;
}

__device__ uint32_t search(uint32_t* table, const uint32_t bitlen,
                           const uint32_t key) {
  const uint32_t size = get_size(bitlen);
  const uint32_t bit_mask = size - 1;
  uint32_t index = hash(key, size);
  for (unsigned i = 0; i < size; i++) {
    const uint32_t val = table[index];
    if (val == key) {
      return 1;
    } else if (val == default_key) {
      return 0;
    }
    index = (index + 1) & bit_mask;  // linear probe
  }
  return 0;
}

}  // namespace hashmap
}  // namespace jasper