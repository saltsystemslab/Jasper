#pragma once

#include <limits>

namespace jasper {

// static constexpr uint32_t INVALID_INDEX = 0x7FFFFFFFu;

// Helper function to read an entry.
// the structure of the entry is
// - 1 bits:  visited
// - 31 bits: index (uint32_t)
// - 32 bits: distance
typedef uint64_t ENTRY_T;

static __device__ __forceinline__ ENTRY_T create_entry(uint32_t index, float distance) {
  uint64_t entry = 0;
  entry |= (uint64_t(index) & 0x7FFFFFFFul) << 32;
  uint32_t dist_bits = __float_as_uint(distance);
  entry |= (static_cast<uint64_t>(dist_bits));
  return entry;
}

static __device__ __forceinline__ ENTRY_T empty_entry() {
  return create_entry(
    std::numeric_limits<uint32_t>::max(),
    std::numeric_limits<float>::max()
  );
}

static __device__ __forceinline__ bool get_visited(ENTRY_T i) {
  return (i >> 63) & 1;
}

static __device__ __forceinline__ ENTRY_T set_visited(ENTRY_T i) {
  constexpr uint64_t MASK = 1ull << 63;
  return i | MASK;
}

static __device__ __forceinline__ uint32_t get_index(ENTRY_T entry) {
  return static_cast<uint32_t>((entry >> 32) & 0x7FFFFFFF);
}

static __device__ __forceinline__ ENTRY_T set_index(ENTRY_T entry, uint32_t index) {
  uint64_t idx64 = static_cast<uint64_t>(index) << 32;
  uint64_t dist64 = entry & 0xFFFFFFFFull;
  return dist64 | idx64;
}

static __device__ __forceinline__ float get_distance(ENTRY_T entry) {
  uint32_t dist_bits = static_cast<uint32_t>(entry & 0xFFFFFFFF);
  return __uint_as_float(dist_bits);
}

static __device__ __forceinline__ ENTRY_T set_distance(ENTRY_T entry, float distance) {
  uint32_t dist_bits = __float_as_uint(static_cast<float>(distance));
  uint64_t dist64 = uint64_t(dist_bits);
  uint64_t idx64 = entry & 0xFFFFFFFF00000000ull;
  return idx64 | dist64;
}

}