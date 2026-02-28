#pragma once

#include <cstdint>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

namespace jasper {

enum class distance_func : uint8_t {
  L2,
  INNER_PRODUCT
};

// L2 squared distance
template <typename DATA_T, typename DISTANCE_T, uint32_t TILE_SIZE>
struct l2_distance {
  template <typename TileT>
  __device__ static DISTANCE_T compute(const DATA_T* a, const DATA_T* b, const uint32_t dim, TileT& tile) {
    DISTANCE_T sum = 0.0f;
    for (uint i = tile.thread_rank(); i < dim; i += TILE_SIZE) {
      DISTANCE_T diff = static_cast<DISTANCE_T>(a[i]) - static_cast<DISTANCE_T>(b[i]);
      sum += diff * diff;
    }
    sum = cg::reduce(tile, sum, cg::plus<DISTANCE_T>());
    return sum;
  }
};

// Inner product
template <typename DATA_T, typename DISTANCE_T, uint32_t TILE_SIZE>
struct inner_product_distance {
  template <typename TileT>
  __device__ static DISTANCE_T compute(const DATA_T* a, const DATA_T* b, const uint32_t dim, TileT& tile) {
    DISTANCE_T sum = 0.0f;
    for (uint i = tile.thread_rank(); i < dim; i += TILE_SIZE) {
      sum += static_cast<DISTANCE_T>(a[i]) * static_cast<DISTANCE_T>(b[i]);
    }
    sum = cg::reduce(tile, sum, cg::plus<DISTANCE_T>());
    return -sum;
  }
};

template <distance_func FUNC, typename DATA_T, typename DISTANCE_T, uint32_t TILE_SIZE, typename TileT>
__device__ DISTANCE_T compute_distance(const DATA_T* a, const DATA_T* b, const uint32_t dim, TileT& tile) {
  if constexpr (FUNC == distance_func::L2) {
    return l2_distance<DATA_T, DISTANCE_T, TILE_SIZE>::compute(a, b, dim, tile);
  } else if constexpr (FUNC == distance_func::INNER_PRODUCT) {
    return inner_product_distance<DATA_T, DISTANCE_T, TILE_SIZE>::compute(a, b, dim, tile);
  } else {
    static_assert(
      FUNC == distance_func::L2 || FUNC == distance_func::INNER_PRODUCT, 
      "Unsupported distance function"
    );
  }
}

}