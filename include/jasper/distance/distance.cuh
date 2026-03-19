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
// template <typename DATA_T, typename DISTANCE_T, uint32_t TILE_SIZE>
// struct l2_distance {
//   template <typename TileT>
//   __device__ static DISTANCE_T compute(const DATA_T* a, const DATA_T* b, const uint32_t dim, TileT& tile) {
//     DISTANCE_T sum = 0.0f;
//     for (uint i = tile.thread_rank(); i < dim; i += TILE_SIZE) {
//       DISTANCE_T diff = static_cast<DISTANCE_T>(a[i]) - static_cast<DISTANCE_T>(b[i]);
//       sum += diff * diff;
//     }
//     sum = cg::reduce(tile, sum, cg::plus<DISTANCE_T>());
//     return sum;
//   }
// };
// L2 squared distance
template <typename DATA_T, typename DISTANCE_T, uint32_t TILE_SIZE>
struct l2_distance {
  template <uint32_t loads_per_round = 2, typename TileT>
  __device__ static DISTANCE_T compute(const DATA_T* a, const DATA_T* b, const uint32_t dim, TileT& tile) {
    DISTANCE_T local_sum = 0.0;

    const uint4* l_ptr = reinterpret_cast<const uint4*>(a);
    const uint4* r_ptr = reinterpret_cast<const uint4*>(b);

    // Alignment is guaranteed by vector_view's padded stride
    // gpu_assert((reinterpret_cast<uintptr_t>(l_ptr) % 16) == 0,
    //            "Left vector not 16-byte aligned\n");
    // gpu_assert((reinterpret_cast<uintptr_t>(r_ptr) % 16) == 0,
    //            "Right vector not 16-byte aligned\n");

    // Process elements using uint4 (16 bytes at a time)
    // Use the padded dim — pad elements are zero so they don't affect the sum
    constexpr int n_element_per_uint4 = 16 / sizeof(DATA_T);
    const int total_loads = dim / n_element_per_uint4;
    const int load_rounds = (total_loads - 1) / (loads_per_round * TILE_SIZE) + 1;

    uint4 l_data[loads_per_round];
    uint4 r_data[loads_per_round];

    for (uint i = 0; i < load_rounds; i += 1) {

      for (uint j = 0; j < loads_per_round; j++) {
        int index = tile.thread_rank() + j * TILE_SIZE + i * TILE_SIZE * loads_per_round;
        if (index < total_loads) {
          l_data[j] = l_ptr[index];
        }
      }

      for (uint j = 0; j < loads_per_round; j++) {
        int index = tile.thread_rank() + j * TILE_SIZE + i * TILE_SIZE * loads_per_round;
        if (index < total_loads) {
          r_data[j] = r_ptr[index];
        }
      }

      for (uint j = 0; j < loads_per_round; j++) {
        int index = tile.thread_rank() + j * TILE_SIZE + i * TILE_SIZE * loads_per_round;
        if (index < total_loads) {
          DATA_T* l_bytes = (DATA_T*)&l_data[j];
          DATA_T* r_bytes = (DATA_T*)&r_data[j];
          for (int k = 0; k < n_element_per_uint4; k++) {
            DISTANCE_T diff = static_cast<DISTANCE_T>(l_bytes[k]) -
                              static_cast<DISTANCE_T>(r_bytes[k]);
            local_sum += diff * diff;
          }
        }
      }
    }

    // Use cooperative groups to reduce across threads
    DISTANCE_T total_sum = cg::reduce(tile, local_sum, cg::plus<DISTANCE_T>());

    // Final safety check
    // gpu_assert(!is_bad(total_sum), "Result is corrupted\n");

    return total_sum;
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