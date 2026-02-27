#pragma once

namespace jasper {

template <typename DATA_T, uint DATA_DIM>
struct __align__(16) vector {
  DATA_T data[DATA_DIM];

  // self populate based on ptr to raw data
  __host__ vector(data_type* ptr_to_data) {
    for (uint i = 0; i < DATA_DIM; i++) {
      data[i] = ptr_to_data[i];
    }
  }

  // Unified default constructor
  __host__ __device__ vector() {
  #ifdef __CUDA_ARCH__
      // Device: do nothing
  #else
      // Host: zero-init
      for (uint i = 0; i < DATA_DIM; i++) {
          data[i] = static_cast<DATA_T>(0);
      }
  #endif
  }

  __host__ __device__ DATA_T& operator[](uint index) { return data[index]; }

  __host__ __device__ const DATA_T& operator[](uint index) const {
    return data[index];
  }

  // print the vector to std::cout
  // requires vector to be on host.
  __host__ void print() {
    std::cout << "Vector: [";
    for (uint i = 0; i < DATA_DIM; i++) {
      std::cout << +data[i];
      if (i != DATA_DIM - 1) {
        std::cout << " ";
      }
    }
    std::cout << "]" << std::endl;
  }
};

// load from a file and populate n_vectors.
// data returned is pinned host memory.
template <typename DATA_T, uint DATA_DIM>
__host__ static vector<DATA_T, DATA_DIM>* load_from_file(std::string filename,
                                            uint64_t& n_vectors) {
  std::ifstream file(filename, std::ios::in | std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Error opening file: " << filename << std::endl;
    return nullptr;
  }
  file.seekg(0, std::ios::end);  // go to end of file
  std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);

  size = size - 8;

  int n_data_points;
  int n_dimensions;
  file.read(reinterpret_cast<char*>(&n_data_points), 4);
  file.read(reinterpret_cast<char*>(&n_dimensions), 4);

  std::cout << "Read " << size << " bytes of data" << std::endl;

  std::cout << "Read value is " << n_data_points << " with dim "
            << n_dimensions << std::endl;

  if (sizeof(data_type) * n_data_points * n_dimensions != size) {
    std::cerr << "DIM mismatch: "
              << " sizeof(data_type)=" << sizeof(data_type)
              << " n_data_points=" << n_data_points
              << " n_dimensions=" << n_dimensions
              << sizeof(data_type) * n_data_points * n_dimensions
              << " != " << size << std::endl;
    return nullptr;
  } else {
    std::cout << "Dimensional calcs match" << std::endl;
  }

  if (size % sizeof(data_type) != 0) {
    std::cerr << "File size is not a multiple of data_type with size "
              << sizeof(data_type) << std::endl;
    return nullptr;
  }

  n_vectors = size / (n_degrees * sizeof(data_type));

  if ((size % n_degrees) != 0) {
    std::cerr << "vector stride does not align with data: "
              << size % n_degrees << " over.\n";
  }

  printf("Size of vector type is %lu, loading %lu vectors\n",
         sizeof(vector_type), n_vectors);
  vector_type* data =
      gallatin::utils::get_host_version<vector_type>(n_vectors);

  if (file.read(reinterpret_cast<char*>(data), size)) {
    std::cout << "Successfully read " << n_vectors << " vectors from "
              << filename << std::endl;
  } else {
    std::cerr << "Error while reading the file." << std::endl;
  }

  return data;
}

}