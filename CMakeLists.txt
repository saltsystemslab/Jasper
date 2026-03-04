cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(Jasper LANGUAGES CXX CUDA)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

include("cmake/CPM.cmake")

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_COMPILER $ENV{CC})
set(CMAKE_CXX_COMPILER $ENV{CXX})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}") 
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -rdc=true -lcudadevrt -lcudart")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  message("CMAKE in debug mode. Appending CUDA Debug flags.")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -g -lineinfo")
endif()

# ── Header-only library ──
add_library(jasper INTERFACE)
target_include_directories(jasper
  INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(jasper INTERFACE cxx_std_17)
target_compile_options(jasper
  INTERFACE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda --expt-relaxed-constexpr>
)
target_link_libraries(jasper
  INTERFACE
    CUDA::cudart
)

# Testing
option(JASPER_BUILD_TESTS "Build unit tests" OFF)
if(JASPER_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

# ── TVM FFI shared library ──────────────────────────────────────
option(JASPER_BUILD_FFI "Build TVM FFI Python bindings" OFF)

if(JASPER_BUILD_FFI)
  # Find tvm_ffi via its CMake config shipped with the pip package
  find_package(Python COMPONENTS Interpreter REQUIRED)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m tvm_ffi.config --cmakedir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE tvm_ffi_ROOT
  )
  find_package(tvm_ffi CONFIG REQUIRED)

  # Build the .so
  add_library(jasper_ffi SHARED ffi/jasper_ffi.cu)
  target_link_libraries(jasper_ffi PRIVATE jasper)

  # tvm_ffi_configure_target sets up:
  #   - include paths for tvm/ffi headers
  #   - link against libtvm_ffi
  #   - visibility flags so TVM_FFI_DLL_EXPORT_TYPED_FUNC works
  tvm_ffi_configure_target(jasper_ffi)

  # Set CUDA arch — adjust to your target GPU
  set_target_properties(jasper_ffi PROPERTIES
    CUDA_ARCHITECTURES "80;86;89;90"
    CUDA_SEPARABLE_COMPILATION ON
  )

  # Debug: add line info for compute-sanitizer
  target_compile_options(jasper_ffi PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>: -lineinfo>
  )

  # Install into python/jasper/lib/ so the Python package can find it
  install(TARGETS jasper_ffi
    LIBRARY DESTINATION ${PROJECT_SOURCE_DIR}/python/jasper/lib
  )
endif()