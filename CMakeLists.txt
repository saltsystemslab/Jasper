cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(Jasper LANGUAGES CXX CUDA)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

include("cmake/CPM.cmake")

find_package(CUDAToolkit REQUIRED)

find_package(Thrust REQUIRED CONFIG)
thrust_create_target(ThrustInterface)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_COMPILER $ENV{CC})
set(CMAKE_CXX_COMPILER $ENV{CXX})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}") 
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -rdc=true -lcudadevrt -lcudart -lineinfo")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
  message("CMAKE in debug mode. Appending CUDA Debug flags.")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -g -lineinfo")
endif()

CPMAddPackage(
  NAME gallatin
  GITHUB_REPOSITORY saltsystemslab/gallatin
  GIT_TAG origin/inlined_commands
)

CPMAddPackage (
  NAME argparse
  GITHUB_REPOSITORY p-ranav/argparse
  GIT_TAG origin/master
)

# ── Feature flags ──
# Stable-id "label" layer (WarpSpeed-backed id<->slot map) and graph compaction
# are ON by default: every vector gets a stable id, deletion runs by id, and
# compact() reclaims tombstoned slots. The single-writer concurrency contract
# (search shared / mutation exclusive) is enforced by a host-side lock
# (graph<>::rw_mutex). Set either OFF to opt back out (slot==id, no WarpSpeed
# dependency; deletion stays mark_deleted + consolidate without slot reclaim).
option(JASPER_STABLE_IDS    "Enable WarpSpeed-backed stable-id labels" ON)
option(JASPER_ENABLE_COMPACT "Enable graph compaction (compact())"     ON)

# WarpSpeed GPU hash tables — only needed when the stable-id layer is enabled.
if(JASPER_STABLE_IDS)
  CPMAddPackage(
    NAME warpSpeed
    GITHUB_REPOSITORY saltsystemslab/warpSpeed
    GIT_TAG main
    DOWNLOAD_ONLY YES
    GIT_SUBMODULES ""      # benchmark deps (BGHT/SlabHash/warpcore) — not needed
  )
endif()

# gpu_error is an optional submodule (debug logging helpers). jasper's own
# sources don't depend on it, so only add it when the submodule is checked out.
if(EXISTS "${PROJECT_SOURCE_DIR}/external/gpu_error/CMakeLists.txt")
  add_subdirectory(external/gpu_error)
endif()

# ── Header-only library ──
add_library(jasper INTERFACE)
target_include_directories(jasper
  INTERFACE
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
# Propagate the feature flags as compile definitions.
target_compile_definitions(jasper INTERFACE
  JASPER_STABLE_IDS=$<BOOL:${JASPER_STABLE_IDS}>
  JASPER_ENABLE_COMPACT=$<BOOL:${JASPER_ENABLE_COMPACT}>
)
if(JASPER_STABLE_IDS)
  target_include_directories(jasper INTERFACE
    $<BUILD_INTERFACE:${warpSpeed_SOURCE_DIR}/include>)
endif()
target_compile_features(jasper INTERFACE cxx_std_17)
target_compile_options(jasper
  INTERFACE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda --expt-relaxed-constexpr>
)
target_link_libraries(jasper
  INTERFACE
    gallatin
    argparse
    CUDA::cudart
    CUDA::toolkit
    CUDA::cublas 
    CUDA::curand
    ThrustInterface
)

# Testing
option(JASPER_BUILD_TESTS "Build unit tests" OFF)
if(JASPER_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

# CLI tool directly calling c++
option(JASPER_BUILD_CMD "Build command-line tools" ON)
if(JASPER_BUILD_CMD)
  add_executable(create_index cmd/create_index.cu)
  target_link_libraries(create_index PRIVATE jasper)
  set_target_properties(create_index PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  )
  set_target_properties(create_index PROPERTIES
    CUDA_ARCHITECTURES "89;90;120"
    CUDA_SEPARABLE_COMPILATION ON
  )

  add_executable(sweep_batch_size cmd/sweep_batch_size.cu)
  target_link_libraries(sweep_batch_size PRIVATE jasper)
  set_target_properties(sweep_batch_size PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  )
  set_target_properties(sweep_batch_size PROPERTIES
    CUDA_ARCHITECTURES "80;89;90;120"
    CUDA_SEPARABLE_COMPILATION ON
  )

  add_executable(create_scale_index cmd/create_scale_index.cu)
  target_link_libraries(create_scale_index PRIVATE jasper)
  set_target_properties(create_scale_index PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  )
  set_target_properties(create_scale_index PROPERTIES
    CUDA_ARCHITECTURES "89;90;120"
    CUDA_SEPARABLE_COMPILATION ON
  )

  add_executable(create_lsh_index cmd/create_lsh_index.cu)
  target_link_libraries(create_lsh_index PRIVATE jasper)
  set_target_properties(create_lsh_index PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  )
  set_target_properties(create_lsh_index PROPERTIES
    CUDA_ARCHITECTURES "80;120"
    CUDA_SEPARABLE_COMPILATION ON
  )
  # target_compile_definitions(create_lsh_index PRIVATE JASPER_PROFILE_CLOCKS)

  add_executable(create_lsh_index_block_tuning cmd/create_lsh_index_block_tuning.cu)
  target_link_libraries(create_lsh_index_block_tuning PRIVATE jasper)
  set_target_properties(create_lsh_index_block_tuning PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  )
  set_target_properties(create_lsh_index_block_tuning PROPERTIES
    CUDA_ARCHITECTURES "120"
    CUDA_SEPARABLE_COMPILATION ON
  )

  add_executable(run_query cmd/run_query.cu)
  target_link_libraries(run_query PRIVATE jasper)
  set_target_properties(run_query PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  )
  set_target_properties(run_query PROPERTIES
    CUDA_ARCHITECTURES "89;90;120"
    CUDA_SEPARABLE_COMPILATION ON
  )

  add_executable(test_deletion cmd/test_deletion.cu)
  target_link_libraries(test_deletion PRIVATE jasper)
  set_target_properties(test_deletion PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  )
  set_target_properties(test_deletion PROPERTIES
    CUDA_ARCHITECTURES "89;90;120"
    CUDA_SEPARABLE_COMPILATION ON
  )

  # test_idmap exercises the WarpSpeed table API directly, so it only builds
  # when the stable-id layer is enabled.
  if(JASPER_STABLE_IDS)
    add_executable(test_idmap cmd/test_idmap.cu)
    target_link_libraries(test_idmap PRIVATE jasper)
    set_target_properties(test_idmap PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    )
    set_target_properties(test_idmap PROPERTIES
      CUDA_ARCHITECTURES "89;90;120"
      CUDA_SEPARABLE_COMPILATION ON
    )
  endif()

  add_executable(test_append cmd/test_append.cu)
  target_link_libraries(test_append PRIVATE jasper)
  set_target_properties(test_append PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  )
  set_target_properties(test_append PROPERTIES
    CUDA_ARCHITECTURES "89;90;120"
    CUDA_SEPARABLE_COMPILATION ON
  )

  add_executable(bench_all cmd/bench_all.cu)
  target_link_libraries(bench_all PRIVATE jasper)
  set_target_properties(bench_all PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
  )
  set_target_properties(bench_all PROPERTIES
    CUDA_ARCHITECTURES "89;90;120"
    CUDA_SEPARABLE_COMPILATION ON
  )
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
  target_link_libraries(jasper_ffi PRIVATE 
    jasper 
    CUDA::cublas
    CUDA::curand
    CUDA::cusolver
    CUDA::cusparse
  )

  # tvm_ffi_configure_target sets up:
  #   - include paths for tvm/ffi headers
  #   - link against libtvm_ffi
  #   - visibility flags so TVM_FFI_DLL_EXPORT_TYPED_FUNC works
  tvm_ffi_configure_target(jasper_ffi)

  # Set CUDA arch — adjust to your target GPU
  set_target_properties(jasper_ffi PROPERTIES
    CUDA_ARCHITECTURES "80;89;90;120"
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