cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(Jasper LANGUAGES CXX CUDA)

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
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
)
target_link_libraries(jasper
  INTERFACE
    CUDA::cudart
    CUDA::cudadevrt
)

option(JASPER_BUILD_TESTS "Build unit tests" OFF)
if(JASPER_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()