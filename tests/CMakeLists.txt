# Fetch GoogleTest via CPM (already included in root)
CPMAddPackage(
  NAME googletest
  GITHUB_REPOSITORY google/googletest
  VERSION 1.14.0
  OPTIONS "INSTALL_GTEST OFF"
)

# Each test file is a separate .cu file
set(TEST_SOURCES
  test_distance.cu
  test_rotation.cu
  test_rabitq.cu
  #test_beam_search.cu
  #test_config.cu
)

add_executable(jasper_tests ${TEST_SOURCES})

target_link_libraries(jasper_tests
  PRIVATE
    jasper             # your header-only library
    GTest::gtest
    GTest::gtest_main  # provides main()
    CUDA::cublas
    CUDA::curand
)

target_include_directories(jasper_tests
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}  # for helpers.cuh
)

# Let CMake treat .cu files as CUDA
set_target_properties(jasper_tests PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_STANDARD 17
)

# Register with ctest
include(GoogleTest)
gtest_discover_tests(jasper_tests)