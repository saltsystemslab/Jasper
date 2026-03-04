Compile library
```bash
# Install tvm_ffi first
pip install apache-tvm-ffi

# Build the FFI shared library
cmake -B build -DJASPER_BUILD_FFI=ON
cmake --build build
cmake --install build  # copies libjasper_ffi.so → python/jasper/lib/

# Install the Python package
pip install -e python/
```

Run tests
```bash
cmake -B build -DJASPER_BUILD_TESTS=ON
cmake --build build
cd build && ctest --output-on-failure
```