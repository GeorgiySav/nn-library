# NN Library

## To build

```bash
cmake -S . -B build-win -G Ninja -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Debug
cmake --build build-win & ctest --test-dir build-win --output-on-failure
```

