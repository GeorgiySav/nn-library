# NN Library

## To build for tests

```bash
cmake -S . -B build-win -G Ninja -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_STANDARD=23
cmake --build build-win & ctest --test-dir build-win --output-on-failure
```

## To build for examples

```bash
cmake -S . -B build-rel -G Ninja -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=23
cmake --build build-rel --target example_mnist
.\build-rel\example_mnist.exe
```

## benchmarking

```bash
cmake -S . -B build-rel -G Ninja -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Release
cmake --build build-rel --target bench_gemm bench_tape
.\build-rel\bench_gemm.exe
.\build-rel\bench_tape.exe
```
