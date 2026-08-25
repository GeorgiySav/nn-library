# NN Library

## Backends

The CUDA backend is optional, gated on `NN_WITH_CUDA`. The default follows
`nvcc`: present means the CUDA backend is built, absent means a CPU-only build
against the naive kernels. `-DNN_WITH_CUDA=OFF` forces CPU-only on a CUDA box,
which is a quick way to check the naive kernels in isolation.

In a CPU-only build every `Device::CUDA` entry point throws, and
`cuda_device_count()` returns 0 — so anything that gates on it (`devices()` in
the tests, the benches, the MNIST example) picks CPU on its own.

## macOS

CPU-only is the only option: NVIDIA's last macOS toolkit was CUDA 10.2, and
Apple Silicon has no NVIDIA GPU. Ninja is optional; this uses the default
generator.

```bash
cmake -S . -B build-mac -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Debug
cmake --build build-mac -j8 && ctest --test-dir build-mac --output-on-failure
```

Debug on non-Windows adds `-fsanitize=address,undefined`; the suite is clean
under both. For the example, build Release — Debug plus sanitizers makes the
naive GEMM far too slow to train on.

```bash
cmake -S . -B build-rel -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Release
cmake --build build-rel --target example_mnist -j8 && ./build-rel/example_mnist
```

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

## TODOs

- Pinned memory

8. Token dataset — Dataset is [N, D] float features with targets; an LM wants a token stream sliced into (x, y) shifted windows. And sampling for generation (argmax_rows is rank-2 and greedy-only).

Worth knowing before it hurts
10. unary_backward holds x and y, so every elementwise op in the graph pins two activation buffers. On a transformer that's a real memory multiplier, and the [B,T,V] logits are already the largest tensor in the model. The .def records which of x/y each derivative actually needs — that information is there, just not acted on yet