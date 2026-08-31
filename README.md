# nn

A small C++23 tensor and autograd library with CPU and CUDA backends, built from scratch.

Tensors, reverse-mode autodiff, a handful of neural network modules, an optimizer, and a checkpoint format.

---

## Contents

- [Features](#features)
- [Requirements](#requirements)
- [Backends](#backends)
- [Building](#building)
  - [Tests](#tests)
  - [Examples](#examples)
  - [macOS](#macos)
- [Project layout](#project-layout)

## Features

- Strided tensor core with broadcasting, views, and both CPU and CUDA storage
- Reverse-mode autograd with a recorded tape
- Modules: `Linear`, `Embedding`, `LayerNorm`, `RMSNorm`, `Dropout`, `MultiHeadAttention`, `Sequential`, `ReLu`, `GeLu`
- AdamW optimizer with lazy moment allocation
- Checkpoint save and load
- Portable CPU and CUDA kernel backends, selectable at build time

## Requirements

| Requirement | Notes |
| --- | --- |
| CMake | 3.20+ |
| Compiler | A C++23 compiler (Clang is used below; MSVC and GCC also work) |
| CUDA Toolkit | Only if building the CUDA backend |

## Backends

The CUDA backend is optional and gated on `NN_WITH_CUDA`. By default it follows whether `nvcc` is found: present means the CUDA backend is built, absent means a CPU-only build against the CPU kernels. Pass `-DNN_WITH_CUDA=OFF` to force a CPU-only build on a machine that does have CUDA, which is a quick way to exercise the CPU kernels in isolation.

In a CPU-only build, every `Device::CUDA` entry point throws and `cuda_device_count()` returns 0, so anything that picks a device automatically (`devices()` in the tests, the examples) falls back to CPU on its own.

## Building

### Tests

```bash
cmake -S . -B build -G Ninja -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_STANDARD=23
cmake --build build
ctest --test-dir build --output-on-failure
```

On non-Windows platforms, Debug builds also enable `-fsanitize=address,undefined`.

### Examples

```bash
cmake -S . -B build-rel -G Ninja -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=23
cmake --build build-rel --target example_mnist
./build-rel/example_mnist
```

Build examples in Release. Debug plus sanitizers makes the CPU backend's GEMM too slow to train with.

### macOS

CPU-only is the only option: NVIDIA's last macOS toolkit was CUDA 10.2, and Apple Silicon has no NVIDIA GPU. Ninja is optional; the commands below use the default generator instead.

```bash
cmake -S . -B build-mac -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Debug
cmake --build build-mac -j8 && ctest --test-dir build-mac --output-on-failure
```

```bash
cmake -S . -B build-rel -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Release
cmake --build build-rel --target example_mnist -j8 && ./build-rel/example_mnist
```

## Project layout

```text
include/nn/     public headers (core, autograd, ops, module, optim, io, data)
src/            implementation, including the CPU and CUDA kernel backends
tests/          ctest-driven unit tests
examples/       MNIST classifier and a small toy language model
```
