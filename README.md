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

## TODOs

- Pinned memory

8. Token dataset — Dataset is [N, D] float features with targets; an LM wants a token stream sliced into (x, y) shifted windows. And sampling for generation (argmax_rows is rank-2 and greedy-only).

Worth knowing before it hurts
10. unary_backward holds x and y, so every elementwise op in the graph pins two activation buffers. On a transformer that's a real memory multiplier, and the [B,T,V] logits are already the largest tensor in the model. The .def records which of x/y each derivative actually needs — that information is there, just not acted on yet