# Results

## GEMM

### Naive

```bash
fc1 forward                4733.4 us      2.71 GFLOP/s
fc1 grad-input             3661.0 us      3.51 GFLOP/s
fc1 grad-w                 2529.9 us      5.08 GFLOP/s
square 256                11399.4 us      2.94 GFLOP/s
square 512               150448.7 us      1.78 GFLOP/s
```

### Fix inner loop (stride 1)

```bash
fc1 forward                 164.4 us     78.13 GFLOP/s
fc1 grad-input             2478.8 us      5.18 GFLOP/s
fc1 grad-w                  149.3 us     86.04 GFLOP/s
square 256                  338.2 us     99.21 GFLOP/s
square 512                 3410.2 us     78.72 GFLOP/s
```

### reassociate

```bash
fc1 forward                 164.2 us     78.23 GFLOP/s
fc1 grad-input              104.4 us    123.04 GFLOP/s
fc1 grad-w                  156.0 us     82.34 GFLOP/s
square 256                  338.2 us     99.21 GFLOP/s
square 512                 3242.3 us     82.79 GFLOP/s
```
