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

## sum_all accumulators

`SumAllFn` takes an `Accum` and switches on it per element, so a norm is one
pass over the gradient with no temporary. The question is whether the switch
costs anything on the plain-sum path everything else uses. A reduction reads
4 B/element -- half a map's traffic -- so it has the least memory time to hide
work behind, and is where a branch would show up if anywhere.

`bench_accum`, RTX 5080 / Release:

```bash
device     n         Sum(ns)      SumSq     SumAbs   sq/sum
cpu        1     M     461300     466100     465900    1.010
cpu        16    M    7412900    7448800    7459300    1.005
cpu        64    M   29845800   30006000   30152700    1.005
cuda       1     M      14500      14400      14300    0.993
cuda       16    M      45000      45000      45000    1.000
cuda       64    M     289900     286700     288700    0.989
```

0.989-1.010x, i.e. noise. The branch is uniform across the warp and the loop is
memory bound, the same reason the elementwise op code is free.
