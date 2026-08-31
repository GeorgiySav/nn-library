#include "cuda_kernels.h"

#include <kernels/kernel_api.h>   // kSumAllWorkspace
#include <kernels/random.h>

#include "../../cuda_common.h"
#include "../strided_index.h"
#include "cuda_reduce.cuh"

namespace nn::kernels {

__global__ void argmax_rows_kernel(const float* X, int32_t* out, int M, int N, int64_t sx) {
  const ArgMax kIdentity{-FLT_MAX, INT_MAX};

  for (int64_t row = blockIdx.x; row < M; row += gridDim.x) {
    const float* r = X + int64_t(row) * sx;

    ArgMax best = kIdentity;
    for (int c = threadIdx.x; c < N; c += blockDim.x) {
      if (r[c] > best.value) best = ArgMax{r[c], c};
    }

    best = block_reduce(best, MaxKeepLowestIndex(), kIdentity);

    if (threadIdx.x == 0) out[row] = best.index;
    // block_reduce's shared scratch buffer is reused every row this block
    // handles, so all threads must finish reading it before the next iteration
    __syncthreads();
  }
}

void cuda_argmax_rows(const Stream& s, const float* X, int32_t* out, int M, int N, int64_t sx) {
  if (M == 0 || N == 0) return;

  auto stream = static_cast<cudaStream_t>(s.handle);
  constexpr int block{256};
  constexpr int kMaxGrid{4096};
  int grid = std::min(M, kMaxGrid);

  argmax_rows_kernel<<<grid, block, 0, stream>>>(X, out, M, N, sx);
  NN_CUDA_CHECK_LAUNCH(stream);
}

__global__ void topk_rows_kernel(const float* X, int M, int N, int k,
                              float* values, int32_t* indices, int64_t sx) {
  extern __shared__ int excluded[];  // k entries, indices already taken this row
  const ArgMax kIdentity{-FLT_MAX, INT_MAX};

  for (int64_t row = blockIdx.x; row < M; row += gridDim.x) {
    const float* r = X + int64_t(row) * sx;

    for (int iter = 0; iter < k; ++iter) {
      ArgMax best = kIdentity;
      for (int c = threadIdx.x; c < N; c += blockDim.x) {
        bool taken = false;
        for (int e = 0; e < iter; ++e) {
          if (excluded[e] == c) { taken = true; break; }
        }
        if (!taken && r[c] > best.value) best = ArgMax{r[c], c};
      }

      best = block_reduce(best, MaxKeepLowestIndex(), kIdentity);

      if (threadIdx.x == 0) {
        values[row * k + iter] = best.value;
        indices[row * k + iter] = best.index;
        excluded[iter] = best.index;
      }
      // every thread needs to see excluded[iter] before scanning again next iteration
      __syncthreads();
    }
  }
}

void cuda_topk_rows(const Stream& s, const float* X, int M, int N, int k,
                    float* values, int32_t* indices, int64_t sx) {
  if (M == 0 || N == 0 || k == 0) return;

  auto stream = static_cast<cudaStream_t>(s.handle);
  constexpr int block{256};
  constexpr int kMaxGrid{4096};
  int grid = std::min(M, kMaxGrid);
  const size_t shared_bytes = size_t(k) * sizeof(int);

  topk_rows_kernel<<<grid, block, shared_bytes, stream>>>(X, M, N, k, values, indices, sx);
  NN_CUDA_CHECK_LAUNCH(stream);
}

namespace {

// idx is range-checked by ops::gather_rows before this ever runs, so the
// kernel itself just reads -- one thread per row, no block cooperation needed.
template <class T>
__global__ void gather_rows_kernel(const T* __restrict__ src, const int32_t* __restrict__ idx,
                                   T* __restrict__ out, int64_t M, int64_t sx) {
  for (int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x; i < M;
       i += int64_t(gridDim.x) * blockDim.x) {
    out[i] = src[i * sx + idx[i]];
  }
}

template <class T>
void launch_gather_rows(const Stream& s, const T* src, const int32_t* idx, T* out,
                        int64_t M, int64_t sx) {
  if (M == 0) return;
  auto stream = static_cast<cudaStream_t>(s.handle);
  constexpr int block = 256;
  constexpr int kMaxGrid = 4096;
  const int grid = int(std::min<int64_t>((M + block - 1) / block, kMaxGrid));
  gather_rows_kernel<T><<<grid, block, 0, stream>>>(src, idx, out, M, sx);
  NN_CUDA_CHECK_LAUNCH(stream);
}

}  // namespace

void cuda_gather_rows(const Stream& s, const float* src, const int32_t* idx,
                      float* out, int M, int64_t sx) {
  launch_gather_rows(s, src, idx, out, int64_t(M), sx);
}

void cuda_gather_rows_i32(const Stream& s, const int32_t* src, const int32_t* idx,
                          int32_t* out, int M, int64_t sx) {
  launch_gather_rows(s, src, idx, out, int64_t(M), sx);
}

// total <= 0 (a caller passing an all-zero row) has no host-side check to
// fall back on here, unlike cpu_multinomial -- a device kernel cannot
// throw, so it clamps to the last column instead of reading garbage.
__global__ void multinomial_kernel(const float* __restrict__ W, int32_t* __restrict__ out,
                                   int M, int N, int64_t sx, uint64_t seed, uint64_t offset) {
  for (int64_t row = blockIdx.x * int64_t(blockDim.x) + threadIdx.x; row < M;
       row += int64_t(gridDim.x) * blockDim.x) {
    const float* r = W + row * sx;
    float total = 0.0f;
    for (int j = 0; j < N; ++j) total += r[j];

    const float target = random_uniform(seed, offset + uint64_t(row)) *
                          (total > 0.0f ? total : 1.0f);
    float cum = 0.0f;
    int chosen = N - 1;
    for (int j = 0; j < N; ++j) {
      cum += r[j];
      if (target < cum) { chosen = j; break; }
    }
    out[row] = chosen;
  }
}

void cuda_multinomial(const Stream& s, const float* W, int32_t* out, int M, int N, int64_t sx,
                      uint64_t seed, uint64_t offset) {
  if (M == 0 || N == 0) return;

  auto stream = static_cast<cudaStream_t>(s.handle);
  constexpr int block = 256;
  constexpr int kMaxGrid = 4096;
  const int grid = int(std::min<int64_t>((int64_t(M) + block - 1) / block, kMaxGrid));
  multinomial_kernel<<<grid, block, 0, stream>>>(W, out, M, N, sx, seed, offset);
  NN_CUDA_CHECK_LAUNCH(stream);
}

namespace {

constexpr int kSumBlock = 256;

__global__ void sum_partials_kernel(const float* __restrict__ X, TensorView v,
                                    Accum a, float* partials, int64_t n) {
  float acc = 0.0f;
  for (int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
       i < n;
       i += int64_t(gridDim.x) * blockDim.x) {
    acc += apply_accum(a, X[offset_of(v, i)]);
  }
  acc = block_reduce(acc, Plus(), 0.0f);
  if (threadIdx.x == 0) partials[blockIdx.x] = acc;
}

// single block that folds the per-block partial sums from sum_partials_kernel
// down to the final scalar
__global__ void sum_finish_kernel(const float* __restrict__ partials,
                                  float* out, int m) {
  float acc = 0.0f;
  for (int i = threadIdx.x; i < m; i += blockDim.x) acc += partials[i];
  acc = block_reduce(acc, Plus(), 0.0f);
  if (threadIdx.x == 0) *out = acc;
}

}  // namespace

__global__ void sum_to_kernel(const float* __restrict__ g, TensorView keep,
                              TensorView red, float* __restrict__ out,
                              int64_t n_out, int64_t n_red) {
  for (int64_t j = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
       j < n_out;
       j += int64_t(gridDim.x) * blockDim.x) {
    const int64_t base = offset_of(keep, j);
    float acc = 0.0f;
    for (int64_t k = 0; k < n_red; ++k) acc += g[base + offset_of(red, k)];
    out[j] = acc;
  }
}

void cuda_sum_to(const Stream& s, const float* g, TensorView keep, TensorView red,
                 float* out, int64_t n_out, int64_t n_red) {
  if (n_out == 0) return;
  auto stream = static_cast<cudaStream_t>(s.handle);
  constexpr int block = 256;
  constexpr int kMaxGrid = 4096;
  const int grid = int(std::min<int64_t>((n_out + block - 1) / block, kMaxGrid));
  sum_to_kernel<<<grid, block, 0, stream>>>(g, keep, red, out, n_out, n_red);
  NN_CUDA_CHECK_LAUNCH(stream);
}

// Two kernels over a fixed grid rather than one with atomicAdd
void cuda_sum_all(const Stream& s, const float* X, TensorView v, Accum a,
                  float* out, float* workspace, int64_t n) {
  auto stream = static_cast<cudaStream_t>(s.handle);
  if (n == 0) {
    NN_CUDA_CHECK(cudaMemsetAsync(out, 0, sizeof(float), stream));
    return;
  }

  sum_partials_kernel<<<kSumAllWorkspace, kSumBlock, 0, stream>>>(X, v, a, workspace, n);
  sum_finish_kernel<<<1, 1024, 0, stream>>>(workspace, out, kSumAllWorkspace);
  NN_CUDA_CHECK_LAUNCH(stream);
}

}
