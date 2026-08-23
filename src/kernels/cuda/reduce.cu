#include "cuda_kernels.h"

#include <nn/kernels/kernel_api.h>   // kSumAllWorkspace

#include "../../cuda_common.h"
#include "../strided_index.h"
#include "cuda_reduce.cuh"

namespace nn::kernels {

__global__ void add_row_bias_kernel(const float* X, const float* b, float* Y, int M, int N, int64_t sx) {
  for (int64_t n = blockIdx.x * blockDim.x + threadIdx.x;
       n < N;
       n += int64_t(gridDim.x) * blockDim.x) {
    float bias = b[n];
    for (int64_t m = blockIdx.y * blockDim.y + threadIdx.y;
         m < M;
         m += int64_t(gridDim.y) * blockDim.y) {
      Y[m*N + n] = X[m*sx + n] + bias;
    }
  }
}

__global__ void col_sum_kernel(const float* X, float* out, int M, int N, int64_t sx) {
  for (int64_t col = blockIdx.x * blockDim.x + threadIdx.x;
       col < N;
       col += int64_t(gridDim.x) * blockDim.x) {
    float sum = 0;
    for (int64_t row = 0; row < M; ++row) {
      sum += X[row*sx + col];
    }
    out[col] = sum;
  }
}

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
    __syncthreads();
  }
}

void cuda_add_row_bias(const Stream& s, const float* X, const float* b, float* Y, int M, int N, int64_t sx) {
  if (M == 0 || N == 0) return;

  auto stream = static_cast<cudaStream_t>(s.handle);
  constexpr dim3 block(256);
  constexpr int kMaxGrid = 4096;
  dim3 grid((N+255) / 256, std::min(M, kMaxGrid));

  add_row_bias_kernel<<<grid, block, 0, stream>>>(X, b, Y, M, N, sx);
  NN_CUDA_CHECK_LAUNCH(stream);
}

void cuda_col_sum(const Stream& s, const float* X, float* out, int M, int N, int64_t sx) {
  if (M == 0 || N == 0) return;

  auto stream = static_cast<cudaStream_t>(s.handle);
  constexpr int block{256};
  constexpr int kMaxGrid{4096};
  int grid = std::min((N + block - 1) / block, kMaxGrid);

  col_sum_kernel<<<grid, block, 0, stream>>>(X, out, M, N, sx);
  NN_CUDA_CHECK_LAUNCH(stream);
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

namespace {

constexpr int kSumBlock = 256;

__global__ void sum_partials_kernel(const float* __restrict__ X,
                                    float* partials, int64_t n) {
  float acc = 0.0f;
  for (int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
       i < n;
       i += int64_t(gridDim.x) * blockDim.x) {
    acc += X[i];
  }
  acc = block_reduce(acc, Plus(), 0.0f);
  if (threadIdx.x == 0) partials[blockIdx.x] = acc;
}

__global__ void sum_partials_strided_kernel(const float* __restrict__ X, TensorView v,
                                            float* partials, int64_t n) {
  float acc = 0.0f;
  for (int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
       i < n;
       i += int64_t(gridDim.x) * blockDim.x) {
    acc += X[offset_of(v, i)];
  }
  acc = block_reduce(acc, Plus(), 0.0f);
  if (threadIdx.x == 0) partials[blockIdx.x] = acc;
}

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

void cuda_sum_all(const Stream& s, const float* X, float* out,
                  float* workspace, int64_t n) {
  auto stream = static_cast<cudaStream_t>(s.handle);
  if (n == 0) {
    NN_CUDA_CHECK(cudaMemsetAsync(out, 0, sizeof(float), stream));
    return;
  }

  sum_partials_kernel<<<kSumAllWorkspace, kSumBlock, 0, stream>>>(X, workspace, n);
  sum_finish_kernel<<<1, 1024, 0, stream>>>(workspace, out, kSumAllWorkspace);
  NN_CUDA_CHECK_LAUNCH(stream);
}

void cuda_sum_all_strided(const Stream& s, const float* X, TensorView v,
                          float* out, float* workspace, int64_t n) {
  auto stream = static_cast<cudaStream_t>(s.handle);
  if (n == 0) {
    NN_CUDA_CHECK(cudaMemsetAsync(out, 0, sizeof(float), stream));
    return;
  }

  sum_partials_strided_kernel<<<kSumAllWorkspace, kSumBlock, 0, stream>>>(
      X, v, workspace, n);
  sum_finish_kernel<<<1, 1024, 0, stream>>>(workspace, out, kSumAllWorkspace);
  NN_CUDA_CHECK_LAUNCH(stream);
}

}
