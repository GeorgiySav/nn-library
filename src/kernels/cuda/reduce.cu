#include "cuda_kernels.h"

#include "../../cuda_common.h"
#include "cuda_reduce.cuh"

namespace nn::kernels {

__global__ void add_row_bias_kernel(const float* X, const float* b, float* Y, int M, int N) {
  for (int64_t n = blockIdx.x * blockDim.x + threadIdx.x;
       n < N;
       n += int64_t(gridDim.x) * blockDim.x) {
    float bias = b[n];
    for (int64_t m = blockIdx.y * blockDim.y + threadIdx.y;
         m < M;
         m += int64_t(gridDim.y) * blockDim.y) {
      Y[m*N + n] = X[m*N + n] + bias;
    }
  }
}

__global__ void col_sum_kernel(const float* X, float* out, int M, int N) {
  for (int64_t col = blockIdx.x * blockDim.x + threadIdx.x;
       col < N;
       col += int64_t(gridDim.x) * blockDim.x) {
    float sum = 0;
    for (int64_t row = 0; row < M; ++row) {
      sum += X[row*N + col];
    }
    out[col] = sum;
  }
}

__global__ void argmax_rows_kernel(const float* X, int32_t* out, int M, int N) {
  const ArgMax kIdentity{-FLT_MAX, INT_MAX};

  for (int64_t row = blockIdx.x; row < M; row += gridDim.x) {
    const float* r = X + int64_t(row) * N;

    ArgMax best = kIdentity;
    for (int c = threadIdx.x; c < N; c += blockDim.x) {
      if (r[c] > best.value) best = ArgMax{r[c], c};
    }

    best = block_reduce(best, MaxKeepLowestIndex(), kIdentity);

    if (threadIdx.x == 0) out[row] = best.index;
    __syncthreads();
  }
}

void cuda_add_row_bias(const Stream& s, const float* X, const float* b, float* Y, int M, int N) {
  if (M == 0 || N == 0) return;

  auto stream = static_cast<cudaStream_t>(s.handle);
  constexpr dim3 block(256);
  constexpr int kMaxGrid = 4096;
  dim3 grid((N+255) / 256, std::min(M, kMaxGrid));

  add_row_bias_kernel<<<grid, block, 0, stream>>>(X, b, Y, M, N);
  NN_CUDA_CHECK_LAUNCH(stream);
}

void cuda_col_sum(const Stream& s, const float* X, float* out, int M, int N) {
  if (M == 0 || N == 0) return;

  auto stream = static_cast<cudaStream_t>(s.handle);
  constexpr int block{256};
  constexpr int kMaxGrid{4096};
  int grid = std::min((N + block - 1) / block, kMaxGrid);

  col_sum_kernel<<<grid, block, 0, stream>>>(X, out, M, N);
  NN_CUDA_CHECK_LAUNCH(stream);
}

void cuda_argmax_rows(const Stream& s, const float* X, int32_t* out, int M, int N) {
  if (M == 0 || N == 0) return;

  auto stream = static_cast<cudaStream_t>(s.handle);
  constexpr int block{256};
  constexpr int kMaxGrid{4096};
  int grid = std::min(M, kMaxGrid);

  argmax_rows_kernel<<<grid, block, 0, stream>>>(X, out, M, N);
  NN_CUDA_CHECK_LAUNCH(stream);
}

}