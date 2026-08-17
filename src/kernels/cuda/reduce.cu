#include "cuda_kernels.h"

#include "../../cuda_common.h"

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

void cuda_add_row_bias(const Stream& s, const float* X, const float* b, float* Y, int M, int N) {
  if (M == 0 || N == 0) return;

  auto stream = static_cast<cudaStream_t>(s.handle);
  constexpr dim3 block(256);
  constexpr int kMaxGrid = 4096;
  dim3 grid(std::ceil((N+255) / 256), std::min(M, kMaxGrid));

  add_row_bias_kernel<<<grid, block, 0, stream>>>(X, b, Y, M, N);
  NN_CUDA_CHECK_LAUNCH(stream);
}

}