#include "cuda_kernels.h"

#include "../../cuda_common.h"

namespace nn::kernels {

template<bool transA, bool transB>
__global__ void gemm_kernel(const float* __restrict__ A, const float* __restrict__ B,
                            float* __restrict__ C, int M, int N, int K) {
  const int64_t n = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t m = int64_t(blockIdx.y) * blockDim.y + threadIdx.y;
  if (m >= M || n >= N) return;

  const float* a = A + (transA ? m : m * K);
  const float* b = B + (transB ? n * K : n);
  const int64_t a_stride = transA ? M : 1;
  const int64_t b_stride = transB ? 1 : N;

  float acc = 0.0f;
  for (int k = 0; k < K; ++k) {
    acc = fmaf(a[k * a_stride], b[k * b_stride], acc);
  }
  C[m * N + n] = acc;
}

void cuda_gemm(const Stream& s, const float* A, const float* B, float* C,
               int M, int N, int K, bool transA, bool transB) {
  if (M == 0 || N == 0) return;

  auto stream = static_cast<cudaStream_t>(s.handle);
  const dim3 block(32, 8); // 256 threads
  const dim3 grid((N + block.x - 1) / block.x,
                  (M + block.y - 1) / block.y);
  
  if      (!transA && !transB) gemm_kernel<false, false><<<grid, block, 0, stream>>>(A, B, C, M, N, K);
  else if ( transA && !transB) gemm_kernel< true, false><<<grid, block, 0, stream>>>(A, B, C, M, N, K);
  else if (!transA &&  transB) gemm_kernel<false,  true><<<grid, block, 0, stream>>>(A, B, C, M, N, K);
  else if ( transA &&  transB) gemm_kernel< true,  true><<<grid, block, 0, stream>>>(A, B, C, M, N, K);

  NN_CUDA_CHECK_LAUNCH(stream);
}

}