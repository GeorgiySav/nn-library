#include "cuda_kernels.h"

#include <cublas_v2.h>

#include "../../cuda_common.h"

namespace nn::kernels {

template<bool transA, bool transB>
__global__ void naive_gemm_kernel(const float* __restrict__ A, const float* __restrict__ B,
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

namespace {

constexpr int BM = 64; // rows of C per block
constexpr int BN = 64; // cols of C per block
constexpr int BK = 16; // depth of one tile
constexpr int TM = 4;  // rows of C per thread
constexpr int TN = 4;  // cols of C per thread

constexpr int kPad = 4;

}

template<bool transA, bool transB>
__global__ void gemm_kernel(const float* __restrict__ A, const float* __restrict__ B,
                            float* __restrict__ C, int M, int N, int K) {
  __shared__ float As[BK][BM + kPad];
  __shared__ float As[BK][BN + kPad];

  const int tx = threadIdx.x, ty = threadIdx.y;
  const int t  = ty * 16 + tx;
  const int64_t bm = int64_t(blockIdx.y) * BM;
  const int64_t bn = int64_t(blockIdx.x) * BN;

  float acc[TM][TN] = {};

  float (int k0 = 0; k0 < K; k0 += BK) {
    #pragma unroll
    for (int s = 0; s < (BM * BK) / 256; ++s) {
      int ai, ak;
      if constexpr (transA) {
        ai = t % BM;
        ak = t / BM + s * 4;
      } else {
        ai = t / BK + s * 16;
        ak = t % BK;
      }
      const int64_t m = bm + ai, k = k0 + ak;
      As[ak][ai] = (m < M && k < K)
                 ? (transA ? A[k * M + m] : A[m * K + k])
                 : 0.0f;
    }

    #pragma unroll
    for (int s = 0; s < (BN * BK) / 256; ++s) {
      int bj, bk;
      if constexpr (transB) {
        bj = t / BK + s * 16;
        bk = t % BK;
      } else {
        bj = t % BN;
        bk = t / BN + s * 4;
      }
      const int64_t n = bn + bj, k = k0 + bk;
      Bs[bk][bj] = (n < N && k < K)
                 ? (transB ? B[n * K + k] : B[k * N + n])
                 : 0.0f;
    }

    __syncthreads();

    #pragma unroll
    for (int k = 0; k < BK; ++k) {
      float a[TM], b[TN];
      #pragma unroll
      for (int i = 0; i < TM; ++i) a[i] = As[k][ty * TM + i];
      #pragma unroll
      for (int j = 0; j < TN; ++j) b[j] = Bs[k][tx * TN + j];
      #pragma unroll
      for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
    }

    __syncthreads(); 
  }

  #pragma unroll
  for (int i = 0; i < TM; ++i) {
    const int64_t m = bm + ty * TM + i;
    if (m >= M) continue;
    const int64_t n0  = bn + tx * TN;
    const int64_t off = m * N + n0;
    if (n0 + TN <= N && (off & 3) == 0) {
      *reinterpret_cast<float4*>(&C[off]) =
        float4{acc[i][0], acc[i][1], acc[i][2], acc[i][3]};
    } else {
      #pragma unroll
      for (int j = 0; j < TN; ++j)
        if (n0 + j < N) C[off + j] = acc[i][j];
    }
  }
}


namespace {

#define NN_CUBLAS_CHECK(expr) \
  do { \
    cublasStatus_t st_ = (expr); \
    if (st_ != CUBLAS_STATUS_SUCCESS) { \
      throw std::runtime_error(std::string("cuBLAS: " #expr ": ") \
                               + cublasGetStatusString(st_)); \
    } \
  } while(0)

cublasHandle_t handle() {
  static cublasHandle_t h = [] {
    cublasHandle_t tmp = nullptr;
    NN_CUBLAS_CHECK(cublasCreate(&tmp));
    return tmp;
  }();
  return h;
}

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

void cublas_gemm(const Stream& s, const float* A, const float* B, float* C,
               int M, int N, int K, bool transA, bool transB) {
  if (M == 0 || N == 0) return;

  auto stream = static_cast<cudaStream_t>(s.handle); 
  cublasHandle_t h = handle();
  NN_CUBLAS_CHECK(cublasSetStream(h, static_cast<cudaStream_t>(s.handle)));

  constexpr float alpha = 1.0f, beta = 0.0f;

  NN_CUBLAS_CHECK(cublasSgemm(
    h,
    transB ? CUBLAS_OP_T : CUBLAS_OP_N,
    transA ? CUBLAS_OP_T : CUBLAS_OP_N,
    N, M, K,
    &alpha,
    B, transB ? K : N,
    A, transA ? M : K,
    &beta,
    C, N
  ));
}

}