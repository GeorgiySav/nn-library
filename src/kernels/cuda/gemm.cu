#include "cuda_kernels.h"

#include <cublas_v2.h>

#include "../../cuda_common.h"

namespace nn::kernels {

// The leading dimension is the stride of the second-to-last axis of the operand
// as stored. It does not depend on the transpose flag: transposing changes which
// index multiplies it, not the layout.
//
//   transA=false  A is [M,K]  A[m*lda + k]   lda >= K
//   transA=true   A is [K,M]  A[k*lda + m]   lda >= M
//   transB=false  B is [K,N]  B[k*ldb + n]   ldb >= N
//   transB=true   B is [N,K]  B[n*ldb + k]   ldb >= K
//
// Only the [M,N] window of C is written; the ldc padding is never touched.
template<bool transA, bool transB>
__global__ void naive_gemm_kernel(const float* __restrict__ A, const float* __restrict__ B,
                            float* __restrict__ C, int M, int N, int K,
                            int64_t lda, int64_t ldb, int64_t ldc) {
  const int64_t n = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t m = int64_t(blockIdx.y) * blockDim.y + threadIdx.y;
  if (m >= M || n >= N) return;

  const float* a = A + (transA ? m : m * lda);
  const float* b = B + (transB ? n * ldb : n);
  const int64_t a_stride = transA ? lda : 1;
  const int64_t b_stride = transB ? 1 : ldb;

  float acc = 0.0f;
  for (int k = 0; k < K; ++k) {
    acc = fmaf(a[k * a_stride], b[k * b_stride], acc);
  }
  C[m * ldc + n] = acc;
}

namespace {

constexpr int BM = 64; // rows of C per block
constexpr int BN = 64; // cols of C per block
constexpr int BK = 16; // depth of one tile
constexpr int TM = 4;  // rows of C per thread
constexpr int TN = 4;  // cols of C per thread

constexpr int kPad = 4; // padding on the shared tiles avoids bank conflicts

}

// blockIdx.z selects which matrix of the batch a block works on. The z grid is
// capped at 65535, so the outer loop is what keeps a larger batch correct
// rather than silently truncated. A stride of zero makes every iteration read
// the same operand, which is how a shared weight is broadcast without a copy.
template<bool transA, bool transB>
__global__ void gemm_kernel(const float* __restrict__ A, const float* __restrict__ B,
                            float* __restrict__ C, int M, int N, int K,
                            int64_t lda, int64_t ldb, int64_t ldc,
                            int batch, int64_t sa, int64_t sb, int64_t sc) {
  __shared__ float As[BK][BM + kPad];
  __shared__ float Bs[BK][BN + kPad];

  const int tx = threadIdx.x, ty = threadIdx.y;
  const int t  = ty * 16 + tx;
  const int64_t bm = int64_t(blockIdx.y) * BM;
  const int64_t bn = int64_t(blockIdx.x) * BN;

  for (int bi = blockIdx.z; bi < batch; bi += gridDim.z) {
    const float* const Ab = A + int64_t(bi) * sa;
    const float* const Bb = B + int64_t(bi) * sb;
    float* const Cb = C + int64_t(bi) * sc;

    float acc[TM][TN] = {};

    for (int k0 = 0; k0 < K; k0 += BK) {
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
                   ? (transA ? Ab[k * lda + m] : Ab[m * lda + k])
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
                   ? (transB ? Bb[n * ldb + k] : Bb[k * ldb + n])
                   : 0.0f;
      }

      __syncthreads();  // wait for the whole tile to land in shared memory before reading it

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

      __syncthreads();  // wait for every thread to finish reading before the next k0 overwrites the tile
    }

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
      const int64_t m = bm + ty * TM + i;
      if (m >= M) continue;
      const int64_t n0  = bn + tx * TN;
      const int64_t off = m * ldc + n0;

      const int64_t abs_off = int64_t(bi) * sc + off;
      // float4 store only when the full 4 columns are in bounds and the
      // address is 16-byte aligned; otherwise fall back to scalar stores
      if (n0 + TN <= N && (abs_off & 3) == 0) {
        *reinterpret_cast<float4*>(&Cb[off]) =
          float4{acc[i][0], acc[i][1], acc[i][2], acc[i][3]};
      } else {
        #pragma unroll
        for (int j = 0; j < TN; ++j)
          if (n0 + j < N) Cb[off + j] = acc[i][j];
      }
    }
    __syncthreads();  // wait for the store before the next batch iteration reuses As/Bs
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
    NN_CUBLAS_CHECK(cublasSetMathMode(tmp, CUBLAS_TF32_TENSOR_OP_MATH));
    return tmp;
  }();
  return h;
}

}

void cuda_gemm(const Stream& s, const float* A, const float* B, float* C,
               int M, int N, int K, int64_t lda, int64_t ldb, int64_t ldc,
               bool transA, bool transB,
               int batch, int64_t sa, int64_t sb, int64_t sc) {
  if (M == 0 || N == 0 || batch == 0) return;

  auto stream = static_cast<cudaStream_t>(s.handle);
  constexpr int kMaxGridZ = 65535;
  const dim3 block(16, 16); // 256 threads
  const dim3 grid((N + BN - 1) / BN,
                  (M + BM - 1) / BM,
                  unsigned(std::min(batch, kMaxGridZ)));

  #define NN_LAUNCH(ta, tb) \
    gemm_kernel<ta, tb><<<grid, block, 0, stream>>>( \
        A, B, C, M, N, K, lda, ldb, ldc, batch, sa, sb, sc)

  if      (!transA && !transB) NN_LAUNCH(false, false);
  else if ( transA && !transB) NN_LAUNCH(true,  false);
  else if (!transA &&  transB) NN_LAUNCH(false, true);
  else                         NN_LAUNCH(true,  true);
  #undef NN_LAUNCH

  NN_CUDA_CHECK_LAUNCH(stream);
}

void cublas_gemm(const Stream& s, const float* A, const float* B, float* C,
               int M, int N, int K, int64_t lda, int64_t ldb, int64_t ldc,
               bool transA, bool transB,
               int batch, int64_t sa, int64_t sb, int64_t sc) {
  if (M == 0 || N == 0 || batch == 0) return;

  cublasHandle_t h = handle();
  NN_CUBLAS_CHECK(cublasSetStream(h, static_cast<cudaStream_t>(s.handle)));

  constexpr float alpha = 1.0f, beta = 0.0f;

  // cuBLAS is column-major, so row-major C = A*B is column-major C^T = B^T*A^T:
  // pass B first, swap M and N. Each operand's op flag and ld depend only on
  // itself, which is why the ld arguments pass straight through -- the old
  // hard-coded expressions were exactly ldb/lda/ldc for a dense operand.
  //
  // The batch strides swap with the operands for the same reason. cuBLAS
  // accepts a stride of 0, so a shared weight needs no copy here either.
  if (batch == 1) {
    NN_CUBLAS_CHECK(cublasSgemm(
      h,
      transB ? CUBLAS_OP_T : CUBLAS_OP_N,
      transA ? CUBLAS_OP_T : CUBLAS_OP_N,
      N, M, K,
      &alpha,
      B, int(ldb),
      A, int(lda),
      &beta,
      C, int(ldc)
    ));
    return;
  }

  NN_CUBLAS_CHECK(cublasSgemmStridedBatched(
    h,
    transB ? CUBLAS_OP_T : CUBLAS_OP_N,
    transA ? CUBLAS_OP_T : CUBLAS_OP_N,
    N, M, K,
    &alpha,
    B, int(ldb), sb,
    A, int(lda), sa,
    &beta,
    C, int(ldc), sc,
    batch
  ));
}

}