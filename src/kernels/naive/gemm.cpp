#include "naive_kernels.h"

#include <cstring>

#include <nn/core/device.h>

namespace nn::kernels {

namespace {

//   transA=false  A is [M,K]  A[m*lda + k]   lda >= K
//   transA=true   A is [K,M]  A[k*lda + m]   lda >= M
//   transB=false  B is [K,N]  B[k*ldb + n]   ldb >= N
//   transB=true   B is [N,K]  B[n*ldb + k]   ldb >= K
void gemm_nn(const float* A, const float* B, float* C, int M, int N, int K,
             int64_t lda, int64_t ldb, int64_t ldc) {
  for (int m = 0; m < M; ++m) {
    float* c = C + int64_t(m) * ldc;
    std::memset(c, 0, size_t(N) * sizeof(float));
    for (int k = 0; k < K; ++k) {
      const float  a = A[int64_t(m) * lda + k];
      const float* b = B + int64_t(k) * ldb;
      for (int n = 0; n < N; ++n) c[n] += a * b[n];
    }
  }
}

void gemm_nt(const float* A, const float* B, float* C, int M, int N, int K,
             int64_t lda, int64_t ldb, int64_t ldc) {
  #pragma clang fp reassociate(on)
  for (int m = 0; m < M; ++m) {
    const float* a = A + int64_t(m) * lda;
    for (int n = 0; n < N; ++n) {
      const float* b = B + int64_t(n) * ldb;
      float sum = 0.0f;
      for (int k = 0; k < K; ++k) sum += a[k] * b[k];
      C[int64_t(m) * ldc + n] = sum;
    }
  }
}

void gemm_tn(const float* A, const float* B, float* C, int M, int N, int K,
             int64_t lda, int64_t ldb, int64_t ldc) {
  for (int m = 0; m < M; ++m) {
    std::memset(C + int64_t(m) * ldc, 0, size_t(N) * sizeof(float));
  }
  for (int k = 0; k < K; ++k) {
    const float* b = B + int64_t(k) * ldb;
    for (int m = 0; m < M; ++m) {
      const float  a = A[int64_t(k) * lda + m];
      float* c = C + int64_t(m) * ldc;
      for (int n = 0; n < N; ++n) c[n] += a * b[n];
    }
  }
}

void gemm_tt(const float* A, const float* B, float* C, int M, int N, int K,
             int64_t lda, int64_t ldb, int64_t ldc) {
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      const float* b = B + int64_t(n) * ldb;
      float sum = 0.0f;
      for (int k = 0; k < K; ++k) {
        sum += A[int64_t(k) * lda + m] * b[k];
      }
      C[int64_t(m) * ldc + n] = sum;
    }
  }
}

}  // namespace

void naive_gemm(const Stream& s, const float* A, const float* B, float* C,
                int M, int N, int K, int64_t lda, int64_t ldb, int64_t ldc,
                bool transA, bool transB) {
  if (!transA && !transB) return gemm_nn(A, B, C, M, N, K, lda, ldb, ldc);
  if (!transA &&  transB) return gemm_nt(A, B, C, M, N, K, lda, ldb, ldc);
  if ( transA && !transB) return gemm_tn(A, B, C, M, N, K, lda, ldb, ldc);
  gemm_tt(A, B, C, M, N, K, lda, ldb, ldc);
}

}