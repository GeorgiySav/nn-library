#include "naive_kernels.h"

#include <cstring>

namespace nn::kernels {

namespace {

void gemm_nn(const float* A, const float* B, float* C, int M, int N, int K) {
  for (int m = 0; m < M; ++m) {
    float* c = C + int64_t(m) * N;
    std::memset(c, 0, size_t(N) * sizeof(float));
    for (int k = 0; k < K; ++k) {
      const float  a = A[int64_t(m) * K + k];
      const float* b = B + int64_t(k) * N;
      for (int n = 0; n < N; ++n) c[n] += a * b[n];
    }
  }
}

void gemm_nt(const float* A, const float* B, float* C, int M, int N, int K) {
  #pragma clang fp reassociate(on)
  for (int m = 0; m < M; ++m) {
    const float* a = A + int64_t(m) * K;
    for (int n = 0; n < N; ++n) {
      const float* b = B + int64_t(n) * K;
      float sum = 0.0f;
      for (int k = 0; k < K; ++k) sum += a[k] * b[k];
      C[int64_t(m) * N + n] = sum;
    }
  }
}

void gemm_tn(const float* A, const float* B, float* C, int M, int N, int K) {
  std::memset(C, 0, size_t(M) * size_t(N) * sizeof(float));
  for (int k = 0; k < K; ++k) {
    const float* b = B + int64_t(k) * N;
    for (int m = 0; m < M; ++m) {
      const float  a = A[int64_t(k) * M + m];
      float* c = C + int64_t(m) * N;
      for (int n = 0; n < N; ++n) c[n] += a * b[n];
    }
  }
}

void gemm_tt(const float* A, const float* B, float* C, int M, int N, int K) {
  for (int m = 0; m < M; ++m) {
    for (int n = 0; n < N; ++n) {
      const float* b = B + int64_t(n) * K;
      float sum = 0.0f;
      for (int k = 0; k < K; ++k) {
        sum += A[int64_t(k) * M + m] * b[k];
      }
      C[int64_t(m) * N + n] = sum;
    }
  }
}

}  // namespace

void naive_gemm(const float* A, const float* B, float* C,
                int M, int N, int K, bool transA, bool transB) {
  if (!transA && !transB) return gemm_nn(A, B, C, M, N, K);
  if (!transA &&  transB) return gemm_nt(A, B, C, M, N, K);
  if ( transA && !transB) return gemm_tn(A, B, C, M, N, K);
  gemm_tt(A, B, C, M, N, K);
}

}