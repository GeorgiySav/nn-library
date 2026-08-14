#include "naive_kernels.h"

namespace nn::kernels {

void naive_gemm(const float* A, const float* B, float* C, int M, int N, int K, bool transA, bool transB) {
  for (int m{0}; m < M; ++m) {
    for (int n{0}; n < N; ++n) {
      float sum = 0.0f;
      for (int k{0}; k < K; ++k) {
        const float a = transA ? A[k*M + m] : A[m*K + k];
        const float b = transB ? B[n*K + k] : B[k*N + n];
        sum += a * b;
        C[m*N + n] = sum;
      }
    }
  }
}

}