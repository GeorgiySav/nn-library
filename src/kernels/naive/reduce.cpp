#include "naive_kernels.h"

namespace nn::kernels {

void naive_col_sum(const float* X, float* out, int M, int N) {
  for (int n{0}; n < N; ++n) {
    float sum = 0.0f;
    for (int m{0}; m < M; ++m) {
      sum += X[m*N + n];
    }
    out[n] = sum;
  }
}

void naive_add_row_bias(const float* X, const float* b, float* Y, int M, int N) {
  for (int m{0}; m < M; ++m) {
    for (int n{0}; n < N; ++n) {
      Y[m*N + n] = X[m*N + n] + b[n];
    }
  }
}

}