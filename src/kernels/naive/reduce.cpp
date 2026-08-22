#include "naive_kernels.h"

#include <nn/core/device.h>

#include "../strided_index.h"

namespace nn::kernels {

void naive_col_sum(const Stream& s, const float* X, float* out, int M, int N, int64_t sx) {
  for (int n{0}; n < N; ++n) {
    float sum = 0.0f;
    for (int m{0}; m < M; ++m) {
      sum += X[int64_t(m)*sx + n];
    }
    out[n] = sum;
  }
}

void naive_add_row_bias(const Stream& s, const float* X, const float* b, float* Y, int M, int N, int64_t sx) {
  for (int m{0}; m < M; ++m) {
    for (int n{0}; n < N; ++n) {
      Y[int64_t(m)*N + n] = X[int64_t(m)*sx + n] + b[n];
    }
  }
}

void naive_argmax_rows(const Stream& s, const float* X, int32_t* out, int M, int N, int64_t sx) {
  for (int i = 0; i < M; ++i) {
    const float* row = X + int64_t(i) * sx;
    int best = 0;
    for (int j = 1; j < N; ++j) {
      if (row[j] > row[best]) best = j;
    }
    out[i] = int32_t(best);
  }
}

void naive_sum_all(const Stream&, const float* X, float* out, float*, int64_t n) {
  double acc = 0.0;
  for (int64_t i = 0; i < n; ++i) acc += double(X[i]);
  *out = float(acc);
}

void naive_sum_all_strided(const Stream&, const float* X, TensorView v,
                           float* out, float*, int64_t n) {
  double acc = 0.0;
  for (int64_t i = 0; i < n; ++i) acc += double(X[offset_of(v, i)]);
  *out = float(acc);
}

}