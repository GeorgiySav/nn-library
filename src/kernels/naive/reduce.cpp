#include "naive_kernels.h"

#include <nn/core/device.h>

#include "../strided_index.h"

namespace nn::kernels {

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

void naive_sum_to(const Stream&, const float* g, TensorView keep, TensorView red,
                  float* out, int64_t n_out, int64_t n_red) {
  for (int64_t j = 0; j < n_out; ++j) {
    const int64_t base = offset_of(keep, j);
    float acc = 0.0f;
    for (int64_t k = 0; k < n_red; ++k) acc += g[base + offset_of(red, k)];
    out[j] = acc;
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
