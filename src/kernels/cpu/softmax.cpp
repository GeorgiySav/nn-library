#include "cpu_kernels.h"

#include <cmath>
#include <cassert>

#include <nn/core/device.h>

namespace nn::kernels {

// softmax over the last axis. rows of X have stride sx; Y is dense.
void cpu_softmax_rows(const Stream&, const float* X, float* Y,
                        int M, int N, int64_t sx) {
  for (int i{0}; i < M; ++i) {
    const float* x = X + int64_t(i) * sx;
    float* y = Y + int64_t(i) * N;

    // subtracting the row max before exponentiating keeps every exponent
    // <= 0, avoiding overflow for large inputs without changing the result
    // (it cancels in the normalization below).
    float m = x[0];
    for (int j{1}; j < N; ++j) {
      if (x[j] > m) m = x[j];
    }

    float s = 0.0f;
    for (int j{0}; j < N; ++j) {
      const float e = std::exp(x[j] - m);
      y[j] = e;
      s += e;
    }

    const float inv_s = 1.0f / s;
    for (int j{0}; j < N; ++j) y[j] *= inv_s;
  }
}

// the Jacobian of a softmax row is diag(y) - y y^T, so
//   gX = y * (g - dot(y, g))
// and the whole row collapses to one dot product.
void cpu_softmax_rows_backward(const Stream&, const float* Y, const float* G,
                                 float* gX, int M, int N,
                                 int64_t sy, int64_t sg) {
  for (int i{0}; i < M; ++i) {
    const float* y = Y + int64_t(i) * sy;
    const float* g = G + int64_t(i) * sg;
    float* out = gX + int64_t(i) * N;

    float dot = 0.0f;
    for (int j{0}; j < N; ++j) dot += y[j] * g[j];
    for (int j{0}; j < N; ++j) out[j] = y[j] * (g[j] - dot);
  }
}

}
