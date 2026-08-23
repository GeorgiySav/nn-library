#include "naive_kernels.h"

#include <cstring>

#include <nn/core/device.h>

namespace nn::kernels {

void naive_embedding(const Stream&, const float* W, const int32_t* idx,
                     float* Y, int64_t n_idx, int D, int V) {
  for (int64_t i = 0; i < n_idx; ++i) {
    const int32_t v = idx[i];
    float* y = Y + i * D;
    if (v < 0 || v >= V) {
      std::memset(y, 0, size_t(D) * sizeof(float));
      continue;
    }
    const float* w = W + int64_t(v) * D;
    for (int d = 0; d < D; ++d) y[d] = w[d];
  }
}

void naive_embedding_backward(const Stream&, const float* G, const int32_t* idx,
                              float* gW, int64_t n_idx, int D, int V) {
  for (int64_t i = 0; i < n_idx; ++i) {
    const int32_t v = idx[i];
    if (v < 0 || v >= V) continue;
    const float* g = G + i * D;
    float* w = gW + int64_t(v) * D;
    for (int d = 0; d < D; ++d) w[d] += g[d];
  }
}

}
