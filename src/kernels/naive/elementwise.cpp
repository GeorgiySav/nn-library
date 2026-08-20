#include "naive_kernels.h"

#include <cmath>

#include <nn/core/device.h>

namespace nn::kernels {

void naive_fill(const Stream& s, float v, float* X, int64_t n) {
  for (int64_t i{0}; i < n; ++i) {
    X[i] = v;
  }
}

void naive_scale(const Stream& s, float alpha, float* X, int64_t n) {
  for (int64_t i{0}; i < n; ++i) {
    X[i] *= alpha;
  }
}

void naive_axpy(const Stream& s, float alpha, const float* X, float* Y, int64_t n) {
  for (int64_t i{0}; i < n; ++i) {
    Y[i] += alpha * X[i];
  }
}

void naive_add(const Stream& s, const float* A, const float* B, float* C, int64_t n) {
  for (int64_t i{0}; i < n; ++i) {
    C[i] = A[i] + B[i];
  }
}

void naive_relu(const Stream& s, const float* X, float* Y, int64_t n) {
  for (int64_t i{0}; i < n; ++i) {
    Y[i] = X[i] > 0.0f ? X[i] : 0.0f;
  }
}

void naive_relu_backward(const Stream& s, const float* X, const float* gY, float* gX, int64_t n) {
  for (int64_t i{0}; i < n; ++i) {
    gX[i] = X[i] > 0.0f ? gY[i] : 0.0f;
  }
}

void naive_adam_step(const Stream& s, float* p, const float* g, float* m, float* v,
                      float lr, float b1, float b2, float eps, float bc1, float bc2, int64_t n) {
  for (int64_t i{0}; i < n; ++i) {
    m[i] = b1 * m[i] + (1.0f - b1) * g[i];
    v[i] = b2 * v[i] + (1.0f - b2) * g[i] * g[i];
    p[i] -= lr * (m[i] / bc1) / (std::sqrt(v[i] / bc2) + eps);
  }
}

}