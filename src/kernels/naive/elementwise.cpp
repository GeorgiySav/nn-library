#include "naive_kernels.h"

#include <cmath>

#include <nn/core/device.h>

#include "../strided_index.h"

namespace nn::kernels {

namespace {

// The strided siblings are generated from the same op the dense kernel uses, so
// there is one place per op where the arithmetic lives.
template <class Op>
void map1_strided(const float* x, TensorView v, float* y, int64_t n, Op op) {
  for (int64_t i = 0; i < n; ++i) y[i] = op(x[offset_of(v, i)]);
}

template <class Op>
void map2_strided(const float* a, TensorView va, const float* b, TensorView vb,
                  float* c, int64_t n, Op op) {
  for (int64_t i = 0; i < n; ++i) {
    c[i] = op(a[offset_of(va, i)], b[offset_of(vb, i)]);
  }
}

}  // namespace

void naive_relu_strided(const Stream&, const float* X, TensorView v,
                        float* Y, int64_t n) {
  map1_strided(X, v, Y, n, [](float x) { return x > 0.0f ? x : 0.0f; });
}

void naive_relu_backward_strided(const Stream&, const float* X, TensorView vx,
                                 const float* gY, TensorView vg,
                                 float* gX, int64_t n) {
  map2_strided(X, vx, gY, vg, gX, n,
               [](float x, float g) { return x > 0.0f ? g : 0.0f; });
}

void naive_add_strided(const Stream&, const float* A, TensorView va,
                       const float* B, TensorView vb, float* C, int64_t n) {
  map2_strided(A, va, B, vb, C, n, [](float a, float b) { return a + b; });
}

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