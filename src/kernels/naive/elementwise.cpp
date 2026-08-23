#include "naive_kernels.h"

#include <cmath>

#include <nn/core/device.h>

#include "../strided_index.h"

namespace nn::kernels {

void naive_unary(const Stream&, UnaryOp op, const float* X, TensorView vx,
                 float* Y, int64_t n) {
  for (int64_t i = 0; i < n; ++i) Y[i] = apply_unary(op, X[offset_of(vx, i)]);
}

void naive_unary_backward(const Stream&, UnaryOp op,
                          const float* X, TensorView vx,
                          const float* Y, TensorView vy,
                          const float* G, TensorView vg,
                          float* gX, int64_t n) {
  for (int64_t i = 0; i < n; ++i) {
    gX[i] = apply_unary_backward(op, X[offset_of(vx, i)], Y[offset_of(vy, i)],
                                 G[offset_of(vg, i)]);
  }
}

void naive_binary(const Stream&, BinaryOp op,
                  const float* A, TensorView va,
                  const float* B, TensorView vb,
                  float* C, int64_t n) {
  for (int64_t i = 0; i < n; ++i) {
    C[i] = apply_binary(op, A[offset_of(va, i)], B[offset_of(vb, i)]);
  }
}

void naive_binary_backward(const Stream&, BinaryOp op, int side,
                           const float* A, TensorView va,
                           const float* B, TensorView vb,
                           const float* C, TensorView vc,
                           const float* G, TensorView vg,
                           float* out, int64_t n) {
  for (int64_t i = 0; i < n; ++i) {
    out[i] = apply_binary_backward(op, side, A[offset_of(va, i)], B[offset_of(vb, i)],
                                   C[offset_of(vc, i)], G[offset_of(vg, i)]);
  }
}

void naive_scalar(const Stream&, ScalarOp op, float k,
                  const float* X, TensorView vx, float* Y, int64_t n) {
  for (int64_t i = 0; i < n; ++i) Y[i] = apply_scalar(op, X[offset_of(vx, i)], k);
}

void naive_scalar_backward(const Stream&, ScalarOp op, float k,
                           const float* X, TensorView vx,
                           const float* Y, TensorView vy,
                           const float* G, TensorView vg,
                           float* gX, int64_t n) {
  for (int64_t i = 0; i < n; ++i) {
    gX[i] = apply_scalar_backward(op, X[offset_of(vx, i)], Y[offset_of(vy, i)],
                                  G[offset_of(vg, i)], k);
  }
}

void naive_fill(const Stream&, float v, float* X, int64_t n) {
  for (int64_t i{0}; i < n; ++i) {
    X[i] = v;
  }
}

void naive_fill_from(const Stream&, const float* src, float* X, int64_t n) {
  const float v = *src;
  for (int64_t i{0}; i < n; ++i) {
    X[i] = v;
  }
}

void naive_axpy(const Stream&, float alpha, const float* X, float* Y, int64_t n) {
  for (int64_t i{0}; i < n; ++i) {
    Y[i] += alpha * X[i];
  }
}

void naive_adam_step(const Stream&, float* p, const float* g, float* m, float* v,
                      float lr, float b1, float b2, float eps, float bc1, float bc2, int64_t n) {
  for (int64_t i{0}; i < n; ++i) {
    m[i] = b1 * m[i] + (1.0f - b1) * g[i];
    v[i] = b2 * v[i] + (1.0f - b2) * g[i] * g[i];
    p[i] -= lr * (m[i] / bc1) / (std::sqrt(v[i] / bc2) + eps);
  }
}

}
