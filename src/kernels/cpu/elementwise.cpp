#include "cpu_kernels.h"

#include <cmath>

#include <nn/core/device.h>
#include <kernels/random.h>

#include "../strided_index.h"

namespace nn::kernels {

void cpu_unary(const Stream&, UnaryOp op, const float* X, TensorView vx,
                 float* Y, int64_t n) {
  for (int64_t i = 0; i < n; ++i) Y[i] = apply_unary(op, X[offset_of(vx, i)]);
}

void cpu_unary_backward(const Stream&, UnaryOp op,
                          const float* X, TensorView vx,
                          const float* Y, TensorView vy,
                          const float* G, TensorView vg,
                          float* gX, int64_t n) {
  // X/Y may be null with a degenerate view when this op's derivative does not
  // read that side (see unary_needs). the caller may have kept nothing else
  // alive for it, so it must not be dereferenced.
  const UnaryNeeds needs = unary_needs(op);
  const bool nx = needs_x(needs), ny = needs_y(needs);
  for (int64_t i = 0; i < n; ++i) {
    const float xv = nx ? X[offset_of(vx, i)] : 0.0f;
    const float yv = ny ? Y[offset_of(vy, i)] : 0.0f;
    gX[i] = apply_unary_backward(op, xv, yv, G[offset_of(vg, i)]);
  }
}

void cpu_binary(const Stream&, BinaryOp op,
                  const float* A, TensorView va,
                  const float* B, TensorView vb,
                  float* C, int64_t n) {
  for (int64_t i = 0; i < n; ++i) {
    C[i] = apply_binary(op, A[offset_of(va, i)], B[offset_of(vb, i)]);
  }
}

void cpu_binary_backward(const Stream&, BinaryOp op, int side,
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

void cpu_scalar(const Stream&, ScalarOp op, float k,
                  const float* X, TensorView vx, float* Y, int64_t n) {
  for (int64_t i = 0; i < n; ++i) Y[i] = apply_scalar(op, X[offset_of(vx, i)], k);
}

void cpu_scalar_backward(const Stream&, ScalarOp op, float k,
                           const float* X, TensorView vx,
                           const float* Y, TensorView vy,
                           const float* G, TensorView vg,
                           float* gX, int64_t n) {
  for (int64_t i = 0; i < n; ++i) {
    gX[i] = apply_scalar_backward(op, X[offset_of(vx, i)], Y[offset_of(vy, i)],
                                  G[offset_of(vg, i)], k);
  }
}

void cpu_fill(const Stream&, float v, float* X, int64_t n) {
  for (int64_t i{0}; i < n; ++i) {
    X[i] = v;
  }
}

void cpu_fill_from(const Stream&, const float* src, float* X, int64_t n) {
  const float v = *src;
  for (int64_t i{0}; i < n; ++i) {
    X[i] = v;
  }
}

void cpu_axpy(const Stream&, float alpha, const float* X, float* Y, int64_t n) {
  for (int64_t i{0}; i < n; ++i) {
    Y[i] += alpha * X[i];
  }
}

void cpu_adam_step(const Stream&, float* p, const float* g, float* m, float* v,
                      float lr, float b1, float b2, float eps, float wd,
                      float bc1, float bc2, int64_t n) {
  for (int64_t i{0}; i < n; ++i) {
    m[i] = b1 * m[i] + (1.0f - b1) * g[i];
    v[i] = b2 * v[i] + (1.0f - b2) * g[i] * g[i];
    p[i] = p[i] * (1.0f - lr * wd)
         - lr * (m[i] / bc1) / (std::sqrt(v[i] / bc2) + eps);
  }
}

void cpu_dropout(const Stream&, const float* X, TensorView vx, float* Y,
                   uint64_t seed, uint64_t offset, float p, float scale,
                   int64_t n) {
  for (int64_t i{0}; i < n; ++i) {
    const float u = random_uniform(seed, offset + uint64_t(i));
    Y[i] = (u >= p) ? X[offset_of(vx, i)] * scale : 0.0f;
  }
}

}
