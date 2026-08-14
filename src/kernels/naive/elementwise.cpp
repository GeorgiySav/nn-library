#include "naive_kernels.h"

namespace nn::kernels {

void naive_fill(float v, float* X, int64_t n) {
  for (int64_t i{0}; i < n; ++i) {
    X[i] = v;
  }
}

void naive_scale(float alpha, float* X, int64_t n) {
  for (int64_t i{0}; i < n; ++i) {
    X[i] *= alpha;
  }
}

void naive_axpy(float alpha, const float* X, float* Y, int64_t n) {
  for (int64_t i{0}; i < n; ++i) {
    Y[i] += alpha * X[i];
  }
}

void naive_add(const float* A, const float* B, float* C, int64_t n) {
  for (int64_t i{0}; i < n; ++i) {
    C[i] = A[i] + B[i];
  }
}

void naive_relu(const float* X, float* Y, int64_t n) {
  for (int64_t i{0}; i < n; ++i) {
    Y[i] = X[i] > 0.0f ? X[i] : 0.0f;
  }
}

void naive_relu_backward(const float* X, const float* gY, float* gX, int64_t n) {
  for (int64_t i{0}; i < n; ++i) {
    gX[i] = X[i] > 0.0f ? gY[i] : 0.0f;
  }
}

}