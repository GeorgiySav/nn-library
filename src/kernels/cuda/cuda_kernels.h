#pragma once

#include <cstdint>

#include <nn/core/device.h>

namespace nn::kernels {
  void cuda_fill(const Stream& s, float v, float* X, int64_t n);
  void cuda_scale(const Stream& s, float alpha, float* X, int64_t n);
  void cuda_axpy(const Stream& s, float alpha, const float* X, float* Y, int64_t n);
  void cuda_add(const Stream& s, const float* A, const float* B, float* C, int64_t n);
  void cuda_relu(const Stream& s, const float* X, float* Y, int64_t n);
  void cuda_relu_backward(const Stream& s, const float* X, const float* gY, float* gX, int64_t n);
}