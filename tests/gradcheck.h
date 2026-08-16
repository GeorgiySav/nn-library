#pragma once

#include <functional>

#include <nn/core/tensor.h>

namespace nn::test {

float gradCheck(Tensor& param,
                const std::function<float()>& loss_fn,
                const std::function<void()>& backward_fn,
                int num_checks = 20,
                float h = 1e-3f,
                uint64_t seed = 42) {
  param.zero_grad();
  loss_fn();
  backward_fn();
  const Tensor grad = param.grad().clone();

  nn::Pcg32 rng(seed);
  float max_error = 0.0f;
  for (int i{0}; i < num_checks; ++i) {
    int idx = static_cast<int>(rng.next_uint32() % param.numel());
    float original_value = param.data()[idx];

    param.data()[idx] = original_value + h;
    float loss_plus_h = loss_fn();

    param.data()[idx] = original_value - h;
    float loss_minus_h = loss_fn();

    param.data()[idx] = original_value;

    float numerical_grad = (loss_plus_h - loss_minus_h) / (2.0f * h);
    float analytical_grad = grad.data()[idx];
    float error = std::abs(numerical_grad - analytical_grad) /
                  std::max(std::abs(numerical_grad) + std::abs(analytical_grad), 1e-4f);
    max_error = std::max(max_error, error);
  }

  return max_error;
}

}