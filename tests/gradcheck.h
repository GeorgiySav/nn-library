#pragma once

#include <functional>

#include <nn/core/allocator.h>
#include <nn/core/tensor.h>

namespace nn::test {

namespace detail {

// Single-element read/write against a tensor that may live on any device.
// Perturbation has to happen in place: the parameter is captured by the
// forward closure, so replacing it with a copy would leave the graph looking
// at the unperturbed original.
inline float read_element(const Tensor& t, int idx) {
  float v = 0.0f;
  copy_bytes(&v, Device::CPU,
             static_cast<const char*>(t.raw()) + size_t(idx) * sizeof(float),
             t.device(), sizeof(float));
  return v;
}

inline void write_element(Tensor& t, int idx, float v) {
  copy_bytes(static_cast<char*>(t.raw()) + size_t(idx) * sizeof(float), t.device(),
             &v, Device::CPU, sizeof(float));
}

}  // namespace detail

inline float gradCheck(Tensor& param,
                       const std::function<float()>& loss_fn,
                       const std::function<void()>& backward_fn,
                       int num_checks = 20,
                       float h = 1e-3f,
                       uint64_t seed = 42) {
  param.zero_grad();
  loss_fn();
  backward_fn();
  // snapshot on the host: the analytic gradients are read many times below
  const Tensor grad = param.grad().clone().to(Device::CPU);

  nn::Pcg32 rng(seed);
  float max_error = 0.0f;
  for (int i{0}; i < num_checks; ++i) {
    int idx = static_cast<int>(rng.next_uint32() % param.numel());
    float original_value = detail::read_element(param, idx);

    detail::write_element(param, idx, original_value + h);
    float loss_plus_h = loss_fn();

    detail::write_element(param, idx, original_value - h);
    float loss_minus_h = loss_fn();

    detail::write_element(param, idx, original_value);

    float numerical_grad = (loss_plus_h - loss_minus_h) / (2.0f * h);
    float analytical_grad = grad.host_data()[idx];
    float error = std::abs(numerical_grad - analytical_grad) /
                  std::max(std::abs(numerical_grad) + std::abs(analytical_grad), 1e-4f);
    max_error = std::max(max_error, error);
  }

  return max_error;
}

}  // namespace nn::test
