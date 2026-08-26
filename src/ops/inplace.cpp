// Ops that write through a destination the caller owns, plus the fused Adam step.

#include <nn/ops/ops.h>

#include <stdexcept>
#include <string>
#include <cmath>
#include <nn/kernels/kernel_api.h>

#include "ops_common.h"

namespace nn::ops {

void add_inplace(Tensor& a, const Tensor& b) {
  same_device(a, b, "add_inplace");
  same_shape(a, b, "add_inplace");
  require_contiguous(a, "add_inplace (destination)");

  const auto& k = nn::kernels::kernels(a.device());
  k.binary(current_stream(a.device()), BinaryOp::Add, a.device_ptr(), view_of(a),
           b.device_ptr(), view_of(b), a.device_ptr(), a.numel());
}

void scalar_inplace(Tensor& a, ScalarOp op, float k) {
  require_contiguous(a, kernels::scalar_op_name(op));

  const auto& kern = nn::kernels::kernels(a.device());
  kern.scalar(current_stream(a.device()), op, k,
              a.device_ptr(), view_of(a), a.device_ptr(), a.numel());
}

void scale_inplace(Tensor& a, float alpha) {
  scalar_inplace(a, ScalarOp::MulScalar, alpha);
}

void axpy_inplace(Tensor& y, float alpha, const Tensor& x) {
  same_device(y, x, "axpy_inplace");
  same_shape(y, x, "axpy_inplace");
  require_contiguous(y, "axpy_inplace (destination)");
  require_contiguous(x, "axpy_inplace");

  const auto& k = nn::kernels::kernels(y.device());
  k.axpy(current_stream(y.device()), alpha, x.device_ptr(), y.device_ptr(), y.numel());
}

void fill_inplace(Tensor& a, float v) {
  require_contiguous(a, "fill_inplace");

  const auto& k = nn::kernels::kernels(a.device());
  k.fill(current_stream(a.device()), v, a.device_ptr(), a.numel());
}

void fill_from(Tensor& a, const Tensor& value) {
  same_device(a, value, "fill_from");
  require_contiguous(a, "fill_from");

  if (value.numel() != 1) {
    throw std::invalid_argument("fill_from: value must be a single element");
  }
  if (a.dtype() != value.dtype()) {
    throw std::invalid_argument("fill_from: dtypes must match");
  }

  const auto& k = nn::kernels::kernels(a.device());
  k.fill_from(current_stream(a.device()), value.device_ptr(), a.device_ptr(), a.numel());
}

void adam(const Tensor& p, const Tensor& g, Tensor& m, Tensor& v,
          float lr, float beta1, float beta2, float eps, float weight_decay, int step) {
  same_device(p, g, "adam");
  same_device(p, m, "adam");
  same_device(p, v, "adam");

  if (p.shape() != g.shape() || p.shape() != m.shape() || p.shape() != v.shape()) {
    throw std::invalid_argument("All tensors must have the same shape for adam");
  }

  const float bc1 = 1.0f - std::pow(beta1, step);
  const float bc2 = 1.0f - std::pow(beta2, step);

  const auto& k = nn::kernels::kernels(p.device());
  k.adam_step(current_stream(p.device()), p.device_ptr(), g.device_ptr(), m.device_ptr(), v.device_ptr(),
              lr, beta1, beta2, eps, weight_decay, bc1, bc2, p.numel());
}

}  // namespace nn::ops
