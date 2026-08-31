#include <nn/ops/ops.h>

#include <stdexcept>
#include <string>

#include <kernels/kernel_api.h>

#include "ops_common.h"

namespace nn::ops {

Tensor unary(UnaryOp op, const Tensor& x) {
  Tensor out(x.shape(), x.device(), x.dtype());
  const auto& k = nn::kernels::kernels(x.device());
  k.unary(current_stream(x.device()), op, x.device_ptr(), view_of(x),
          out.device_ptr(), out.numel());
  return out;
}

Tensor unary_backward(UnaryOp op, const Tensor& x, const Tensor& y, const Tensor& g) {
  // x and y may be undefined placeholders: unary_ops.def's Needs column says
  // which one (if either) this op's derivative actually reads, and the caller
  // only keeps the one it needs alive. g always has the true shape, device
  // and dtype, since a unary op never changes any of them.
  if (x.defined()) {
    same_device(x, g, kernels::unary_op_name(op));
    same_shape(x, g, kernels::unary_op_name(op));
  }
  if (y.defined()) {
    same_device(y, g, kernels::unary_op_name(op));
    same_shape(y, g, kernels::unary_op_name(op));
  }

  Tensor gx(g.shape(), g.device(), g.dtype());
  const auto& k = nn::kernels::kernels(g.device());
  k.unary_backward(current_stream(g.device()), op,
                   x.defined() ? x.device_ptr() : nullptr,
                   x.defined() ? view_of(x)     : TensorView{},
                   y.defined() ? y.device_ptr() : nullptr,
                   y.defined() ? view_of(y)     : TensorView{},
                   g.device_ptr(), view_of(g),
                   gx.device_ptr(), gx.numel());
  return gx;
}

Tensor binary(BinaryOp op, const Tensor& a, const Tensor& b) {
  same_device(a, b, kernels::binary_op_name(op));

  // expand_view broadcasts both operands up to the output shape with stride 0
  // on the stretched axes, so the kernel just reads two same-shaped views.
  const Shape out_shape = broadcast_shapes(a.shape(), b.shape());
  const Tensor ea = a.expand_view(out_shape);
  const Tensor eb = b.expand_view(out_shape);

  Tensor out(out_shape, a.device(), a.dtype());
  const auto& k = nn::kernels::kernels(a.device());
  k.binary(current_stream(a.device()), op, ea.device_ptr(), view_of(ea),
           eb.device_ptr(), view_of(eb), out.device_ptr(), out.numel());
  return out;
}


Tensor binary_backward(BinaryOp op, int side, const Tensor& a, const Tensor& b,
                       const Tensor& c, const Tensor& g) {
  const char* name = kernels::binary_op_name(op);
  same_shape(c, g, name);

  // Add/Sub pass the gradient straight through (negated for Sub's right
  // side), only summed back down if that operand was broadcast up to c.
  const Shape& target = (side == 0) ? a.shape() : b.shape();
  if (op == BinaryOp::Add || (op == BinaryOp::Sub && side == 0)) {
    return sum_to(g, target);
  }
  if (op == BinaryOp::Sub) {
    return scalar(ScalarOp::MulScalar, sum_to(g, target), -1.0f);
  }

  const Shape bshape = c.shape();
  const Tensor ea = a.expand_view(bshape);
  const Tensor eb = b.expand_view(bshape);

  Tensor full(bshape, c.device(), c.dtype());
  const auto& k = nn::kernels::kernels(c.device());
  k.binary_backward(current_stream(c.device()), op, side,
                    ea.device_ptr(), view_of(ea),
                    eb.device_ptr(), view_of(eb),
                    c.device_ptr(), view_of(c),
                    g.device_ptr(), view_of(g),
                    full.device_ptr(), full.numel());
  return sum_to(full, (side == 0) ? a.shape() : b.shape());
}

Tensor scalar(ScalarOp op, const Tensor& x, float k_value) {
  Tensor out(x.shape(), x.device(), x.dtype());
  const auto& k = nn::kernels::kernels(x.device());
  k.scalar(current_stream(x.device()), op, k_value, x.device_ptr(), view_of(x),
           out.device_ptr(), out.numel());
  return out;
}

Tensor scalar_backward(ScalarOp op, const Tensor& x, const Tensor& y,
                       const Tensor& g, float k_value) {
  same_shape(x, y, kernels::scalar_op_name(op));
  same_shape(x, g, kernels::scalar_op_name(op));

  Tensor gx(x.shape(), x.device(), x.dtype());
  const auto& k = nn::kernels::kernels(x.device());
  k.scalar_backward(current_stream(x.device()), op, k_value,
                    x.device_ptr(), view_of(x),
                    y.device_ptr(), view_of(y),
                    g.device_ptr(), view_of(g),
                    gx.device_ptr(), gx.numel());
  return gx;
}

Tensor relu(const Tensor& x) { return unary(UnaryOp::Relu, x); }

Tensor relu_backward(const Tensor& x, const Tensor& g_out) {
  same_device(x, g_out, "relu_backward");
  same_shape(x, g_out, "relu_backward");
  return unary_backward(UnaryOp::Relu, x, x, g_out);
}

Tensor add(const Tensor& a, const Tensor& b) { return binary(BinaryOp::Add, a, b); }
Tensor mul(const Tensor& a, const Tensor& b) { return binary(BinaryOp::Mul, a, b); }

Tensor dropout(const Tensor& x, float p, uint64_t seed, uint64_t offset) {
  if (!(p >= 0.0f && p <= 1.0f)) {
    throw std::invalid_argument("dropout: p must be in [0, 1]");
  }
  const float scale = (p < 1.0f) ? 1.0f / (1.0f - p) : 0.0f;

  Tensor out(x.shape(), x.device(), x.dtype());
  const auto& k = nn::kernels::kernels(x.device());
  k.dropout(current_stream(x.device()), x.device_ptr(), view_of(x),
            out.device_ptr(), seed, offset, p, scale, x.numel());
  return out;
}

}  // namespace nn::ops
