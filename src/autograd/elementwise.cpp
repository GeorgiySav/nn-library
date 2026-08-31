#include <nn/autograd/functions.h>

#include <nn/ops/ops.h>

#include "autograd_common.h"

// unary_needs and the op-name lookups are part of the private kernel layer;
// this file is library implementation, so it may reach into src/.
#include <kernels/elementwise_ops.h>

namespace nn::autograd {

Tensor unary(ops::UnaryOp op, const Tensor& x) {
  Tensor out = ops::unary(op, x);

  // Keep alive only whichever of x, out the derivative actually reads (see
  // unary_ops.def's Needs column).
  const kernels::UnaryNeeds needs = kernels::unary_needs(op);
  Tensor saved_x = kernels::needs_x(needs) ? x   : Tensor{};
  Tensor saved_y = kernels::needs_y(needs) ? out : Tensor{};

  record_op(out, kernels::unary_op_name(op),
    [op, saved_x, saved_y](const Tensor& g, std::span<Tensor> g_in) {
      g_in[0] = ops::unary_backward(op, saved_x, saved_y, g);
    }, x);

  return out;
}

Tensor binary(ops::BinaryOp op, const Tensor& a, const Tensor& b) {
  Tensor out = ops::binary(op, a, b);

  record_op(out, kernels::binary_op_name(op),
    [op, a, b, out](const Tensor& g, std::span<Tensor> g_in) {
      g_in[0] = ops::binary_backward(op, 0, a, b, out, g);
      g_in[1] = ops::binary_backward(op, 1, a, b, out, g);
    }, a, b);

  return out;
}

Tensor scalar(ops::ScalarOp op, const Tensor& x, float k) {
  Tensor out = ops::scalar(op, x, k);

  record_op(out, kernels::scalar_op_name(op),
    [op, k, x, out](const Tensor& g, std::span<Tensor> g_in) {
      g_in[0] = ops::scalar_backward(op, x, out, g, k);
    }, x);

  return out;
}

// The named wrappers, generated from the same lists as the kernels.
//
// autograd:: is spelled out so these read the same as the Tensor-method
// wrappers in tensor_methods.cpp, which have to qualify.

#define NN_UNARY(Name, method) \
  Tensor method(const Tensor& x) { return autograd::unary(ops::UnaryOp::Name, x); }
#include <nn/ops/unary_ops.def>
#undef NN_UNARY

#define NN_BINARY(Name, method)                        \
  Tensor method(const Tensor& a, const Tensor& b) {    \
    return autograd::binary(ops::BinaryOp::Name, a, b); }
#include <nn/ops/binary_ops.def>
#undef NN_BINARY

#define NN_SCALAR(Name, method)                        \
  Tensor method(const Tensor& x, float k) {            \
    return autograd::scalar(ops::ScalarOp::Name, x, k); }
#include <nn/ops/scalar_ops.def>
#undef NN_SCALAR

}  // namespace nn::autograd
