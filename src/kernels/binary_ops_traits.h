#pragma once

#include <kernels/ew_inline.h>
#include <nn/ops/op_enums.h>

// forward/backward arithmetic for every binary op, one struct per op. c is
// the forward result and g the incoming gradient. dfda and dfdb are the two
// partial derivatives, and apply_binary_backward picks between them at
// runtime with side (0 for d/da, 1 for d/db). both operands are broadcast to
// a common shape before the kernel runs, so dfda/dfdb produce gradients at
// that broadcast shape, and ops::binary_backward sums them back down to the
// original operand shapes.
//
// see unary_ops_traits.h for the dispatch mechanism and how to add a new op.

namespace nn::kernels::binary_ops {

struct Add {
  static constexpr BinaryOp kOp = BinaryOp::Add;
  NN_EW_INLINE static float fwd(float a, float b) { return a + b; }
  NN_EW_INLINE static float dfda(float, float, float, float g) { return g; }
  NN_EW_INLINE static float dfdb(float, float, float, float g) { return g; }
};

struct Sub {
  static constexpr BinaryOp kOp = BinaryOp::Sub;
  NN_EW_INLINE static float fwd(float a, float b) { return a - b; }
  NN_EW_INLINE static float dfda(float, float, float, float g) { return g; }
  NN_EW_INLINE static float dfdb(float, float, float, float g) { return -g; }
};

struct Mul {
  static constexpr BinaryOp kOp = BinaryOp::Mul;
  NN_EW_INLINE static float fwd(float a, float b) { return a * b; }
  NN_EW_INLINE static float dfda(float, float b, float, float g) { return g * b; }
  NN_EW_INLINE static float dfdb(float a, float, float, float g) { return g * a; }
};

// d/db of a/b is -a/b^2, which simplifies to -c/b, saving a division since
// the backward kernel already has c loaded.
struct Div {
  static constexpr BinaryOp kOp = BinaryOp::Div;
  NN_EW_INLINE static float fwd(float a, float b) { return a / b; }
  NN_EW_INLINE static float dfda(float, float b, float, float g) { return g / b; }
  NN_EW_INLINE static float dfdb(float, float b, float c, float g) { return -g * c / b; }
};

struct Pow {
  static constexpr BinaryOp kOp = BinaryOp::Pow;
  NN_EW_INLINE static float fwd(float a, float b) { return powf(a, b); }
  NN_EW_INLINE static float dfda(float a, float b, float, float g) {
    return g * b * powf(a, b - 1.0f);
  }
  NN_EW_INLINE static float dfdb(float a, float, float c, float g) { return g * c * logf(a); }
};

struct Maximum {
  static constexpr BinaryOp kOp = BinaryOp::Maximum;
  NN_EW_INLINE static float fwd(float a, float b) { return fmaxf(a, b); }
  NN_EW_INLINE static float dfda(float a, float b, float, float g) { return a >= b ? g : 0.0f; }
  NN_EW_INLINE static float dfdb(float a, float b, float, float g) { return a >= b ? 0.0f : g; }
};

struct Minimum {
  static constexpr BinaryOp kOp = BinaryOp::Minimum;
  NN_EW_INLINE static float fwd(float a, float b) { return fminf(a, b); }
  NN_EW_INLINE static float dfda(float a, float b, float, float g) { return a <= b ? g : 0.0f; }
  NN_EW_INLINE static float dfdb(float a, float b, float, float g) { return a <= b ? 0.0f : g; }
};

namespace detail {

template <class Op, class... Rest>
NN_EW_INLINE float dispatch_fwd(BinaryOp op, float a, float b) {
  if (op == Op::kOp) return Op::fwd(a, b);
  if constexpr (sizeof...(Rest) > 0) return dispatch_fwd<Rest...>(op, a, b);
  else return 0.0f;
}

template <class Op, class... Rest>
NN_EW_INLINE float dispatch_dfda(BinaryOp op, float a, float b, float c, float g) {
  if (op == Op::kOp) return Op::dfda(a, b, c, g);
  if constexpr (sizeof...(Rest) > 0) return dispatch_dfda<Rest...>(op, a, b, c, g);
  else return 0.0f;
}

template <class Op, class... Rest>
NN_EW_INLINE float dispatch_dfdb(BinaryOp op, float a, float b, float c, float g) {
  if (op == Op::kOp) return Op::dfdb(a, b, c, g);
  if constexpr (sizeof...(Rest) > 0) return dispatch_dfdb<Rest...>(op, a, b, c, g);
  else return 0.0f;
}

}  // namespace detail

template <class... Ops>
struct BinaryOpList {
  static constexpr int kSize = sizeof...(Ops);
  NN_EW_INLINE static float fwd(BinaryOp op, float a, float b) {
    return detail::dispatch_fwd<Ops...>(op, a, b);
  }
  NN_EW_INLINE static float dfda(BinaryOp op, float a, float b, float c, float g) {
    return detail::dispatch_dfda<Ops...>(op, a, b, c, g);
  }
  NN_EW_INLINE static float dfdb(BinaryOp op, float a, float b, float c, float g) {
    return detail::dispatch_dfdb<Ops...>(op, a, b, c, g);
  }
};

using All = BinaryOpList<Add, Sub, Mul, Div, Pow, Maximum, Minimum>;

}  // namespace nn::kernels::binary_ops
