#pragma once

// The arithmetic of every elementwise op lives one struct per op in
// unary_ops_traits.h / binary_ops_traits.h / scalar_ops_traits.h, each
// compiled into both backends via NN_EW_INLINE.

#include <kernels/ew_inline.h>
#include <nn/ops/op_enums.h>
#include <kernels/unary_ops_traits.h>
#include <kernels/binary_ops_traits.h>
#include <kernels/scalar_ops_traits.h>

namespace nn::kernels {

// Accum, UnaryOp, BinaryOp and ScalarOp are declared in <nn/ops/op_enums.h>
// and aliased into this namespace there -- the codes are public vocabulary,
// only the arithmetic below is private.

NN_EW_INLINE float apply_accum(Accum a, float x) {
  switch (a) {
    case Accum::SumSq:  return x * x;
    case Accum::SumAbs: return fabsf(x);
    case Accum::Sum:    break;
  }
  return x;
}

NN_EW_INLINE float apply_unary(UnaryOp op, float x) {
  return unary_ops::All::fwd(op, x);
}

// y is the forward result and g the incoming gradient. Callers that already
// know this op's Needs (autograd::unary, the backward kernel drivers) may
// pass 0.0f for whichever of x/y this op does not read.
NN_EW_INLINE float apply_unary_backward(UnaryOp op, float x, float y, float g) {
  return unary_ops::All::bwd(op, x, y, g);
}

// Host-side only: this decides what a Tensor closure keeps alive and what a
// kernel driver reads before launch, neither of which happens on the device.
inline UnaryNeeds unary_needs(UnaryOp op) {
  return unary_ops::All::needs(op);
}

NN_EW_INLINE float apply_binary(BinaryOp op, float a, float b) {
  return binary_ops::All::fwd(op, a, b);
}

// side 0 -> d/da, side 1 -> d/db. c is the forward result.
NN_EW_INLINE float apply_binary_backward(BinaryOp op, int side,
                                         float a, float b, float c, float g) {
  return side == 0 ? binary_ops::All::dfda(op, a, b, c, g)
                   : binary_ops::All::dfdb(op, a, b, c, g);
}

NN_EW_INLINE float apply_scalar(ScalarOp op, float x, float k) {
  return scalar_ops::All::fwd(op, x, k);
}

NN_EW_INLINE float apply_scalar_backward(ScalarOp op, float x, float y, float g, float k) {
  return scalar_ops::All::bwd(op, x, y, g, k);
}

// Names, for error messages and tests. Indexed by the enum's value.
inline const char* unary_op_name(UnaryOp op) {
  switch (op) {
#define NN_UNARY(Name, method) case UnaryOp::Name: return #method;
#include <nn/ops/unary_ops.def>
#undef NN_UNARY
  }
  return "?";
}
inline const char* binary_op_name(BinaryOp op) {
  switch (op) {
#define NN_BINARY(Name, method) case BinaryOp::Name: return #method;
#include <nn/ops/binary_ops.def>
#undef NN_BINARY
  }
  return "?";
}
inline const char* scalar_op_name(ScalarOp op) {
  switch (op) {
#define NN_SCALAR(Name, method) case ScalarOp::Name: return #method;
#include <nn/ops/scalar_ops.def>
#undef NN_SCALAR
  }
  return "?";
}

inline constexpr int kUnaryOpCount = 0
#define NN_UNARY(Name, method) + 1
#include <nn/ops/unary_ops.def>
#undef NN_UNARY
    ;
inline constexpr int kBinaryOpCount = 0
#define NN_BINARY(Name, method) + 1
#include <nn/ops/binary_ops.def>
#undef NN_BINARY
    ;
inline constexpr int kScalarOpCount = 0
#define NN_SCALAR(Name, method) + 1
#include <nn/ops/scalar_ops.def>
#undef NN_SCALAR
    ;

static_assert(unary_ops::All::kSize == kUnaryOpCount,
              "unary_ops.def and unary_ops_traits.h::All must list exactly the same ops");
static_assert(binary_ops::All::kSize == kBinaryOpCount,
              "binary_ops.def and binary_ops_traits.h::All must list exactly the same ops");
static_assert(scalar_ops::All::kSize == kScalarOpCount,
              "scalar_ops.def and scalar_ops_traits.h::All must list exactly the same ops");

}  // namespace nn::kernels
