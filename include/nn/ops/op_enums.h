#pragma once

// The op codes that appear in public ops:: signatures, generated from the same
// three lists that generate Tensor's methods and the autograd wrappers.
//
// Public, unlike the rest of the kernel layer: a caller writing
// ops::unary(UnaryOp::Relu, x) needs these names, and Tensor declares one
// method per row of the .def files next to this header. What stays private in
// src/kernels is everything behind the code -- the arithmetic for each op, the
// kernel function types, and the table that dispatches them.

// A note on the namespace, because the header's location and its contents
// disagree on purpose. The codes are declared in nn::kernels and aliased into
// nn::ops below, not the other way round: nn::ops and nn::autograd both define
// unary/binary/scalar, so an op code living in either namespace would drag that
// namespace into every call's ADL set and make `unary(op, x)` ambiguous inside
// the other. nn::kernels has no such names, so it is the neutral home.

#include <cstdint>

namespace nn::kernels {

enum class UnaryOp : int {
#define NN_UNARY(Name, method) Name,
#include <nn/ops/unary_ops.def>
#undef NN_UNARY
};

// Which of a unary op's forward input (x) and forward output (y) its
// backward actually reads.
enum class UnaryNeeds : uint8_t { None = 0, X = 1, Y = 2, Both = 3 };
inline bool needs_x(UnaryNeeds n) { return (uint8_t(n) & uint8_t(UnaryNeeds::X)) != 0; }
inline bool needs_y(UnaryNeeds n) { return (uint8_t(n) & uint8_t(UnaryNeeds::Y)) != 0; }

enum class BinaryOp : int {
#define NN_BINARY(Name, method) Name,
#include <nn/ops/binary_ops.def>
#undef NN_BINARY
};

enum class ScalarOp : int {
#define NN_SCALAR(Name, method) Name,
#include <nn/ops/scalar_ops.def>
#undef NN_SCALAR
};

// How sum_all folds its input.
enum class Accum : int { Sum, SumSq, SumAbs };

}  // namespace nn::kernels

// The public spelling. ops::UnaryOp and kernels::UnaryOp name the same type;
// a caller writing ops::unary(ops::UnaryOp::Relu, x) never has to know the
// codes are declared a layer down.
namespace nn::ops {

using kernels::Accum;
using kernels::BinaryOp;
using kernels::ScalarOp;
using kernels::UnaryNeeds;
using kernels::UnaryOp;
using kernels::needs_x;
using kernels::needs_y;

}  // namespace nn::ops
