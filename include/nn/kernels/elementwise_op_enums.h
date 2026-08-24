#pragma once

#include <cstdint>

namespace nn::kernels {

enum class UnaryOp : int {
#define NN_UNARY(Name, method) Name,
#include <nn/kernels/unary_ops.def>
#undef NN_UNARY
};

// Which of a unary op's forward input (x) and forward output (y) its
// backward actually reads.
enum class UnaryNeeds : uint8_t { None = 0, X = 1, Y = 2, Both = 3 };
inline bool needs_x(UnaryNeeds n)    { return (uint8_t(n) & uint8_t(UnaryNeeds::X)) != 0; }
inline bool needs_y(UnaryNeeds n)    { return (uint8_t(n) & uint8_t(UnaryNeeds::Y)) != 0; }
inline bool needs_both(UnaryNeeds n) { return (uint8_t(n) & uint8_t(UnaryNeeds::Both)) != 0; }

enum class BinaryOp : int {
#define NN_BINARY(Name, method) Name,
#include <nn/kernels/binary_ops.def>
#undef NN_BINARY
};

enum class ScalarOp : int {
#define NN_SCALAR(Name, method) Name,
#include <nn/kernels/scalar_ops.def>
#undef NN_SCALAR
};

}  // namespace nn::kernels
