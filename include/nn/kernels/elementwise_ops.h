#pragma once

// The arithmetic of every elementwise op, in one place, compiled into both
// backends. The naive backend includes this as host code and the CUDA backend
// as __host__ __device__ code, so the two can never disagree about what `gelu`
// means -- there is only one expression.

#if defined(__CUDACC__)
#  define NN_EW_INLINE __host__ __device__ inline
#else
#  include <cmath>
#  define NN_EW_INLINE inline
#endif

namespace nn::kernels {

enum class UnaryOp : int {
#define NN_UNARY(Name, method, fwd, bwd) Name,
#include <nn/kernels/unary_ops.def>
#undef NN_UNARY
};

enum class BinaryOp : int {
#define NN_BINARY(Name, method, fwd, da, db) Name,
#include <nn/kernels/binary_ops.def>
#undef NN_BINARY
};

enum class ScalarOp : int {
#define NN_SCALAR(Name, method, fwd, bwd) Name,
#include <nn/kernels/scalar_ops.def>
#undef NN_SCALAR
};

enum class Accum : int { Sum, SumSq, SumAbs };

NN_EW_INLINE float apply_accum(Accum a, float x) {
  switch (a) {
    case Accum::SumSq:  return x * x;
    case Accum::SumAbs: return fabsf(x);
    case Accum::Sum:    break;
  }
  return x;
}

NN_EW_INLINE float apply_unary(UnaryOp op, float x) {
  switch (op) {
#define NN_UNARY(Name, method, fwd, bwd) case UnaryOp::Name: return (fwd);
#include <nn/kernels/unary_ops.def>
#undef NN_UNARY
  }
  return 0.0f;
}

// y is the forward result and g the incoming gradient; see unary_ops.def.
NN_EW_INLINE float apply_unary_backward(UnaryOp op, float x, float y, float g) {
  switch (op) {
#define NN_UNARY(Name, method, fwd, bwd) case UnaryOp::Name: return (bwd);
#include <nn/kernels/unary_ops.def>
#undef NN_UNARY
  }
  return 0.0f;
}

NN_EW_INLINE float apply_binary(BinaryOp op, float a, float b) {
  switch (op) {
#define NN_BINARY(Name, method, fwd, da, db) case BinaryOp::Name: return (fwd);
#include <nn/kernels/binary_ops.def>
#undef NN_BINARY
  }
  return 0.0f;
}

// side 0 -> d/da, side 1 -> d/db. c is the forward result.
NN_EW_INLINE float apply_binary_backward(BinaryOp op, int side,
                                         float a, float b, float c, float g) {
  if (side == 0) {
    switch (op) {
#define NN_BINARY(Name, method, fwd, da, db) case BinaryOp::Name: return (da);
#include <nn/kernels/binary_ops.def>
#undef NN_BINARY
    }
  } else {
    switch (op) {
#define NN_BINARY(Name, method, fwd, da, db) case BinaryOp::Name: return (db);
#include <nn/kernels/binary_ops.def>
#undef NN_BINARY
    }
  }
  return 0.0f;
}

NN_EW_INLINE float apply_scalar(ScalarOp op, float x, float k) {
  switch (op) {
#define NN_SCALAR(Name, method, fwd, bwd) case ScalarOp::Name: return (fwd);
#include <nn/kernels/scalar_ops.def>
#undef NN_SCALAR
  }
  return 0.0f;
}

NN_EW_INLINE float apply_scalar_backward(ScalarOp op, float x, float y, float g, float k) {
  switch (op) {
#define NN_SCALAR(Name, method, fwd, bwd) case ScalarOp::Name: return (bwd);
#include <nn/kernels/scalar_ops.def>
#undef NN_SCALAR
  }
  return 0.0f;
}

// Names, for error messages and tests. Indexed by the enum's value.
inline const char* unary_op_name(UnaryOp op) {
  switch (op) {
#define NN_UNARY(Name, method, fwd, bwd) case UnaryOp::Name: return #method;
#include <nn/kernels/unary_ops.def>
#undef NN_UNARY
  }
  return "?";
}
inline const char* binary_op_name(BinaryOp op) {
  switch (op) {
#define NN_BINARY(Name, method, fwd, da, db) case BinaryOp::Name: return #method;
#include <nn/kernels/binary_ops.def>
#undef NN_BINARY
  }
  return "?";
}
inline const char* scalar_op_name(ScalarOp op) {
  switch (op) {
#define NN_SCALAR(Name, method, fwd, bwd) case ScalarOp::Name: return #method;
#include <nn/kernels/scalar_ops.def>
#undef NN_SCALAR
  }
  return "?";
}

inline constexpr int kUnaryOpCount = 0
#define NN_UNARY(Name, method, fwd, bwd) + 1
#include <nn/kernels/unary_ops.def>
#undef NN_UNARY
    ;
inline constexpr int kBinaryOpCount = 0
#define NN_BINARY(Name, method, fwd, da, db) + 1
#include <nn/kernels/binary_ops.def>
#undef NN_BINARY
    ;
inline constexpr int kScalarOpCount = 0
#define NN_SCALAR(Name, method, fwd, bwd) + 1
#include <nn/kernels/scalar_ops.def>
#undef NN_SCALAR
    ;

}  // namespace nn::kernels
