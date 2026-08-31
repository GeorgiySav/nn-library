#pragma once

#include <kernels/ew_inline.h>
#include <nn/ops/op_enums.h>

// forward/backward arithmetic for every unary op, one struct per op. each
// fwd/bwd is NN_EW_INLINE, so it compiles once and is reused as plain host
// code (cpu backend) and as __host__ __device__ code (CUDA backend), one
// definition for both backends.
//
// kNeeds says which of x, y the backward method actually reads, one of
// None, X, Y or Both.
//
// to add an op, add a struct below, add it to the All alias at the bottom,
// and add its (Name, method) row to unary_ops.def.

namespace nn::kernels::unary_ops {

struct Neg {
  static constexpr UnaryOp kOp = UnaryOp::Neg;
  static constexpr UnaryNeeds kNeeds = UnaryNeeds::None;
  NN_EW_INLINE static float fwd(float x) { return -x; }
  NN_EW_INLINE static float bwd(float, float, float g) { return -g; }
};

struct Abs {
  static constexpr UnaryOp kOp = UnaryOp::Abs;
  static constexpr UnaryNeeds kNeeds = UnaryNeeds::X;
  NN_EW_INLINE static float fwd(float x) { return fabsf(x); }
  NN_EW_INLINE static float bwd(float x, float, float g) { return x < 0.0f ? -g : g; }
};

struct Sign {
  static constexpr UnaryOp kOp = UnaryOp::Sign;
  static constexpr UnaryNeeds kNeeds = UnaryNeeds::None;
  NN_EW_INLINE static float fwd(float x) { return float((x > 0.0f) - (x < 0.0f)); }
  NN_EW_INLINE static float bwd(float, float, float g) { return 0.0f * g; }
};

struct Relu {
  static constexpr UnaryOp kOp = UnaryOp::Relu;
  static constexpr UnaryNeeds kNeeds = UnaryNeeds::X;
  NN_EW_INLINE static float fwd(float x) { return fmaxf(x, 0.0f); }
  NN_EW_INLINE static float bwd(float x, float, float g) { return x > 0.0f ? g : 0.0f; }
};

struct Exp {
  static constexpr UnaryOp kOp = UnaryOp::Exp;
  static constexpr UnaryNeeds kNeeds = UnaryNeeds::Y;
  NN_EW_INLINE static float fwd(float x) { return expf(x); }
  NN_EW_INLINE static float bwd(float, float y, float g) { return g * y; }
};

struct Log {
  static constexpr UnaryOp kOp = UnaryOp::Log;
  static constexpr UnaryNeeds kNeeds = UnaryNeeds::X;
  NN_EW_INLINE static float fwd(float x) { return logf(x); }
  NN_EW_INLINE static float bwd(float x, float, float g) { return g / x; }
};

struct Sqrt {
  static constexpr UnaryOp kOp = UnaryOp::Sqrt;
  static constexpr UnaryNeeds kNeeds = UnaryNeeds::Y;
  NN_EW_INLINE static float fwd(float x) { return sqrtf(x); }
  NN_EW_INLINE static float bwd(float, float y, float g) { return g * 0.5f / y; }
};

struct Rsqrt {
  static constexpr UnaryOp kOp = UnaryOp::Rsqrt;
  static constexpr UnaryNeeds kNeeds = UnaryNeeds::Y;
  NN_EW_INLINE static float fwd(float x) { return 1.0f / sqrtf(x); }
  NN_EW_INLINE static float bwd(float, float y, float g) { return -0.5f * g * y * y * y; }
};

struct Recip {
  static constexpr UnaryOp kOp = UnaryOp::Recip;
  static constexpr UnaryNeeds kNeeds = UnaryNeeds::Y;
  NN_EW_INLINE static float fwd(float x) { return 1.0f / x; }
  NN_EW_INLINE static float bwd(float, float y, float g) { return -g * y * y; }
};

struct Tanh {
  static constexpr UnaryOp kOp = UnaryOp::Tanh;
  static constexpr UnaryNeeds kNeeds = UnaryNeeds::Y;
  NN_EW_INLINE static float fwd(float x) { return tanhf(x); }
  NN_EW_INLINE static float bwd(float, float y, float g) { return g * (1.0f - y * y); }
};

struct Sigmoid {
  static constexpr UnaryOp kOp = UnaryOp::Sigmoid;
  static constexpr UnaryNeeds kNeeds = UnaryNeeds::Y;
  NN_EW_INLINE static float fwd(float x) { return 1.0f / (1.0f + expf(-x)); }
  NN_EW_INLINE static float bwd(float, float y, float g) { return g * y * (1.0f - y); }
};

// exact GELU (using erf), not the tanh approximation
struct Gelu {
  static constexpr UnaryOp kOp = UnaryOp::Gelu;
  static constexpr UnaryNeeds kNeeds = UnaryNeeds::X;
  NN_EW_INLINE static float fwd(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.70710678118654752f));
  }
  NN_EW_INLINE static float bwd(float x, float, float g) {
    return g * (0.5f * (1.0f + erff(x * 0.70710678118654752f)) +
                x * 0.39894228040143268f * expf(-0.5f * x * x));
  }
};

// silu(x) = x * sigmoid(x)
struct Silu {
  static constexpr UnaryOp kOp = UnaryOp::Silu;
  static constexpr UnaryNeeds kNeeds = UnaryNeeds::X;
  NN_EW_INLINE static float fwd(float x) { return x / (1.0f + expf(-x)); }
  NN_EW_INLINE static float bwd(float x, float, float g) {
    return g * (1.0f + x * (1.0f - 1.0f / (1.0f + expf(-x)))) / (1.0f + expf(-x));
  }
};

struct Sin {
  static constexpr UnaryOp kOp = UnaryOp::Sin;
  static constexpr UnaryNeeds kNeeds = UnaryNeeds::X;
  NN_EW_INLINE static float fwd(float x) { return sinf(x); }
  NN_EW_INLINE static float bwd(float x, float, float g) { return g * cosf(x); }
};

struct Cos {
  static constexpr UnaryOp kOp = UnaryOp::Cos;
  static constexpr UnaryNeeds kNeeds = UnaryNeeds::X;
  NN_EW_INLINE static float fwd(float x) { return cosf(x); }
  NN_EW_INLINE static float bwd(float x, float, float g) { return -g * sinf(x); }
};

namespace detail {

// walks the op list Ops... one at a time, comparing each Op::kOp against the
// runtime op code until it matches; recursion bottoms out at the last type
// in the pack rather than an explicit base case.
template <class Op, class... Rest>
NN_EW_INLINE float dispatch_fwd(UnaryOp op, float x) {
  if (op == Op::kOp) return Op::fwd(x);
  if constexpr (sizeof...(Rest) > 0) return dispatch_fwd<Rest...>(op, x);
  else return 0.0f;
}

template <class Op, class... Rest>
NN_EW_INLINE float dispatch_bwd(UnaryOp op, float x, float y, float g) {
  if (op == Op::kOp) return Op::bwd(x, y, g);
  if constexpr (sizeof...(Rest) > 0) return dispatch_bwd<Rest...>(op, x, y, g);
  else return 0.0f;
}

template <class Op, class... Rest>
inline UnaryNeeds dispatch_needs(UnaryOp op) {
  if (op == Op::kOp) return Op::kNeeds;
  if constexpr (sizeof...(Rest) > 0) return dispatch_needs<Rest...>(op);
  else return UnaryNeeds::Both;  // unknown op, so keep everything rather than drop a gradient
}

}  // namespace detail

template <class... Ops>
struct UnaryOpList {
  static constexpr int kSize = sizeof...(Ops);
  NN_EW_INLINE static float fwd(UnaryOp op, float x) {
    return detail::dispatch_fwd<Ops...>(op, x);
  }
  NN_EW_INLINE static float bwd(UnaryOp op, float x, float y, float g) {
    return detail::dispatch_bwd<Ops...>(op, x, y, g);
  }
  static UnaryNeeds needs(UnaryOp op) { return detail::dispatch_needs<Ops...>(op); }
};

using All = UnaryOpList<Neg, Abs, Sign, Relu, Exp, Log, Sqrt, Rsqrt, Recip,
                        Tanh, Sigmoid, Gelu, Silu, Sin, Cos>;

}  // namespace nn::kernels::unary_ops
