#pragma once

#include <kernels/ew_inline.h>
#include <nn/ops/op_enums.h>

// Per-op forward/backward arithmetic for every "tensor combined with a host
// float" op. `y` is the forward result, `g` the incoming gradient, `k` the
// scalar operand. See unary_ops_traits.h for the dispatch mechanism and the
// "add an op" recipe.

namespace nn::kernels::scalar_ops {

struct AddScalar {
  static constexpr ScalarOp kOp = ScalarOp::AddScalar;
  NN_EW_INLINE static float fwd(float x, float k) { return x + k; }
  NN_EW_INLINE static float bwd(float, float, float g, float) { return g; }
};

struct MulScalar {
  static constexpr ScalarOp kOp = ScalarOp::MulScalar;
  NN_EW_INLINE static float fwd(float x, float k) { return x * k; }
  NN_EW_INLINE static float bwd(float, float, float g, float k) { return g * k; }
};

struct RsubScalar {
  static constexpr ScalarOp kOp = ScalarOp::RsubScalar;
  NN_EW_INLINE static float fwd(float x, float k) { return k - x; }
  NN_EW_INLINE static float bwd(float, float, float g, float) { return -g; }
};

struct RdivScalar {
  static constexpr ScalarOp kOp = ScalarOp::RdivScalar;
  NN_EW_INLINE static float fwd(float x, float k) { return k / x; }
  NN_EW_INLINE static float bwd(float x, float y, float g, float) { return -g * y / x; }
};

struct PowScalar {
  static constexpr ScalarOp kOp = ScalarOp::PowScalar;
  NN_EW_INLINE static float fwd(float x, float k) { return powf(x, k); }
  NN_EW_INLINE static float bwd(float x, float, float g, float k) {
    return g * k * powf(x, k - 1.0f);
  }
};

struct ClampMin {
  static constexpr ScalarOp kOp = ScalarOp::ClampMin;
  NN_EW_INLINE static float fwd(float x, float k) { return fmaxf(x, k); }
  NN_EW_INLINE static float bwd(float x, float, float g, float k) { return x > k ? g : 0.0f; }
};

struct ClampMax {
  static constexpr ScalarOp kOp = ScalarOp::ClampMax;
  NN_EW_INLINE static float fwd(float x, float k) { return fminf(x, k); }
  NN_EW_INLINE static float bwd(float x, float, float g, float k) { return x < k ? g : 0.0f; }
};

// Predicates, for building masks: a step function, so the gradient is zero
// everywhere it is defined.
struct GtScalar {
  static constexpr ScalarOp kOp = ScalarOp::GtScalar;
  NN_EW_INLINE static float fwd(float x, float k) { return float(x > k); }
  NN_EW_INLINE static float bwd(float, float, float g, float) { return 0.0f * g; }
};
struct GeScalar {
  static constexpr ScalarOp kOp = ScalarOp::GeScalar;
  NN_EW_INLINE static float fwd(float x, float k) { return float(x >= k); }
  NN_EW_INLINE static float bwd(float, float, float g, float) { return 0.0f * g; }
};
struct LtScalar {
  static constexpr ScalarOp kOp = ScalarOp::LtScalar;
  NN_EW_INLINE static float fwd(float x, float k) { return float(x < k); }
  NN_EW_INLINE static float bwd(float, float, float g, float) { return 0.0f * g; }
};
struct LeScalar {
  static constexpr ScalarOp kOp = ScalarOp::LeScalar;
  NN_EW_INLINE static float fwd(float x, float k) { return float(x <= k); }
  NN_EW_INLINE static float bwd(float, float, float g, float) { return 0.0f * g; }
};
struct EqScalar {
  static constexpr ScalarOp kOp = ScalarOp::EqScalar;
  NN_EW_INLINE static float fwd(float x, float k) { return float(x == k); }
  NN_EW_INLINE static float bwd(float, float, float g, float) { return 0.0f * g; }
};

namespace detail {

template <class Op, class... Rest>
NN_EW_INLINE float dispatch_fwd(ScalarOp op, float x, float k) {
  if (op == Op::kOp) return Op::fwd(x, k);
  if constexpr (sizeof...(Rest) > 0) return dispatch_fwd<Rest...>(op, x, k);
  else return 0.0f;
}

template <class Op, class... Rest>
NN_EW_INLINE float dispatch_bwd(ScalarOp op, float x, float y, float g, float k) {
  if (op == Op::kOp) return Op::bwd(x, y, g, k);
  if constexpr (sizeof...(Rest) > 0) return dispatch_bwd<Rest...>(op, x, y, g, k);
  else return 0.0f;
}

}  // namespace detail

template <class... Ops>
struct ScalarOpList {
  static constexpr int kSize = sizeof...(Ops);
  NN_EW_INLINE static float fwd(ScalarOp op, float x, float k) {
    return detail::dispatch_fwd<Ops...>(op, x, k);
  }
  NN_EW_INLINE static float bwd(ScalarOp op, float x, float y, float g, float k) {
    return detail::dispatch_bwd<Ops...>(op, x, y, g, k);
  }
};

using All = ScalarOpList<AddScalar, MulScalar, RsubScalar, RdivScalar, PowScalar,
                         ClampMin, ClampMax, GtScalar, GeScalar, LtScalar, LeScalar, EqScalar>;

}  // namespace nn::kernels::scalar_ops
