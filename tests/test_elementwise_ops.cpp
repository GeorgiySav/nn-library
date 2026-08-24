#include "test_harness.h"
#include "devices.h"
#include "gradcheck.h"

#include <cmath>
#include <vector>

#include <nn/autograd/functions.h>
#include <nn/autograd/tape.h>
#include <nn/core/tensor.h>
#include <nn/ops/ops.h>

namespace {

using nn::kernels::BinaryOp;
using nn::kernels::ScalarOp;
using nn::kernels::UnaryOp;

struct UnaryCase { UnaryOp op; const char* name; nn::kernels::UnaryNeeds needs; };
struct BinaryCase { BinaryOp op; const char* name; };
struct ScalarCase { ScalarOp op; const char* name; };

const UnaryCase kUnaryOps[] = {
#define NN_UNARY(Name, method, fwd, bwd, needs) {UnaryOp::Name, #method, nn::kernels::UnaryNeeds::needs},
#include <nn/kernels/unary_ops.def>
#undef NN_UNARY
};

const BinaryCase kBinaryOps[] = {
#define NN_BINARY(Name, method, fwd, da, db) {BinaryOp::Name, #method},
#include <nn/kernels/binary_ops.def>
#undef NN_BINARY
};

const ScalarCase kScalarOps[] = {
#define NN_SCALAR(Name, method, fwd, bwd) {ScalarOp::Name, #method},
#include <nn/kernels/scalar_ops.def>
#undef NN_SCALAR
};

// The scalar operand used everywhere below. It sits inside the sample range so
// that clamp_min/clamp_max and the comparisons see both sides of their kink,
// but no sample comes within a finite-difference step of it.
constexpr float kK = 0.9f;

// Strictly positive and away from zero, so every op in the three lists is
// inside its domain: log, sqrt, rsqrt, recip and pow all need that, and div
// needs a denominator that is not near zero.
const std::vector<float> kSafeA{0.31f, 0.52f, 0.73f, 1.14f, 1.35f, 1.66f, 0.44f, 1.51f};
const std::vector<float> kSafeB{1.21f, 0.67f, 1.43f, 0.38f, 1.58f, 0.85f, 1.32f, 0.59f};

// Deliberately outside several domains. The backends are asked to agree, not
// to produce a number: log of a negative must be NaN on both, not NaN on one.
const std::vector<float> kWideA{-3.5f, -1.0f, -0.25f, -1e-3f, 0.0f,
                                1e-3f, 0.25f, 1.0f, 3.5f, 12.0f};
const std::vector<float> kWideB{2.0f, 0.0f, 3.5f, -1.0f, 1e-3f,
                                -0.25f, 12.0f, -3.5f, 0.25f, -1e-3f};

std::vector<float> host_of(const nn::Tensor& t) {
  const nn::Tensor h = t.pack().to(nn::Device::CPU);
  return std::vector<float>(h.host_data(), h.host_data() + h.numel());
}

nn::Tensor tensor_of(const std::vector<float>& v, nn::Device d) {
  return nn::Tensor::from(v, nn::Shape({int(v.size())}), d);
}

// Two floats produced by two different backends from the same expression.
// They have to be the same kind of value, and if that kind is a number, close.
void check_agrees(float a, float b, const char* op, int i) {
  const std::string where = std::string(op) + "[" + std::to_string(i) + "]";
  if (std::isnan(a) || std::isnan(b)) {
    NN_CHECK(std::isnan(a) && std::isnan(b));
    return;
  }
  if (std::isinf(a) || std::isinf(b)) {
    NN_CHECK(std::isinf(a) && std::isinf(b) && (a > 0) == (b > 0));
    return;
  }
  NN_CHECK_CLOSE(a, b, 2e-5);
}

bool have_cuda() {
  for (nn::Device d : nn::test::devices()) {
    if (d == nn::Device::CUDA) return true;
  }
  return false;
}

// dL/dx for L = sum(op(x) * w). w is a plain tensor with no gradient of its
// own, so the upstream gradient reaching op is w -- not the all-ones vector a
// bare sum would hand it, which would let an op that ignored g still pass.
const std::vector<float> kWeights{1.7f, -0.4f, 0.9f, 2.3f, -1.1f, 0.6f, -2.0f, 1.3f};

float check_gradient(nn::Tensor& x, const std::function<nn::Tensor()>& forward,
                     nn::Device dev) {
  const nn::Tensor w = tensor_of(kWeights, dev);

  nn::autograd::Tape tape;
  nn::Tensor loss;
  auto loss_fn = [&]() -> float {
    tape.clear();
    nn::autograd::TapeScope scope(tape);
    loss = nn::autograd::sum_all(nn::autograd::mul(forward(), w));
    return loss.item();
  };
  auto backward_fn = [&]() { tape.backward(loss, true); };

  return nn::test::gradCheck(x, loss_fn, backward_fn, /*num_checks=*/8);
}

}  // namespace

NN_TEST(elementwise_lists_are_not_empty) {
  // A guard on the generation itself: if a .def stopped being included, every
  // loop below would pass by iterating nothing.
  NN_CHECK(std::size(kUnaryOps) == size_t(nn::kernels::kUnaryOpCount));
  NN_CHECK(std::size(kBinaryOps) == size_t(nn::kernels::kBinaryOpCount));
  NN_CHECK(std::size(kScalarOps) == size_t(nn::kernels::kScalarOpCount));
  NN_CHECK(nn::kernels::kUnaryOpCount >= 12);
  NN_CHECK(nn::kernels::kBinaryOpCount >= 7);
  NN_CHECK(nn::kernels::kScalarOpCount >= 12);
}

// The Needs column is hand-written, separately from the arithmetic it
// describes, so nothing stops it drifting out of sync with a derivative edit
// -- understating it silently reads the wrong value (0.0f in place of the
// real x or y), overstating it just wastes memory. This probes the actual
// arithmetic directly: apply_unary_backward is pure, so evaluating it at two
// values that differ in both sign and magnitude reveals real dependence on
// that argument, independent of what the column claims.
NN_TEST(unary_needs_matches_the_real_dependency) {
  using nn::kernels::apply_unary_backward;
  using nn::kernels::needs_x;
  using nn::kernels::needs_y;

  constexpr float kG = 1.3f;
  // Differ in sign and in magnitude, so both a step function (Abs, Relu) and
  // a continuous one (Log, Gelu, ...) reveal dependence either way.
  constexpr float kXa = -1.7f, kXb = 2.3f;
  constexpr float kYa = -0.6f, kYb = 0.85f;

  for (const UnaryCase& c : kUnaryOps) {
    const bool depends_x = apply_unary_backward(c.op, kXa, kYa, kG) !=
                           apply_unary_backward(c.op, kXb, kYa, kG);
    const bool depends_y = apply_unary_backward(c.op, kXa, kYa, kG) !=
                           apply_unary_backward(c.op, kXa, kYb, kG);

    if (depends_x != needs_x(c.needs)) {
      nn::test::report(__FILE__, __LINE__, std::string(c.name) +
          ": Needs says x is " + (needs_x(c.needs) ? "read" : "not read") +
          " but the derivative " + (depends_x ? "does" : "does not") + " depend on it");
    }
    if (depends_y != needs_y(c.needs)) {
      nn::test::report(__FILE__, __LINE__, std::string(c.name) +
          ": Needs says y is " + (needs_y(c.needs) ? "read" : "not read") +
          " but the derivative " + (depends_y ? "does" : "does not") + " depend on it");
    }
  }
}

// Both backends compile the same expression out of the .def, so this is really
// asking whether one of them mis-registered a slot or mis-ordered an argument.
NN_TEST(unary_ops_agree_across_backends) {
  if (!have_cuda()) return;

  const nn::Tensor cpu = tensor_of(kWideA, nn::Device::CPU);
  const nn::Tensor gpu = tensor_of(kWideA, nn::Device::CUDA);

  for (const UnaryCase& c : kUnaryOps) {
    const std::vector<float> a = host_of(nn::ops::unary(c.op, cpu));
    const std::vector<float> b = host_of(nn::ops::unary(c.op, gpu));
    for (size_t i = 0; i < a.size(); ++i) check_agrees(a[i], b[i], c.name, int(i));
  }
}

NN_TEST(binary_ops_agree_across_backends) {
  if (!have_cuda()) return;

  const nn::Tensor ca = tensor_of(kWideA, nn::Device::CPU);
  const nn::Tensor cb = tensor_of(kWideB, nn::Device::CPU);
  const nn::Tensor ga = tensor_of(kWideA, nn::Device::CUDA);
  const nn::Tensor gb = tensor_of(kWideB, nn::Device::CUDA);

  for (const BinaryCase& c : kBinaryOps) {
    const std::vector<float> a = host_of(nn::ops::binary(c.op, ca, cb));
    const std::vector<float> b = host_of(nn::ops::binary(c.op, ga, gb));
    for (size_t i = 0; i < a.size(); ++i) check_agrees(a[i], b[i], c.name, int(i));
  }
}

NN_TEST(scalar_ops_agree_across_backends) {
  if (!have_cuda()) return;

  const nn::Tensor cpu = tensor_of(kWideA, nn::Device::CPU);
  const nn::Tensor gpu = tensor_of(kWideA, nn::Device::CUDA);

  for (const ScalarCase& c : kScalarOps) {
    const std::vector<float> a = host_of(nn::ops::scalar(c.op, cpu, kK));
    const std::vector<float> b = host_of(nn::ops::scalar(c.op, gpu, kK));
    for (size_t i = 0; i < a.size(); ++i) check_agrees(a[i], b[i], c.name, int(i));
  }
}

NN_TEST(unary_ops_gradcheck) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    for (const UnaryCase& c : kUnaryOps) {
      nn::Tensor x = tensor_of(kSafeA, dev);
      x.set_requires_grad(true);
      const float err = check_gradient(x, [&] { return nn::autograd::unary(c.op, x); }, dev);
      if (err >= 2e-2f) {
        ::nn::test::report(__FILE__, __LINE__,
            std::string("unary ") + c.name + ": gradient error " + std::to_string(err));
      }
    }
  }
}

NN_TEST(binary_ops_gradcheck) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    for (const BinaryCase& c : kBinaryOps) {
      // Both operands, because d/da and d/db are separate columns in the .def
      // and a swapped pair would otherwise go unnoticed.
      for (int side = 0; side < 2; ++side) {
        nn::Tensor a = tensor_of(kSafeA, dev);
        nn::Tensor b = tensor_of(kSafeB, dev);
        nn::Tensor& target = (side == 0) ? a : b;
        target.set_requires_grad(true);

        const float err = check_gradient(
            target, [&] { return nn::autograd::binary(c.op, a, b); }, dev);
        if (err >= 2e-2f) {
          ::nn::test::report(__FILE__, __LINE__,
              std::string("binary ") + c.name + " side " + std::to_string(side) +
              ": gradient error " + std::to_string(err));
        }
      }
    }
  }
}

NN_TEST(scalar_ops_gradcheck) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    for (const ScalarCase& c : kScalarOps) {
      nn::Tensor x = tensor_of(kSafeA, dev);
      x.set_requires_grad(true);
      const float err =
          check_gradient(x, [&] { return nn::autograd::scalar(c.op, x, kK); }, dev);
      if (err >= 2e-2f) {
        ::nn::test::report(__FILE__, __LINE__,
            std::string("scalar ") + c.name + ": gradient error " + std::to_string(err));
      }
    }
  }
}

// The generated checks above run on a domain with no kinks in it, which is what
// makes a finite difference meaningful. These pin the pieces that domain skips.
NN_TEST(kinked_ops_have_the_right_values) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    const nn::Tensor x = nn::Tensor::from({-2.0f, -0.5f, 0.0f, 0.5f, 2.0f}, dev);

    const std::vector<float> relu = host_of(nn::ops::unary(UnaryOp::Relu, x));
    const std::vector<float> want_relu{0.0f, 0.0f, 0.0f, 0.5f, 2.0f};
    for (size_t i = 0; i < want_relu.size(); ++i) NN_CHECK(relu[i] == want_relu[i]);

    const std::vector<float> sign = host_of(nn::ops::unary(UnaryOp::Sign, x));
    const std::vector<float> want_sign{-1.0f, -1.0f, 0.0f, 1.0f, 1.0f};
    for (size_t i = 0; i < want_sign.size(); ++i) NN_CHECK(sign[i] == want_sign[i]);

    const std::vector<float> ab = host_of(nn::ops::unary(UnaryOp::Abs, x));
    const std::vector<float> want_abs{2.0f, 0.5f, 0.0f, 0.5f, 2.0f};
    for (size_t i = 0; i < want_abs.size(); ++i) NN_CHECK(ab[i] == want_abs[i]);

    // Predicates are exactly 0 or 1, never something that merely rounds to it.
    const std::vector<float> gt = host_of(nn::ops::scalar(ScalarOp::GtScalar, x, 0.0f));
    const std::vector<float> want_gt{0.0f, 0.0f, 0.0f, 1.0f, 1.0f};
    for (size_t i = 0; i < want_gt.size(); ++i) NN_CHECK(gt[i] == want_gt[i]);

    const std::vector<float> ge = host_of(nn::ops::scalar(ScalarOp::GeScalar, x, 0.0f));
    const std::vector<float> want_ge{0.0f, 0.0f, 1.0f, 1.0f, 1.0f};
    for (size_t i = 0; i < want_ge.size(); ++i) NN_CHECK(ge[i] == want_ge[i]);

    const std::vector<float> cm = host_of(nn::ops::scalar(ScalarOp::ClampMin, x, -0.5f));
    const std::vector<float> want_cm{-0.5f, -0.5f, 0.0f, 0.5f, 2.0f};
    for (size_t i = 0; i < want_cm.size(); ++i) NN_CHECK(cm[i] == want_cm[i]);
  }
}

// Values a reader can check by hand, so the .def expressions are pinned to
// something other than themselves.
NN_TEST(named_ops_match_known_values) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    const nn::Tensor x = nn::Tensor::from({0.0f, 1.0f, -1.0f, 2.0f}, dev);

    const std::vector<float> sig = host_of(nn::ops::unary(UnaryOp::Sigmoid, x));
    NN_CHECK_CLOSE(sig[0], 0.5f, 1e-6);
    NN_CHECK_CLOSE(sig[1], 0.7310585786f, 1e-6);
    NN_CHECK_CLOSE(sig[2], 0.2689414214f, 1e-6);

    const std::vector<float> th = host_of(nn::ops::unary(UnaryOp::Tanh, x));
    NN_CHECK_CLOSE(th[0], 0.0f, 1e-6);
    NN_CHECK_CLOSE(th[1], 0.7615941560f, 1e-6);
    NN_CHECK_CLOSE(th[2], -0.7615941560f, 1e-6);

    // 0.5 * x * (1 + erf(x / sqrt(2)))
    const std::vector<float> ge = host_of(nn::ops::unary(UnaryOp::Gelu, x));
    NN_CHECK_CLOSE(ge[0], 0.0f, 1e-6);
    NN_CHECK_CLOSE(ge[1], 0.8413447461f, 1e-5);
    NN_CHECK_CLOSE(ge[2], -0.1586552539f, 1e-5);
    NN_CHECK_CLOSE(ge[3], 1.9544997361f, 1e-5);

    // exp and log undo each other, which no single-op table can fake.
    const nn::Tensor pos = nn::Tensor::from({0.25f, 1.0f, 3.0f, 7.5f}, dev);
    const std::vector<float> round =
        host_of(nn::ops::unary(UnaryOp::Log, nn::ops::unary(UnaryOp::Exp, pos)));
    const std::vector<float> src = host_of(pos);
    for (size_t i = 0; i < src.size(); ++i) NN_CHECK_CLOSE(round[i], src[i], 1e-5);

    const std::vector<float> rs = host_of(nn::ops::unary(UnaryOp::Rsqrt, pos));
    for (size_t i = 0; i < src.size(); ++i) {
      NN_CHECK_CLOSE(rs[i], 1.0f / std::sqrt(src[i]), 1e-6);
    }
  }
}

// Elementwise inputs go in as views, so a strided operand has to give the same
// answer as the materialised one -- for every op, not just the ones that had a
// hand-written strided sibling before.
NN_TEST(elementwise_absorbs_strides) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    // [4, 7] sliced to [4, 4]: a row stride of 7 with a contiguous inner axis.
    std::vector<float> raw(28);
    for (size_t i = 0; i < raw.size(); ++i) raw[i] = 0.3f + 0.11f * float(i);
    const nn::Tensor wide = nn::Tensor::from(raw, nn::Shape({4, 7}), dev);
    const nn::Tensor view = wide.slice_view(1, 1, 4);
    NN_CHECK(!view.is_contiguous());
    const nn::Tensor dense = view.pack();

    for (const UnaryCase& c : kUnaryOps) {
      const std::vector<float> a = host_of(nn::ops::unary(c.op, view));
      const std::vector<float> b = host_of(nn::ops::unary(c.op, dense));
      for (size_t i = 0; i < a.size(); ++i) check_agrees(a[i], b[i], c.name, int(i));
    }
    for (const ScalarCase& c : kScalarOps) {
      const std::vector<float> a = host_of(nn::ops::scalar(c.op, view, kK));
      const std::vector<float> b = host_of(nn::ops::scalar(c.op, dense, kK));
      for (size_t i = 0; i < a.size(); ++i) check_agrees(a[i], b[i], c.name, int(i));
    }
    // A transposed second operand as well, so the two views differ from each
    // other and not only from the dense layout.
    const nn::Tensor other = wide.slice_view(1, 2, 4);
    for (const BinaryCase& c : kBinaryOps) {
      const std::vector<float> a = host_of(nn::ops::binary(c.op, view, other));
      const std::vector<float> b =
          host_of(nn::ops::binary(c.op, dense, other.pack()));
      for (size_t i = 0; i < a.size(); ++i) check_agrees(a[i], b[i], c.name, int(i));
    }
  }
}

// The measured reason the elementwise family can afford to be view-only.
NN_TEST(view_of_collapses_contiguous_axes) {
  const nn::Tensor dense = nn::Tensor::zeros({4, 5, 6}, nn::Device::CPU);
  const nn::TensorView v = nn::view_of(dense);
  NN_CHECK(v.rank == 1);
  NN_CHECK(v.shape[0] == 120 && v.stride[0] == 1);

  // A row stride survives as a second axis; the inner run still merges.
  const nn::Tensor rows = nn::Tensor::zeros({4, 5, 8}, nn::Device::CPU).slice_view(2, 0, 6);
  const nn::TensorView vr = nn::view_of(rows);
  NN_CHECK(vr.rank == 2);
  NN_CHECK(vr.shape[0] == 20 && vr.stride[0] == 8);
  NN_CHECK(vr.shape[1] == 6 && vr.stride[1] == 1);

  // Size-1 axes carry no index at all, so they disappear entirely.
  const nn::Tensor squeezy = nn::Tensor::zeros({1, 7, 1}, nn::Device::CPU);
  NN_CHECK(nn::view_of(squeezy).rank == 1);

  // A broadcast axis merges with its neighbour only when that one is also
  // stride 0, which keeps every index pointing at the same element either way.
  const nn::Tensor stretched =
      nn::Tensor::zeros({4}, nn::Device::CPU).expand_view(nn::Shape({2, 3, 4}));
  const nn::TensorView vs = nn::view_of(stretched);
  NN_CHECK(vs.rank == 2);
  NN_CHECK(vs.shape[0] == 6 && vs.stride[0] == 0);
  NN_CHECK(vs.shape[1] == 4 && vs.stride[1] == 1);
}
