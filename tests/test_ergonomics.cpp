#include "test_harness.h"
#include "devices.h"
#include "gradcheck.h"

#include <stdexcept>
#include <vector>

#include <nn/autograd/functions.h>
#include <nn/autograd/tape.h>
#include <nn/core/rng.h>
#include <nn/core/tensor.h>
#include <nn/module.h>
#include <nn/ops/ops.h>

namespace {

std::vector<float> host_of(const nn::Tensor& t) {
  const nn::Tensor h = t.pack().to(nn::Device::CPU);
  return std::vector<float>(h.host_data(), h.host_data() + h.numel());
}

void same_values(const nn::Tensor& a, const nn::Tensor& b, float tol = 1e-6f) {
  NN_CHECK(a.shape() == b.shape());
  const std::vector<float> ha = host_of(a), hb = host_of(b);
  for (size_t i = 0; i < ha.size(); ++i) NN_CHECK_CLOSE(ha[i], hb[i], tol);
}

nn::Tensor ramp(nn::Shape s, nn::Device d, float start, float step) {
  std::vector<float> v(size_t(s.numel()));
  for (size_t i = 0; i < v.size(); ++i) v[i] = start + step * float(i);
  return nn::Tensor::from(v, s, d);
}

}  // namespace

NN_TEST(tensor_operators_match_the_ops_they_stand_for) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    const nn::Tensor a = ramp(nn::Shape({3, 4}), dev, 0.7f, 0.13f);
    const nn::Tensor b = ramp(nn::Shape({3, 4}), dev, 1.9f, -0.07f);

    same_values(a + b, nn::ops::binary(nn::ops::BinaryOp::Add, a, b));
    same_values(a - b, nn::ops::binary(nn::ops::BinaryOp::Sub, a, b));
    same_values(a * b, nn::ops::binary(nn::ops::BinaryOp::Mul, a, b));
    same_values(a / b, nn::ops::binary(nn::ops::BinaryOp::Div, a, b));

    same_values(-a, nn::ops::unary(nn::ops::UnaryOp::Neg, a));
    same_values(a.relu(), nn::ops::relu(a));
  }
}

NN_TEST(scalar_operators_cover_both_orders) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    const nn::Tensor x = ramp(nn::Shape({6}), dev, 0.4f, 0.3f);
    const std::vector<float> h = host_of(x);

    const std::vector<float> plus  = host_of(x + 2.0f);
    const std::vector<float> rplus = host_of(2.0f + x);
    const std::vector<float> minus = host_of(x - 2.0f);
    const std::vector<float> rminus= host_of(2.0f - x);
    const std::vector<float> times = host_of(x * 3.0f);
    const std::vector<float> rtimes= host_of(3.0f * x);
    const std::vector<float> over  = host_of(x / 4.0f);
    const std::vector<float> rover = host_of(4.0f / x);

    for (size_t i = 0; i < h.size(); ++i) {
      NN_CHECK_CLOSE(plus[i],   h[i] + 2.0f, 1e-6);
      NN_CHECK_CLOSE(rplus[i],  h[i] + 2.0f, 1e-6);
      NN_CHECK_CLOSE(minus[i],  h[i] - 2.0f, 1e-6);
      NN_CHECK_CLOSE(rminus[i], 2.0f - h[i], 1e-6);
      NN_CHECK_CLOSE(times[i],  h[i] * 3.0f, 1e-6);
      NN_CHECK_CLOSE(rtimes[i], h[i] * 3.0f, 1e-6);
      NN_CHECK_CLOSE(over[i],   h[i] / 4.0f, 1e-6);
      NN_CHECK_CLOSE(rover[i],  4.0f / h[i], 1e-5);
    }
  }
}

// The operators broadcast because the ops under them do; nothing extra is
// needed to write `logits + bias`.
NN_TEST(operators_broadcast) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    const nn::Tensor x = ramp(nn::Shape({4, 3}), dev, 1.0f, 1.0f);
    const nn::Tensor row = ramp(nn::Shape({3}), dev, 10.0f, 10.0f);

    const nn::Tensor sum = x + row;
    NN_CHECK(sum.shape() == nn::Shape({4, 3}));
    const std::vector<float> hs = host_of(sum);
    const std::vector<float> hx = host_of(x), hr = host_of(row);
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 3; ++j)
        NN_CHECK_CLOSE(hs[size_t(i) * 3 + j], hx[size_t(i) * 3 + j] + hr[j], 1e-6);
  }
}

NN_TEST(methods_chain_left_to_right) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    nn::Pcg32 rng(5);
    const nn::Tensor x = nn::Tensor::randn({4, 6}, rng, 1.0f, dev);
    const nn::Tensor w = nn::Tensor::randn({6, 3}, rng, 1.0f, dev);
    const nn::Tensor b = nn::Tensor::randn({3}, rng, 1.0f, dev);

    // x.mm(w) + b, then relu, spelled both ways
    const nn::Tensor chained = (x.mm(w) + b).relu();
    const nn::Tensor spelled = nn::autograd::relu(
        nn::autograd::add(nn::autograd::matmul(x, w), b));
    same_values(chained, spelled);

    // t() swaps the last two axes and is a view, not a copy
    const nn::Tensor xt = x.t();
    NN_CHECK(xt.shape() == nn::Shape({6, 4}));
    same_values(xt.pack(), x.transpose(0, 1).pack());
  }
}

NN_TEST(tensor_backward_matches_the_tape_call) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    nn::Pcg32 rng(31);
    const nn::Tensor xs = nn::Tensor::randn({4, 5}, rng, 1.0f, dev);
    const nn::Tensor labels = nn::Tensor::from_i32({0, 2, 4, 1}, dev);

    std::vector<float> viaTape, viaTensor;

    {   // the old spelling
      nn::Tensor w = ramp(nn::Shape({5, 5}), dev, 0.1f, 0.03f);
      w.set_requires_grad(true);
      nn::autograd::Tape tape;
      nn::Tensor loss;
      {
        nn::autograd::TapeScope scope(tape);
        loss = nn::cross_entropy(xs.mm(w), labels);
      }
      tape.backward(loss);
      viaTape = host_of(w.grad());
    }
    {   // the new one
      nn::Tensor w = ramp(nn::Shape({5, 5}), dev, 0.1f, 0.03f);
      w.set_requires_grad(true);
      nn::autograd::GradScope grad;
      nn::Tensor loss = nn::cross_entropy(xs.mm(w), labels);
      loss.backward();
      viaTensor = host_of(w.grad());
    }

    NN_CHECK(viaTape.size() == viaTensor.size());
    for (size_t i = 0; i < viaTape.size(); ++i) {
      NN_CHECK_CLOSE(viaTape[i], viaTensor[i], 1e-6);
    }
  }
}

// A tensor finds its own tape from the epoch already stamped in its meta, so
// the scope does not have to still be open -- only the tape has to still exist.
NN_TEST(backward_works_after_the_scope_closes) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    nn::Tensor x = ramp(nn::Shape({4}), dev, 0.5f, 0.25f);
    x.set_requires_grad(true);

    nn::autograd::Tape tape;
    nn::Tensor loss;
    {
      nn::autograd::TapeScope scope(tape);
      loss = (x * x).sum();
    }
    NN_CHECK(nn::autograd::active_tape() == nullptr);

    loss.backward();

    const std::vector<float> g = host_of(x.grad());
    const std::vector<float> h = host_of(x);
    for (size_t i = 0; i < g.size(); ++i) NN_CHECK_CLOSE(g[i], 2.0f * h[i], 1e-5);
  }
}

NN_TEST(backward_says_so_when_there_is_no_tape_to_walk) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    nn::Tensor x = ramp(nn::Shape({4}), dev, 0.5f, 0.25f);
    x.set_requires_grad(true);

    // Never recorded: no tape was active.
    const nn::Tensor loose = (x * x).sum();
    NN_CHECK_THROWS(loose.backward(), std::invalid_argument);

    // Recorded, but on a tape that has since been destroyed. The epoch in the
    // meta no longer resolves, which is the point of looking it up rather than
    // holding a pointer.
    nn::Tensor orphan;
    {
      nn::autograd::GradScope grad;
      orphan = (x * x).sum();
    }
    NN_CHECK_THROWS(orphan.backward(), std::invalid_argument);

    // Same again for a tape that is alive but has been cleared.
    nn::autograd::Tape tape;
    nn::Tensor stale;
    {
      nn::autograd::TapeScope scope(tape);
      stale = (x * x).sum();
    }
    tape.clear();
    NN_CHECK_THROWS(stale.backward(), std::invalid_argument);
  }
}

NN_TEST(grad_scope_refuses_to_nest) {
  nn::autograd::GradScope outer;
  NN_CHECK(nn::autograd::active_tape() == &outer.tape());
  NN_CHECK_THROWS(([] { nn::autograd::GradScope inner; }()), std::logic_error);
  // The failed inner scope must not have disturbed the outer one.
  NN_CHECK(nn::autograd::active_tape() == &outer.tape());
}

NN_TEST(retain_graph_lets_backward_run_twice) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    nn::Tensor x = ramp(nn::Shape({4}), dev, 0.5f, 0.25f);
    x.set_requires_grad(true);

    nn::autograd::GradScope grad;
    const nn::Tensor loss = (x * 3.0f).sum();
    loss.backward(/*retain_graph=*/true);
    const std::vector<float> once = host_of(x.grad());
    loss.backward(/*retain_graph=*/true);
    const std::vector<float> twice = host_of(x.grad());

    for (size_t i = 0; i < once.size(); ++i) {
      NN_CHECK_CLOSE(once[i], 3.0f, 1e-6);
      NN_CHECK_CLOSE(twice[i], 2.0f * once[i], 1e-6);
    }
  }
}

NN_TEST(module_is_callable) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    nn::Pcg32 rng(37);
    nn::Sequential model(nn::Linear(6, 8, rng), nn::ReLu(), nn::Linear(8, 3, rng));
    model.to(dev);

    const nn::Tensor x = nn::Tensor::randn({4, 6}, rng, 1.0f, dev);
    same_values(model(x), model.forward(x));
  }
}

NN_TEST(the_new_layers_train_end_to_end) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    nn::Pcg32 rng(41);
    nn::Sequential model(nn::Linear(5, 6, rng), nn::LayerNorm(6), nn::GeLu(),
                         nn::Linear(6, 3, rng));
    model.to(dev);

    const nn::Tensor x = nn::Tensor::randn({4, 5}, rng, 1.0f, dev);
    const nn::Tensor labels = nn::Tensor::from_i32({0, 2, 1, 2}, dev);

    float first = 0.0f, last = 0.0f;
    for (int step = 0; step < 40; ++step) {
      model.zero_grad();
      nn::autograd::GradScope grad;
      const nn::Tensor loss = nn::cross_entropy(model(x), labels);
      loss.backward();

      const float value = loss.item();
      if (step == 0) first = value;
      last = value;

      // plain SGD, so the test depends on nothing but the gradients
      for (nn::Tensor* p : model.parameters()) {
        nn::ops::axpy_inplace(*p, -0.1f, p->grad());
      }
    }
    NN_CHECK(last < first);
    NN_CHECK(last < 0.2f * first);
  }
}

// Gradients have to survive the operator spelling, not just the function one.
// `a` appears twice in the expression, so this also checks that the two paths
// into it accumulate rather than overwrite.
//
// The scale is deliberately small. Drive the same expression with stddev 0.8
// and tanh saturates: the true gradient falls to ~1e-4 while the loss stays
// order 10, and a central difference at h = 1e-3 then measures rounding.
// Verified -- 0.29 error saturated, 6e-4 here, and 2e-4 with tanh removed
// entirely, which is what says the arithmetic is right and the conditioning
// was the problem.
NN_TEST(gradcheck_through_an_operator_expression) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    nn::Pcg32 rng(43);
    nn::Tensor a = nn::Tensor::randn({3, 4}, rng, 0.3f, dev);
    nn::Tensor b = nn::Tensor::randn({4}, rng, 0.3f, dev);
    a.set_requires_grad(true);
    b.set_requires_grad(true);

    nn::autograd::Tape tape;
    nn::Tensor loss;
    auto forward = [&]() -> float {
      tape.clear();
      nn::autograd::TapeScope scope(tape);
      // broadcast, scalar and unary forms in one expression
      loss = ((a * 2.0f + b) * (1.0f - a)).tanh().sum();
      return loss.item();
    };
    auto backward = [&]() { tape.backward(loss, true); };

    NN_CHECK(nn::test::gradCheck(a, forward, backward) < 2e-2f);
    NN_CHECK(nn::test::gradCheck(b, forward, backward) < 2e-2f);
  }
}
