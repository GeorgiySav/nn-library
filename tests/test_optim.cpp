#include "test_harness.h"
#include "devices.h"

#include <cmath>
#include <stdexcept>
#include <vector>

#include <nn/autograd/functions.h>
#include <nn/autograd/tape.h>
#include <nn/core/tensor.h>
#include <nn/nn/module.h>
#include <nn/optim/optim.h>

namespace {

constexpr float kB1 = 0.9f, kB2 = 0.999f, kEps = 1e-8f;

std::vector<float> host_of(const nn::Tensor& t) {
  const nn::Tensor h = t.pack().to(nn::Device::CPU);
  return std::vector<float>(h.host_data(), h.host_data() + h.numel());
}

nn::Tensor with_grad(std::initializer_list<float> value, nn::Device d) {
  nn::Tensor p = nn::Tensor::from(value, nn::Device::CPU).to(d);
  p.set_requires_grad(true);
  return p;
}

void set_grad(nn::Tensor& p, std::initializer_list<float> g) {
  p.grad() = nn::Tensor::from(g, nn::Device::CPU).to(p.device());
}

struct Reference {
  double p, m = 0.0, v = 0.0;
  int t = 0;

  void step(double g, double lr, double wd = 0.0) {
    ++t;
    m = kB1 * m + (1.0 - kB1) * g;
    v = kB2 * v + (1.0 - kB2) * g * g;
    const double bc1 = 1.0 - std::pow(double(kB1), t);
    const double bc2 = 1.0 - std::pow(double(kB2), t);
    p = p * (1.0 - lr * wd) - lr * (m / bc1) / (std::sqrt(v / bc2) + kEps);
  }
};

}  // namespace

NN_TEST(adam_matches_the_reference_over_several_steps) {
  const float lr = 0.05f;
  const std::vector<float> grads{0.7f, -0.2f, 0.35f, 0.35f, -1.1f};

  NN_TEST_FOR_EACH_DEVICE(dev) {
    nn::Tensor p = with_grad({1.25f}, dev);
    nn::optim::Adam opt({&p}, lr, kB1, kB2, kEps);

    Reference ref{1.25};
    for (float g : grads) {
      set_grad(p, {g});
      opt.step();
      ref.step(g, lr);
      NN_CHECK_CLOSE(host_of(p)[0], float(ref.p), 2e-6f);
    }
  }
}

NN_TEST(adamw_matches_the_reference_with_decay) {
  const float lr = 0.05f, wd = 0.1f;
  const std::vector<float> grads{0.7f, -0.2f, 0.35f, 0.35f, -1.1f};

  NN_TEST_FOR_EACH_DEVICE(dev) {
    nn::Tensor p = with_grad({1.25f}, dev);
    // rank 1, so the default predicate would skip it -- decay everything here
    nn::optim::AdamW opt({&p}, lr, wd, kB1, kB2, kEps,
                         [](const nn::Tensor&) { return true; });

    Reference ref{1.25};
    for (float g : grads) {
      set_grad(p, {g});
      opt.step();
      ref.step(g, lr, wd);
      NN_CHECK_CLOSE(host_of(p)[0], float(ref.p), 2e-6f);
    }
  }
}

NN_TEST(adamw_decay_is_a_plain_geometric_shrink_when_the_gradient_is_zero) {
  const float lr = 0.1f, wd = 0.5f;
  const float factor = 1.0f - lr * wd;   // 0.95

  NN_TEST_FOR_EACH_DEVICE(dev) {
    nn::Tensor p = with_grad({2.0f, -4.0f}, dev);
    nn::optim::AdamW opt({&p}, lr, wd, kB1, kB2, kEps,
                         [](const nn::Tensor&) { return true; });

    float want_a = 2.0f, want_b = -4.0f;
    for (int i = 0; i < 4; ++i) {
      set_grad(p, {0.0f, 0.0f});
      opt.step();
      want_a *= factor;
      want_b *= factor;

      const std::vector<float> got = host_of(p);
      NN_CHECK_CLOSE(got[0], want_a, 1e-6f);
      NN_CHECK_CLOSE(got[1], want_b, 1e-6f);
    }
  }
}

NN_TEST(decoupled_decay_differs_from_l2_in_the_gradient) {
  const float lr = 0.05f, wd = 0.2f;
  const float p0 = 3.0f, g = 0.4f;

  NN_TEST_FOR_EACH_DEVICE(dev) {
    nn::Tensor decoupled = with_grad({p0}, dev);
    nn::optim::AdamW wopt({&decoupled}, lr, wd, kB1, kB2, kEps,
                          [](const nn::Tensor&) { return true; });

    nn::Tensor coupled = with_grad({p0}, dev);
    nn::optim::Adam aopt({&coupled}, lr, kB1, kB2, kEps);

    for (int i = 0; i < 3; ++i) {
      set_grad(decoupled, {g});
      wopt.step();

      // L2 the old way: fold wd * p into the gradient before the step
      set_grad(coupled, {g + wd * host_of(coupled)[0]});
      aopt.step();
    }

    const float w = host_of(decoupled)[0], a = host_of(coupled)[0];

    // the coupled run moved by 3*lr and no more: its decay was normalised away
    NN_CHECK_CLOSE(p0 - a, 3.0f * lr, 1e-3f);
    // the decoupled run moved further, by the geometric shrink on top
    NN_CHECK(w < a - 1e-2f);
    NN_CHECK_CLOSE(p0 - w, 3.0f * lr + 0.0877f, 1e-3f);
  }
}

NN_TEST(weight_decay_skips_biases_and_gains_by_default) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    nn::Pcg32 rng(11);
    nn::Linear fc(3, 4, rng);       // w is [3,4], b is [4]
    nn::LayerNorm ln(4);            // gain and bias are both [4]
    nn::Embedding emb(5, 4, rng);   // w is [5,4]
    fc.to(dev); ln.to(dev); emb.to(dev);

    std::vector<nn::Tensor*> params;
    fc.collect_parameters(params);
    ln.collect_parameters(params);
    emb.collect_parameters(params);
    NN_CHECK(params.size() == 5);

    nn::optim::AdamW opt(params, 1e-3f, 0.1f);
    const std::span<const float> wd = opt.weight_decays();

    for (size_t i = 0; i < params.size(); ++i) {
      const float want = params[i]->shape().rank() >= 2 ? 0.1f : 0.0f;
      NN_CHECK_CLOSE(wd[i], want, 0.0f);
    }
    NN_CHECK_CLOSE(wd[0], 0.1f, 0.0f);   // fc.w  [3,4]
    NN_CHECK_CLOSE(wd[1], 0.0f, 0.0f);   // fc.b  [4]
    NN_CHECK_CLOSE(wd[4], 0.1f, 0.0f);   // emb.w [5,4]

    // and it is only the predicate: opting everything in is one argument
    nn::optim::AdamW all(params, 1e-3f, 0.1f, 0.9f, 0.999f, 1e-8f,
                         [](const nn::Tensor&) { return true; });
    for (float w : all.weight_decays()) NN_CHECK_CLOSE(w, 0.1f, 0.0f);
  }
}

NN_TEST(an_undecayed_parameter_moves_exactly_as_adam_would) {
  const float lr = 0.05f;

  NN_TEST_FOR_EACH_DEVICE(dev) {
    nn::Tensor bias   = nn::Tensor::from({0.5f, -0.25f}, dev);   // rank 1: no decay
    nn::Tensor matrix = nn::Tensor::from({{1.0f, 2.0f}}, dev);   // rank 2: decays
    bias.set_requires_grad(true);
    matrix.set_requires_grad(true);

    nn::optim::AdamW opt({&bias, &matrix}, lr, /*weight_decay=*/0.5f, kB1, kB2, kEps);

    Reference b0{0.5}, b1{-0.25};
    for (int i = 0; i < 3; ++i) {
      set_grad(bias, {0.3f, 0.3f});
      matrix.grad() = nn::Tensor::from({{0.3f, 0.3f}}, nn::Device::CPU).to(dev);
      opt.step();
      b0.step(0.3, lr);       // no wd
      b1.step(0.3, lr);
    }

    const std::vector<float> gb = host_of(bias);
    NN_CHECK_CLOSE(gb[0], float(b0.p), 2e-6f);
    NN_CHECK_CLOSE(gb[1], float(b1.p), 2e-6f);

    // the matrix, given the same gradient, must have gone somewhere else
    const std::vector<float> gm = host_of(matrix);
    NN_CHECK(std::fabs(gm[0] - 1.0f) > 0.0f);
    NN_CHECK(gm[0] < 1.0f);   // decayed towards zero on top of the step
  }
}

NN_TEST(set_lr_changes_the_next_step_and_keeps_the_moments) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    nn::Tensor p = with_grad({1.0f}, dev);
    nn::optim::AdamW opt({&p}, 0.01f, 0.0f, kB1, kB2, kEps);
    NN_CHECK_CLOSE(opt.lr(), 0.01f, 0.0f);

    Reference ref{1.0};
    set_grad(p, {0.5f});
    opt.step();
    ref.step(0.5, 0.01);

    opt.set_lr(0.2f);
    NN_CHECK_CLOSE(opt.lr(), 0.2f, 0.0f);
    set_grad(p, {0.5f});
    opt.step();
    ref.step(0.5, 0.2);      // same m and v, new lr

    NN_CHECK_CLOSE(host_of(p)[0], float(ref.p), 2e-6f);
  }
}

NN_TEST(adamw_trains_a_linear_layer) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    nn::Pcg32 rng(21);
    nn::Linear fc(4, 3, rng);
    fc.to(dev);

    const nn::Tensor x = nn::Tensor::from({{0.4f, -0.7f, 1.1f, 0.2f},
                                           {-0.3f, 0.9f, 0.1f, -1.2f}}, dev);
    const nn::Tensor labels = nn::Tensor::from_i32({0, 2}, dev);

    nn::optim::AdamW opt(fc.parameters(), 0.05f, 0.01f);

    float first = 0.0f, last = 0.0f;
    for (int i = 0; i < 40; ++i) {
      opt.zero_grad();
      nn::autograd::GradScope grad;
      nn::Tensor loss = nn::cross_entropy(fc(x), labels);
      last = loss.item();
      if (i == 0) first = last;
      loss.backward();
      nn::optim::clip_grad_norm(fc.parameters(), 1.0f);
      opt.step();
    }

    NN_CHECK(last < first * 0.1f);
  }
}


NN_TEST(schedule_warmup_ramps_and_hands_over_without_a_step) {
  const float peak = 0.4f;
  nn::optim::Schedule s(peak, /*total=*/10, /*warmup=*/4);

  // step 0 already trains, and the ramp is linear in the step number
  NN_CHECK_CLOSE(s.at(0), peak * 0.25f, 1e-7f);
  NN_CHECK_CLOSE(s.at(1), peak * 0.50f, 1e-7f);
  NN_CHECK_CLOSE(s.at(2), peak * 0.75f, 1e-7f);
  NN_CHECK_CLOSE(s.at(3), peak, 1e-7f);          // last warmup step is the peak
  NN_CHECK_CLOSE(s.at(4), peak, 1e-7f);          // decay starts there: no jump

  // no warmup at all is a valid schedule, and starts at the peak
  nn::optim::Schedule none(peak, 10);
  NN_CHECK_CLOSE(none.at(0), peak, 1e-7f);
}

NN_TEST(cosine_hits_its_endpoints_and_midpoint_exactly) {
  const float peak = 0.001f, floor = 0.0001f;
  nn::optim::Schedule s(peak, /*total=*/100, /*warmup=*/0,
                        nn::optim::Decay::Cosine, floor);

  NN_CHECK_CLOSE(s.at(0), peak, 0.0f);                        // cos(0) == 1
  NN_CHECK_CLOSE(s.at(50), floor + 0.5f * (peak - floor), 1e-9f);
  NN_CHECK_CLOSE(s.at(100), floor, 0.0f);                     // cos(pi) == -1

  // and it stays at the floor past the end rather than turning back up
  NN_CHECK_CLOSE(s.at(101), floor, 0.0f);
  NN_CHECK_CLOSE(s.at(100000), floor, 0.0f);

  // monotone all the way down, and never outside [floor, peak]
  float prev = s.at(0);
  for (int64_t i = 1; i <= 100; ++i) {
    const float now = s.at(i);
    NN_CHECK(now <= prev);
    NN_CHECK(now >= floor - 1e-9f && now <= peak + 1e-9f);
    prev = now;
  }
}

NN_TEST(linear_and_constant_decays) {
  const float peak = 1.0f, floor = 0.2f;

  nn::optim::Schedule lin(peak, 8, 0, nn::optim::Decay::Linear, floor);
  NN_CHECK_CLOSE(lin.at(0), 1.0f, 1e-7f);
  NN_CHECK_CLOSE(lin.at(2), 0.8f, 1e-6f);     // 1 - 2/8 of the way from 1 to 0.2
  NN_CHECK_CLOSE(lin.at(4), 0.6f, 1e-6f);
  NN_CHECK_CLOSE(lin.at(8), floor, 1e-7f);
  NN_CHECK_CLOSE(lin.at(20), floor, 1e-7f);

  // constant holds the peak after warmup; min_lr does not apply
  nn::optim::Schedule flat(peak, 8, 2, nn::optim::Decay::Constant, floor);
  NN_CHECK_CLOSE(flat.at(0), 0.5f, 1e-7f);    // still warms up
  NN_CHECK_CLOSE(flat.at(1), 1.0f, 1e-7f);
  NN_CHECK_CLOSE(flat.at(7), 1.0f, 1e-7f);
  NN_CHECK_CLOSE(flat.at(99), 1.0f, 1e-7f);
}

NN_TEST(schedule_rejects_impossible_configurations) {
  NN_CHECK_THROWS(nn::optim::Schedule(0.1f, 0), std::invalid_argument);
  NN_CHECK_THROWS(nn::optim::Schedule(0.1f, -5), std::invalid_argument);
  NN_CHECK_THROWS(nn::optim::Schedule(0.1f, 10, -1), std::invalid_argument);
  NN_CHECK_THROWS(nn::optim::Schedule(0.1f, 10, 11), std::invalid_argument);
  // a floor above the peak would make the curve run upwards
  NN_CHECK_THROWS(nn::optim::Schedule(0.1f, 10, 0, nn::optim::Decay::Cosine, 0.2f),
                  std::invalid_argument);
  NN_CHECK_THROWS(nn::optim::Schedule(0.1f, 10, 0, nn::optim::Decay::Cosine, -0.1f),
                  std::invalid_argument);
  NN_CHECK_THROWS(nn::optim::Schedule(0.1f, 10).at(-1), std::invalid_argument);

  // warmup covering the whole run is legal: it ramps and never decays
  nn::optim::Schedule all_warmup(0.1f, 4, 4);
  NN_CHECK_CLOSE(all_warmup.at(3), 0.1f, 1e-7f);
  NN_CHECK_CLOSE(all_warmup.at(4), 0.0f, 0.0f);   // past the end, at the floor
}

NN_TEST(a_schedule_drives_any_optimizer) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    nn::Tensor p = with_grad({1.0f}, dev);
    nn::optim::SGD sgd({&p}, 0.0f);
    nn::optim::Schedule sched(0.5f, /*total=*/4, /*warmup=*/2,
                              nn::optim::Decay::Linear);

    // SGD with a unit gradient moves by exactly -lr, so the parameter records
    // the schedule it was driven with.
    float want = 1.0f;
    for (int64_t s = 0; s < 4; ++s) {
      sgd.set_lr(sched.at(s));
      NN_CHECK_CLOSE(sgd.lr(), sched.at(s), 0.0f);
      set_grad(p, {1.0f});
      sgd.step();
      want -= sched.at(s);
    }
    // 0.25 + 0.5 + 0.5 + 0.25 = 1.5
    NN_CHECK_CLOSE(want, -0.5f, 1e-6f);
    NN_CHECK_CLOSE(host_of(p)[0], want, 1e-6f);
  }
}

NN_TEST(a_scheduled_run_trains) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    auto train = [&](bool scheduled) {
      nn::Pcg32 rng(21);
      nn::Linear fc(4, 3, rng);
      fc.to(dev);

      const nn::Tensor x = nn::Tensor::from({{0.4f, -0.7f, 1.1f, 0.2f},
                                             {-0.3f, 0.9f, 0.1f, -1.2f}}, dev);
      const nn::Tensor labels = nn::Tensor::from_i32({0, 2}, dev);

      const int64_t steps = 40;
      nn::optim::AdamW opt(fc.parameters(), 0.05f, 0.01f);
      nn::optim::Schedule sched(0.05f, steps, /*warmup=*/5);

      float loss_v = 0.0f;
      for (int64_t s = 0; s < steps; ++s) {
        if (scheduled) opt.set_lr(sched.at(s));
        // the last executed step is total-1, so the rate is near the floor but
        // not on it -- the floor is the limit, reached one step past the run
        if (scheduled && s + 1 == steps) NN_CHECK(opt.lr() < 0.01f * 0.05f);
        opt.zero_grad();
        nn::autograd::GradScope grad;
        nn::Tensor loss = nn::cross_entropy(fc(x), labels);
        loss_v = loss.item();
        loss.backward();
        opt.step();
      }
      return loss_v;
    };

    NN_CHECK(train(false) < 0.1f);
    NN_CHECK(train(true) < 0.1f);
  }
}
