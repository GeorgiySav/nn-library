// Gradient clipping, and the sum_all accumulators it is built on.

#include "test_harness.h"
#include "devices.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include <nn/autograd/functions.h>
#include <nn/autograd/tape.h>
#include <nn/core/tensor.h>
#include <nn/module/linear.h>
#include <nn/optim/optim.h>

namespace {

std::vector<float> host_of(const nn::Tensor& t) {
  const nn::Tensor h = t.pack().to(nn::Device::CPU);
  return std::vector<float>(h.host_data(), h.host_data() + h.numel());
}

nn::Tensor param_with_grad(std::initializer_list<float> g, nn::Device d) {
  nn::Tensor p = nn::Tensor::zeros({int(g.size())}, d);
  p.set_requires_grad(true);
  p.grad() = nn::Tensor::from(g, nn::Device::CPU).to(d);
  return p;
}

float l2(const std::vector<float>& v) {
  double s = 0.0;
  for (float x : v) s += double(x) * x;
  return float(std::sqrt(s));
}

}  // namespace

NN_TEST(sum_all_accumulators) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    // deliberately signed and not symmetric, so Sum, SumSq and SumAbs all differ
    const nn::Tensor x = nn::Tensor::from({1.0f, -2.0f, 3.0f, -4.0f}, dev);

    NN_CHECK_CLOSE(nn::ops::sum_all(x).item(), -2.0f, 1e-6f);
    NN_CHECK_CLOSE(nn::ops::sum_all(x, nn::ops::Accum::SumSq).item(), 30.0f, 1e-5f);
    NN_CHECK_CLOSE(nn::ops::sum_all(x, nn::ops::Accum::SumAbs).item(), 10.0f, 1e-6f);
  }
}

NN_TEST(sum_all_accumulators_on_a_view) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    const std::vector<float> raw{1.0f, -2.0f, 99.0f, 3.0f, -4.0f, 99.0f};
    const nn::Tensor wide = nn::Tensor::from(raw, nn::Shape({2, 3}), dev);
    const nn::Tensor v = wide.slice_view(1, 0, 2);   // drops the 99s

    NN_CHECK(!v.is_contiguous());
    NN_CHECK_CLOSE(nn::ops::sum_all(v, nn::ops::Accum::SumSq).item(), 30.0f, 1e-5f);
    NN_CHECK_CLOSE(nn::ops::sum_all(v, nn::ops::Accum::SumAbs).item(), 10.0f, 1e-6f);
  }
}

NN_TEST(grad_norm_spans_all_parameters) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    nn::Tensor a = param_with_grad({3.0f, 4.0f}, dev);          // norm 5
    nn::Tensor b = param_with_grad({12.0f}, dev);               // norm 12
    std::vector<nn::Tensor*> params{&a, &b};

    NN_CHECK_CLOSE(nn::optim::grad_norm(params), 13.0f, 1e-5f); // not 5, not 12, not 17

    // a parameter backward never reached is skipped, not counted as zero-length
    nn::Tensor c = nn::Tensor::zeros({4}, dev);
    params.push_back(&c);
    NN_CHECK_CLOSE(nn::optim::grad_norm(params), 13.0f, 1e-5f);

    NN_CHECK_CLOSE(nn::optim::grad_norm(std::span<nn::Tensor* const>{}), 0.0f, 0.0f);
  }
}

NN_TEST(clip_grad_norm_scales_only_when_over_the_limit) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    {
      nn::Tensor a = param_with_grad({3.0f, 4.0f}, dev);
      std::vector<nn::Tensor*> params{&a};

      NN_CHECK_CLOSE(nn::optim::clip_grad_norm(params, 10.0f), 5.0f, 1e-5f);
      const std::vector<float> g = host_of(a.grad());
      NN_CHECK_CLOSE(g[0], 3.0f, 0.0f);
      NN_CHECK_CLOSE(g[1], 4.0f, 0.0f);
    }

    {
      nn::Tensor a = param_with_grad({3.0f, 4.0f}, dev);   // contributes 5
      nn::Tensor b = param_with_grad({12.0f}, dev);        // contributes 12
      std::vector<nn::Tensor*> params{&a, &b};

      NN_CHECK_CLOSE(nn::optim::clip_grad_norm(params, 1.0f), 13.0f, 1e-5f);

      const std::vector<float> ga = host_of(a.grad()), gb = host_of(b.grad());
      const std::vector<float> all{ga[0], ga[1], gb[0]};
      NN_CHECK_CLOSE(l2(all), 1.0f, 1e-5f);          // the whole point

      // every component shrank by the same factor: ratios are unchanged
      NN_CHECK_CLOSE(ga[1] / ga[0], 4.0f / 3.0f, 1e-5f);
      NN_CHECK_CLOSE(gb[0] / ga[0], 4.0f, 1e-5f);
    }
  }
}

NN_TEST(clip_grad_norm_at_the_boundary_is_a_no_op) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    nn::Tensor a = param_with_grad({3.0f, 4.0f}, dev);
    std::vector<nn::Tensor*> params{&a};

    NN_CHECK_CLOSE(nn::optim::clip_grad_norm(params, 5.0f), 5.0f, 1e-5f);
    NN_CHECK_CLOSE(l2(host_of(a.grad())), 5.0f, 1e-5f);

    // a zero gradient is under every limit and must not divide by zero
    nn::Tensor z = param_with_grad({0.0f, 0.0f}, dev);
    std::vector<nn::Tensor*> zp{&z};
    NN_CHECK_CLOSE(nn::optim::clip_grad_norm(zp, 1.0f), 0.0f, 0.0f);
    NN_CHECK_CLOSE(host_of(z.grad())[0], 0.0f, 0.0f);
  }
}

NN_TEST(clip_grad_norm_reports_a_non_finite_norm_without_touching_the_gradients) {
  const float inf = std::numeric_limits<float>::infinity();
  const float nan = std::numeric_limits<float>::quiet_NaN();

  NN_TEST_FOR_EACH_DEVICE(dev) {
    for (float bad : {inf, nan}) {
      nn::Tensor a = param_with_grad({bad, 1.0f}, dev);
      nn::Tensor b = param_with_grad({2.0f}, dev);
      std::vector<nn::Tensor*> params{&a, &b};

      const float n = nn::optim::clip_grad_norm(params, 1.0f);
      NN_CHECK(!std::isfinite(n));

      // the finite parameter is untouched, so the caller can skip the step
      NN_CHECK_CLOSE(host_of(b.grad())[0], 2.0f, 0.0f);
      NN_CHECK_CLOSE(host_of(a.grad())[1], 1.0f, 0.0f);
    }

    nn::Tensor a = param_with_grad({1.0f}, dev);
    std::vector<nn::Tensor*> params{&a};
    NN_CHECK_THROWS(nn::optim::clip_grad_norm(params, 0.0f), std::invalid_argument);
    NN_CHECK_THROWS(nn::optim::clip_grad_norm(params, -1.0f), std::invalid_argument);
  }
}

NN_TEST(clip_grad_value_clamps_each_element) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    nn::Tensor a = param_with_grad({-3.0f, -0.5f, 0.25f, 7.0f}, dev);
    std::vector<nn::Tensor*> params{&a};

    nn::optim::clip_grad_value(params, 1.0f);

    const std::vector<float> g = host_of(a.grad());
    NN_CHECK_CLOSE(g[0], -1.0f, 0.0f);
    NN_CHECK_CLOSE(g[1], -0.5f, 0.0f);   // inside the band, unchanged
    NN_CHECK_CLOSE(g[2], 0.25f, 0.0f);
    NN_CHECK_CLOSE(g[3], 1.0f, 0.0f);

    NN_CHECK_THROWS(nn::optim::clip_grad_value(params, 0.0f), std::invalid_argument);
  }
}

NN_TEST(clipping_shortens_a_step_without_turning_it) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    auto run = [&](bool clip) {
      nn::Pcg32 rng(5);
      nn::Linear fc(4, 3, rng);
      fc.to(dev);

      const nn::Tensor x = nn::Tensor::full({2, 4}, 4.0f, dev);
      const nn::Tensor labels = nn::Tensor::from_i32({0, 2}, dev);

      const std::vector<nn::Tensor*> params = fc.parameters();
      const std::vector<float> before = host_of(*params[0]);

      {
        nn::autograd::GradScope grad;
        nn::Tensor loss = nn::cross_entropy(fc(x), labels);
        loss.backward();
      }

      const float norm = nn::optim::grad_norm(params);
      if (clip) nn::optim::clip_grad_norm(params, 0.1f);

      nn::optim::SGD opt(params, 1.0f);
      opt.step();

      std::vector<float> delta = host_of(*params[0]);
      for (size_t i = 0; i < delta.size(); ++i) delta[i] -= before[i];
      return std::pair{norm, delta};
    };

    const auto [norm, plain]   = run(false);
    const auto [n2,  clipped] = run(true);

    NN_CHECK_CLOSE(norm, n2, 1e-6f);     // clipping does not change the norm reported
    NN_CHECK(norm > 0.1f);               // the case is actually exercised

    const float want = 0.1f / (norm + 1e-6f);
    NN_CHECK(l2(plain) > l2(clipped));
    for (size_t i = 0; i < plain.size(); ++i) {
      NN_CHECK_CLOSE(clipped[i], plain[i] * want, 1e-5f);
    }
  }
}
