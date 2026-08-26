#include "test_harness.h"
#include "devices.h"
#include "gradcheck.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include <nn/autograd/functions.h>
#include <nn/autograd/tape.h>
#include <nn/core/rng.h>
#include <nn/core/tensor.h>
#include <kernels/random.h>
#include <nn/nn/module.h>
#include <nn/ops/ops.h>

namespace {

std::vector<float> host_of(const nn::Tensor& t) {
  const nn::Tensor h = t.pack().to(nn::Device::CPU);
  return std::vector<float>(h.host_data(), h.host_data() + h.numel());
}

nn::Tensor ones(int n, nn::Device d) { return nn::Tensor::full({n}, 1.0f, d); }

}  // namespace

NN_TEST(random_uniform_is_in_range_and_flat) {
  const int64_t n = 1 << 20;
  const int kBuckets = 16;
  int64_t hits[kBuckets] = {0};
  double sum = 0.0;

  for (int64_t i = 0; i < n; ++i) {
    const float u = nn::kernels::random_uniform(12345, uint64_t(i));
    NN_CHECK(u >= 0.0f && u < 1.0f);
    hits[int(u * kBuckets)]++;
    sum += u;
  }

  NN_CHECK_CLOSE(float(sum / double(n)), 0.5f, 2e-3f);

  // 5 sigma on a binomial(n, 1/16) is about 1210 out of 65536
  const double expect = double(n) / kBuckets;
  const double sigma = std::sqrt(expect * (1.0 - 1.0 / kBuckets));
  for (int b = 0; b < kBuckets; ++b) {
    NN_CHECK(std::fabs(double(hits[b]) - expect) < 5.0 * sigma);
  }
}

NN_TEST(adjacent_counters_are_uncorrelated) {
  const int64_t n = 1 << 20;
  double sx = 0, sy = 0, sxy = 0, sxx = 0, syy = 0;

  for (int64_t i = 0; i < n; ++i) {
    const double x = nn::kernels::random_uniform(999, uint64_t(i));
    const double y = nn::kernels::random_uniform(999, uint64_t(i + 1));
    sx += x; sy += y; sxy += x * y; sxx += x * x; syy += y * y;
  }

  const double num = double(n) * sxy - sx * sy;
  const double den = std::sqrt((double(n) * sxx - sx * sx) * (double(n) * syy - sy * sy));
  const double r = num / den;
  // the standard error on r at this n is 1e-3, so 5 sigma is 5e-3
  NN_CHECK(std::fabs(r) < 5e-3);
}

NN_TEST(different_seeds_give_different_streams) {
  int differing = 0;
  for (uint64_t i = 0; i < 1000; ++i) {
    if (nn::kernels::random_bits(1, i) != nn::kernels::random_bits(2, i)) ++differing;
  }
  NN_CHECK(differing == 1000);

  // and no counter value repeats within one stream over a short run
  std::vector<uint32_t> seen;
  for (uint64_t i = 0; i < 4096; ++i) seen.push_back(nn::kernels::random_bits(7, i));
  std::sort(seen.begin(), seen.end());
  NN_CHECK(std::adjacent_find(seen.begin(), seen.end()) == seen.end());
}

// The mask is integer arithmetic, so both backends must produce the same one.
NN_TEST(the_same_seed_and_offset_give_the_same_mask_on_both_devices) {
  if (!nn::test::have_cuda()) return;

  const int n = 4096;
  const nn::Tensor cpu = nn::ops::dropout(ones(n, nn::Device::CPU), 0.3f, 42, 1000);
  const nn::Tensor gpu = nn::ops::dropout(ones(n, nn::Device::CUDA), 0.3f, 42, 1000);

  const std::vector<float> a = host_of(cpu), b = host_of(gpu);
  for (int i = 0; i < n; ++i) NN_CHECK_CLOSE(a[i], b[i], 0.0f);
}

NN_TEST(dropout_drops_at_the_requested_rate_and_scales_the_rest) {
  const int n = 1 << 16;

  NN_TEST_FOR_EACH_DEVICE(dev) {
    for (float p : {0.1f, 0.5f, 0.9f}) {
      const nn::Tensor y = nn::ops::dropout(ones(n, dev), p, 7, 0);
      const std::vector<float> v = host_of(y);

      const float scale = 1.0f / (1.0f - p);
      int64_t dropped = 0;
      double sum = 0.0;
      for (float x : v) {
        if (x == 0.0f) ++dropped;
        else NN_CHECK_CLOSE(x, scale, 1e-6f);   // survivors are scaled, exactly
        sum += x;
      }

      // 5 sigma on binomial(65536, p)
      const double expect = double(n) * p;
      const double sigma = std::sqrt(expect * (1.0 - p));
      NN_CHECK(std::fabs(double(dropped) - expect) < 5.0 * sigma);

      // the point of inverted dropout: the mean survives
      NN_CHECK_CLOSE(float(sum / double(n)), 1.0f, 0.05f);
    }
  }
}

NN_TEST(dropout_endpoints) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    // p == 0 keeps everything, unscaled
    for (float x : host_of(nn::ops::dropout(ones(64, dev), 0.0f, 3, 0))) {
      NN_CHECK_CLOSE(x, 1.0f, 0.0f);
    }
    // p == 1 drops everything, and 1/(1-p) never becomes an inf in the output
    for (float x : host_of(nn::ops::dropout(ones(64, dev), 1.0f, 3, 0))) {
      NN_CHECK_CLOSE(x, 0.0f, 0.0f);
    }
    NN_CHECK_THROWS(nn::ops::dropout(ones(4, dev), 1.5f, 0, 0), std::invalid_argument);
    NN_CHECK_THROWS(nn::ops::dropout(ones(4, dev), -0.1f, 0, 0), std::invalid_argument);
    NN_CHECK_THROWS(nn::Dropout(2.0f), std::invalid_argument);
  }
}

NN_TEST(the_backward_mask_is_the_forward_mask) {
  const int n = 1024;

  NN_TEST_FOR_EACH_DEVICE(dev) {
    nn::Tensor x = nn::Tensor::full({n}, 2.0f, dev);
    x.set_requires_grad(true);

    nn::autograd::Tape tape;
    nn::autograd::TapeScope scope(tape);

    const nn::Tensor y = nn::autograd::dropout(x, 0.4f);
    nn::Tensor loss = nn::autograd::sum_all(y);
    tape.backward(loss);

    const std::vector<float> fwd = host_of(y), g = host_of(x.grad());
    const float scale = 1.0f / 0.6f;

    int kept = 0;
    for (int i = 0; i < n; ++i) {
      if (fwd[i] == 0.0f) {
        NN_CHECK_CLOSE(g[i], 0.0f, 0.0f);       // dropped: no gradient at all
      } else {
        NN_CHECK_CLOSE(fwd[i], 2.0f * scale, 1e-6f);
        NN_CHECK_CLOSE(g[i], scale, 1e-6f);     // kept: scaled, same as forward
        ++kept;
      }
    }
    NN_CHECK(kept > 0 && kept < n);             // the case is actually exercised
  }
}

NN_TEST(gradcheck_dropout) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    nn::Pcg32 rng(5);
    nn::Tensor x = nn::Tensor::randn({6, 4}, rng, 0.8f, dev);
    x.set_requires_grad(true);

    const nn::Tensor labels = nn::Tensor::from_i32({0, 3, 1, 2, 0, 3}, dev);

    nn::autograd::Tape tape;
    nn::Tensor loss;
    auto forward = [&]() -> float {
      nn::manual_seed(99);            // same seed, same offsets, same mask
      tape.clear();
      nn::autograd::TapeScope scope(tape);
      loss = nn::cross_entropy(nn::dropout(x, 0.3f), labels);
      return loss.item();
    };
    auto backward = [&]() { tape.backward(loss, true); };

    NN_CHECK(nn::test::gradCheck(x, forward, backward) < 2e-2f);
  }
}

NN_TEST(successive_calls_draw_different_masks_and_a_reseed_repeats_them) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    auto draw = [&] { return host_of(nn::dropout(ones(256, dev), 0.5f)); };

    nn::manual_seed(1234);
    const std::vector<float> a = draw();
    const std::vector<float> b = draw();

    int same = 0;
    for (size_t i = 0; i < a.size(); ++i) same += (a[i] == b[i]);
    NN_CHECK(same < int(a.size()));      // the offset advanced

    nn::manual_seed(1234);
    const std::vector<float> a2 = draw();
    const std::vector<float> b2 = draw();
    for (size_t i = 0; i < a.size(); ++i) {
      NN_CHECK_CLOSE(a2[i], a[i], 0.0f);
      NN_CHECK_CLOSE(b2[i], b[i], 0.0f);
    }
  }
}

NN_TEST(eval_mode_is_the_identity_and_records_nothing) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    nn::Tensor x = nn::Tensor::full({32}, 3.0f, dev);
    x.set_requires_grad(true);

    nn::Dropout drop(0.5f);
    NN_CHECK(drop.training());

    nn::autograd::Tape tape;
    nn::autograd::TapeScope scope(tape);

    drop.eval();
    NN_CHECK(!drop.training());
    const nn::Tensor same = drop(x);
    NN_CHECK(same.device_ptr() == x.device_ptr());   // not even a copy
    NN_CHECK(tape.size() == 0);                      // and no node

    drop.train();
    const nn::Tensor dropped = drop(x);
    NN_CHECK(dropped.device_ptr() != x.device_ptr());
    NN_CHECK(tape.size() > 0);

    // p == 0 is the identity in training mode too
    nn::Dropout none(0.0f);
    NN_CHECK(none(x).device_ptr() == x.device_ptr());
  }
}

NN_TEST(sequential_propagates_train_and_eval_to_its_children) {
  nn::Pcg32 rng(3);
  nn::Sequential model(nn::Linear(4, 8, rng), nn::ReLu(),
                       nn::Dropout(0.5f), nn::Linear(8, 2, rng));

  const nn::Tensor x = nn::Tensor::full({2, 4}, 1.0f);

  // in eval the model is deterministic; in training it is not
  model.eval();
  NN_CHECK(!model.training());
  const std::vector<float> a = host_of(model(x)), b = host_of(model(x));
  for (size_t i = 0; i < a.size(); ++i) NN_CHECK_CLOSE(a[i], b[i], 0.0f);

  model.train();
  NN_CHECK(model.training());
  int differing = 0;
  for (int trial = 0; trial < 8; ++trial) {
    const std::vector<float> c = host_of(model(x)), d = host_of(model(x));
    for (size_t i = 0; i < c.size(); ++i) differing += (c[i] != d[i]);
  }
  NN_CHECK(differing > 0);   // the Dropout in the middle is live again
}
