#include "test_harness.h"
#include "devices.h"

#include <stdexcept>
#include <vector>

#include <nn/core/rng.h>
#include <nn/core/tensor.h>
#include <nn/autograd/functions.h>
#include <nn/autograd/tape.h>
#include <nn/ops/ops.h>

#include "gradcheck.h"

namespace {

std::vector<float> host_of(const nn::Tensor& t) {
  const nn::Tensor h = t.contiguous().to(nn::Device::CPU);
  return std::vector<float>(h.host_data(), h.host_data() + h.numel());
}

nn::Tensor ramp(nn::Shape s, nn::Device d) {
  std::vector<float> v(size_t(s.numel()));
  for (size_t i = 0; i < v.size(); ++i) v[i] = float(i) * 0.5f - 3.0f;
  return nn::Tensor::from(v, s, d);
}

}  // namespace

NN_TEST(broadcast_shapes_follows_numpy_rules) {
  using nn::Shape;
  NN_CHECK(nn::broadcast_shapes(Shape({3, 4}), Shape({3, 4})) == Shape({3, 4}));
  NN_CHECK(nn::broadcast_shapes(Shape({3, 4}), Shape({4}))    == Shape({3, 4}));
  NN_CHECK(nn::broadcast_shapes(Shape({4}), Shape({3, 4}))    == Shape({3, 4}));
  NN_CHECK(nn::broadcast_shapes(Shape({3, 1}), Shape({1, 4})) == Shape({3, 4}));
  NN_CHECK(nn::broadcast_shapes(Shape({2, 1, 4}), Shape({3, 1})) == Shape({2, 3, 4}));
  NN_CHECK(nn::broadcast_shapes(Shape({}), Shape({3, 4}))     == Shape({3, 4}));

  NN_CHECK_THROWS(nn::broadcast_shapes(Shape({3}), Shape({4})), std::invalid_argument);
  NN_CHECK_THROWS(nn::broadcast_shapes(Shape({2, 3}), Shape({2, 4})), std::invalid_argument);
}

// A broadcast is a stride-0 view, not a copy -- that is the whole reason it is
// free. numel() then exceeds the storage, which is legal only because every
// byte-walking path goes through contiguous() first.
NN_TEST(expand_is_a_zero_stride_view) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    const nn::Tensor b = ramp(nn::Shape({4}), dev);
    const nn::Tensor e = b.expand(nn::Shape({3, 4}));

    NN_CHECK(e.shape() == nn::Shape({3, 4}));
    NN_CHECK(e.stride(0) == 0 && e.stride(1) == 1);
    NN_CHECK(!e.is_contiguous());
    NN_CHECK(e.numel() == 12);              // over four floats of storage

    const std::vector<float> got = host_of(e);
    const std::vector<float> src = host_of(b);
    for (int r = 0; r < 3; ++r)
      for (int c = 0; c < 4; ++c) NN_CHECK_CLOSE(got[size_t(r) * 4 + c], src[c], 0.0f);

    NN_CHECK_THROWS(b.expand(nn::Shape({3, 5})), std::invalid_argument);
    NN_CHECK_THROWS(nn::Tensor::zeros({3, 4}, dev).expand(nn::Shape({4})),
                    std::invalid_argument);
  }
}

// Every case is checked against explicitly materialised operands, so a wrong
// stride shows up as wrong values rather than a plausible-looking result.
NN_TEST(add_broadcasts) {
  struct Case { nn::Shape a, b, out; };
  const Case cases[] = {
    {nn::Shape({3, 4}),    nn::Shape({4}),    nn::Shape({3, 4})},
    {nn::Shape({4}),       nn::Shape({3, 4}), nn::Shape({3, 4})},
    {nn::Shape({3, 1}),    nn::Shape({1, 4}), nn::Shape({3, 4})},
    {nn::Shape({2, 1, 4}), nn::Shape({3, 1}), nn::Shape({2, 3, 4})},
    {nn::Shape({2, 3}),    nn::Shape({}),     nn::Shape({2, 3})},
    {nn::Shape({5, 6}),    nn::Shape({5, 6}), nn::Shape({5, 6})},
  };

  NN_TEST_FOR_EACH_DEVICE(dev) {
    for (const Case& c : cases) {
      const nn::Tensor a = ramp(c.a, dev);
      const nn::Tensor b = ramp(c.b, dev);

      const nn::Tensor got = nn::ops::add(a, b);
      NN_CHECK(got.shape() == c.out);

      // reference: materialise both sides, then take the plain dense path
      const nn::Tensor want = nn::ops::add(a.expand(c.out).contiguous(),
                                           b.expand(c.out).contiguous());
      const std::vector<float> g = host_of(got), w = host_of(want);
      for (size_t i = 0; i < g.size(); ++i) NN_CHECK_CLOSE(g[i], w[i], 1e-6f);
    }
    NN_CHECK_THROWS(nn::ops::add(ramp(nn::Shape({3}), dev), ramp(nn::Shape({4}), dev)),
                    std::invalid_argument);
  }
}

NN_TEST(sum_to_reverses_a_broadcast) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    const nn::Tensor g = ramp(nn::Shape({2, 3, 4}), dev);
    const std::vector<float> h = host_of(g);

    {   // down to [4]: sum over the first two axes
      const std::vector<float> out = host_of(nn::ops::sum_to(g, nn::Shape({4})));
      for (int c = 0; c < 4; ++c) {
        float want = 0.0f;
        for (int i = 0; i < 2; ++i)
          for (int j = 0; j < 3; ++j) want += h[(size_t(i) * 3 + j) * 4 + c];
        NN_CHECK_CLOSE(out[size_t(c)], want, 1e-5f);
      }
    }
    {   // down to [2,1,4]: sum over the middle axis only
      const std::vector<float> out = host_of(nn::ops::sum_to(g, nn::Shape({2, 1, 4})));
      for (int i = 0; i < 2; ++i)
        for (int c = 0; c < 4; ++c) {
          float want = 0.0f;
          for (int j = 0; j < 3; ++j) want += h[(size_t(i) * 3 + j) * 4 + c];
          NN_CHECK_CLOSE(out[size_t(i) * 4 + c], want, 1e-5f);
        }
    }
    {   // all the way down
      NN_CHECK_CLOSE(nn::ops::sum_to(g, nn::Shape({})).item(),
                     nn::ops::sum_all(g).item(), 1e-4f);
    }
    // identity, and the direction it refuses
    NN_CHECK(nn::ops::sum_to(g, nn::Shape({2, 3, 4})).shape() == nn::Shape({2, 3, 4}));
    NN_CHECK_THROWS(nn::ops::sum_to(g, nn::Shape({2, 5, 4})), std::invalid_argument);
  }
}

// sum_to must read through the gradient's strides, not its storage.
NN_TEST(sum_to_handles_a_strided_gradient) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    const nn::Tensor wide = ramp(nn::Shape({4, 7}), dev);
    const nn::Tensor g = wide.slice(1, 1, 4);      // [4,4], row stride 7
    NN_CHECK(!g.is_contiguous());

    const std::vector<float> a = host_of(nn::ops::sum_to(g, nn::Shape({4})));
    const std::vector<float> b = host_of(nn::ops::sum_to(g.contiguous(), nn::Shape({4})));
    for (size_t i = 0; i < a.size(); ++i) NN_CHECK_CLOSE(a[i], b[i], 1e-6f);
  }
}

// The end-to-end claim: a broadcast add is differentiable, and the stretched
// operand collects the gradient from every output it fed.
NN_TEST(gradcheck_broadcast_add) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    nn::Pcg32 rng(11);
    nn::Tensor x = nn::Tensor::randn({6, 4}, rng, 0.5f, dev);
    nn::Tensor b = nn::Tensor::randn({4}, rng, 0.5f, dev);
    x.set_requires_grad(true);
    b.set_requires_grad(true);

    const nn::Tensor labels = nn::Tensor::from_i32({0, 3, 1, 2, 0, 3}, dev);

    nn::autograd::Tape tape;
    nn::Tensor loss;
    auto forward = [&]() -> float {
      tape.clear();
      nn::autograd::TapeScope scope(tape);
      loss = nn::autograd::cross_entropy(nn::autograd::add(x, b), labels);
      return loss.item();
    };
    auto backward = [&]() { tape.backward(loss, true); };

    NN_CHECK(nn::test::gradCheck(x, forward, backward) < 2e-2f);
    NN_CHECK(nn::test::gradCheck(b, forward, backward) < 2e-2f);
  }
}

// add_row_bias is exactly this special case, and col_sum is exactly its
// backward -- which is the argument for deleting both.
NN_TEST(broadcast_add_subsumes_add_row_bias) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    const nn::Tensor x = ramp(nn::Shape({5, 7}), dev);
    const nn::Tensor b = ramp(nn::Shape({7}), dev);

    const std::vector<float> viaBroadcast = host_of(nn::ops::add(x, b));
    const std::vector<float> viaRowBias   = host_of(nn::ops::add_row_bias(x, b));
    for (size_t i = 0; i < viaBroadcast.size(); ++i) {
      NN_CHECK_CLOSE(viaBroadcast[i], viaRowBias[i], 1e-6f);
    }

    const nn::Tensor g = ramp(nn::Shape({5, 7}), dev);
    const std::vector<float> viaSumTo  = host_of(nn::ops::sum_to(g, nn::Shape({7})));
    const std::vector<float> viaColSum = host_of(nn::ops::col_sum(g));
    for (size_t i = 0; i < viaSumTo.size(); ++i) {
      NN_CHECK_CLOSE(viaSumTo[i], viaColSum[i], 1e-5f);
    }
  }
}
