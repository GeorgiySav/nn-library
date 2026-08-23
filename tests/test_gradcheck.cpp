#include "test_harness.h"
#include "devices.h"

#include <nn/autograd/tape.h>
#include <nn/autograd/functions.h>

#include <array>
#include <stdexcept>
#include <utility>
#include <vector>

#include "gradcheck.h"

NN_TEST(gradcheck_two_layer_mlp) {
  const int B = 4, D = 3, H = 5, C = 4;

  NN_TEST_FOR_EACH_DEVICE(dev) {
    nn::Pcg32 rng(1234);

    nn::Tensor x  = nn::Tensor::randn({B, D}, rng, 1.0f, dev);
    nn::Tensor w1 = nn::Tensor::randn({D, H}, rng, 0.5f, dev);
    nn::Tensor b1 = nn::Tensor::zeros({H}, dev);
    nn::Tensor w2 = nn::Tensor::randn({H, C}, rng, 0.5f, dev);
    nn::Tensor b2 = nn::Tensor::zeros({C}, dev);

    for (nn::Tensor* p : {&w1, &b1, &w2, &b2})
      p->set_requires_grad(true);

    std::vector<int32_t> labels_host(B);
    for (int i{0}; i < B; ++i) labels_host[i] = i % C;
    nn::Tensor labels = nn::Tensor::from_i32(labels_host, nn::Shape({B}), dev);

    nn::autograd::Tape tape;
    nn::Tensor loss;

    auto forward = [&]() -> float {
      tape.clear(); // new epoch
      nn::autograd::TapeScope scope(tape);

      nn::Tensor h1 = nn::autograd::relu(
        nn::autograd::add(nn::autograd::matmul(x, w1), b1)
      );

      nn::Tensor logits = nn::autograd::relu(
        nn::autograd::add(nn::autograd::matmul(h1, w2), b2)
      );

      loss = nn::autograd::cross_entropy(logits, labels);
      return loss.item();
    };

    auto backward = [&]() {
      tape.backward(loss, true);
    };

    NN_CHECK(nn::test::gradCheck(w1, forward, backward) < 2e-2f);
    NN_CHECK(nn::test::gradCheck(b1, forward, backward) < 2e-2f);
    NN_CHECK(nn::test::gradCheck(w2, forward, backward) < 2e-2f);
    NN_CHECK(nn::test::gradCheck(b2, forward, backward) < 2e-2f);
  }
}

NN_TEST(gradcheck_permute) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    nn::Pcg32 rng(7);
    nn::Tensor x = nn::Tensor::randn({2, 3, 4}, rng, 0.5f, dev);
    x.set_requires_grad(true);

    const nn::Tensor labels = nn::Tensor::from_i32({0, 3, 1, 2, 0, 3}, dev);

    nn::autograd::Tape tape;
    nn::Tensor loss;
    const int order[3] = {1, 2, 0};      // a 3-cycle, not self-inverse
    const std::array<int, 2> rows{6, 4};

    auto forward = [&]() -> float {
      tape.clear();
      nn::autograd::TapeScope scope(tape);
      loss = nn::autograd::cross_entropy(
        x.permute(order).reshape(nn::Shape(rows)), labels);
      return loss.item();
    };
    auto backward = [&]() { tape.backward(loss); };

    NN_CHECK(nn::test::gradCheck(x, forward, backward) < 2e-2f);
  }
}

NN_TEST(gradcheck_reshape) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    nn::Pcg32 rng(7);
    nn::Tensor x = nn::Tensor::randn({2, 3, 4}, rng, 0.5f, dev);
    x.set_requires_grad(true);

    const nn::Tensor labels = nn::Tensor::from_i32({0, 3, 1, 2, 0, 3}, dev);

    nn::autograd::Tape tape;
    nn::Tensor loss;
    const std::array<int, 2> rows{6, 4};

    auto forward = [&]() -> float {
      tape.clear();
      nn::autograd::TapeScope scope(tape);
      loss = nn::autograd::cross_entropy(x.reshape(nn::Shape(rows)), labels);
      return loss.item();
    };
    auto backward = [&]() { tape.backward(loss); };

    NN_CHECK(nn::test::gradCheck(x, forward, backward) < 2e-2f);
  }
}

NN_TEST(gradcheck_slice) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    nn::Pcg32 rng(7);
    nn::Tensor x = nn::Tensor::randn({2, 3, 4}, rng, 0.5f, dev);
    x.set_requires_grad(true);

    const nn::Tensor labels = nn::Tensor::from_i32({0, 3, 1, 2}, dev);

    nn::autograd::Tape tape;
    nn::Tensor loss;
    const std::array<int, 2> rows{4, 4};   // slice is [2,2,4] = 16 elements

    auto forward = [&]() -> float {
      tape.clear();
      nn::autograd::TapeScope scope(tape);
      loss = nn::autograd::cross_entropy(
        x.slice(1, 1, 2).reshape(nn::Shape(rows)), labels);
      return loss.item();
    };
    auto backward = [&]() { tape.backward(loss); };

    NN_CHECK(nn::test::gradCheck(x, forward, backward) < 2e-2f);
  }
}

NN_TEST(slice_backward_is_zero_outside_the_window) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    nn::Pcg32 rng(7);
    nn::Tensor x = nn::Tensor::randn({2, 3, 4}, rng, 0.5f, dev);
    x.set_requires_grad(true);

    nn::autograd::Tape tape;
    nn::autograd::TapeScope scope(tape);
    nn::Tensor loss = nn::autograd::sum_all(nn::autograd::slice(x, 1, 1, 2));
    tape.backward(loss);

    const nn::Tensor g = x.grad().to(nn::Device::CPU);
    NN_CHECK(g.shape() == nn::Shape({2, 3, 4}));
    for (int b = 0; b < 2; ++b)
      for (int t = 0; t < 3; ++t)
        for (int c = 0; c < 4; ++c) {
          const float want = (t >= 1 && t < 3) ? 1.0f : 0.0f;
          NN_CHECK_CLOSE(g.host_data()[(size_t(b) * 3 + t) * 4 + c], want, 0.0f);
        }
  }
}

NN_TEST(a_raw_view_cuts_the_branch_a_differentiable_one_keeps_it) {
  enum Spelling { kSafe, kRawView, kRawPack };

  NN_TEST_FOR_EACH_DEVICE(dev) {
    auto residual_grad = [&](Spelling how) {
      nn::Tensor x = nn::Tensor::full({2, 3}, 1.5f, dev);
      x.set_requires_grad(true);

      nn::autograd::Tape tape;
      nn::autograd::TapeScope scope(tape);

      nn::Tensor side;
      switch (how) {
        case kSafe:    side = x.transpose(0, 1).transpose(0, 1); break;
        case kRawView: side = x.transpose_view(0, 1).transpose_view(0, 1); break;
        // only the pack is raw here -- both transposes record, so this is the
        // pack alone cutting the branch
        case kRawPack: side = x.transpose(0, 1).pack().transpose_view(0, 1); break;
      }

      nn::Tensor loss = nn::autograd::sum_all(nn::autograd::add(x, side));
      const float forward = loss.item();
      tape.backward(loss);
      return std::pair{forward, x.grad().to(nn::Device::CPU).host_data()[0]};
    };

    const auto [f_view, g_view] = residual_grad(kRawView);
    const auto [f_pack, g_pack] = residual_grad(kRawPack);
    const auto [f_ok,   g_ok  ] = residual_grad(kSafe);

    NN_CHECK_CLOSE(f_view, f_ok, 0.0f);   // identical forward
    NN_CHECK_CLOSE(f_pack, f_ok, 0.0f);
    NN_CHECK_CLOSE(g_ok,   2.0f, 0.0f);   // both branches counted
    NN_CHECK_CLOSE(g_view, 1.0f, 0.0f);   // the view branch was dropped
    NN_CHECK_CLOSE(g_pack, 1.0f, 0.0f);   // ... and so was the packed one
  }
}

// contiguous() is a copy, not a view, so its backward is the identity -- and
// when the input is already dense it is not even a copy, and must not become
// one or leave a node behind.
NN_TEST(contiguous_is_differentiable_and_free_when_already_dense) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    nn::Pcg32 rng(33);
    nn::Tensor x = nn::Tensor::randn({3, 4}, rng, 0.7f, dev);
    x.set_requires_grad(true);

    nn::autograd::Tape tape;
    nn::autograd::TapeScope scope(tape);

    const nn::Tensor same = x.contiguous();
    NN_CHECK(same.device_ptr() == x.device_ptr());   // no copy, no node
    NN_CHECK(tape.size() == 0);

    // a strided input does copy, and the gradient still lands on x
    const nn::Tensor dense = x.t().contiguous();
    NN_CHECK(dense.is_contiguous());
    nn::Tensor loss = nn::autograd::sum_all(nn::autograd::mul(dense, dense));
    tape.backward(loss);

    const nn::Tensor g = x.grad().to(nn::Device::CPU);
    const nn::Tensor h = x.to(nn::Device::CPU);
    for (int i = 0; i < 12; ++i) NN_CHECK_CLOSE(g.host_data()[i], 2.0f * h.host_data()[i], 1e-5f);
  }
}

NN_TEST(gradcheck_chained_view_methods) {
  const int B = 2, T = 3, H = 2, hd = 2;

  NN_TEST_FOR_EACH_DEVICE(dev) {
    nn::Pcg32 rng(31);
    nn::Tensor x = nn::Tensor::randn({B, T, H * hd}, rng, 0.7f, dev);
    x.set_requires_grad(true);

    nn::autograd::Tape tape;
    nn::Tensor loss;

    std::vector<int32_t> labels_host(size_t(B) * H);
    for (size_t i = 0; i < labels_host.size(); ++i) labels_host[i] = int32_t(i % T);
    const nn::Tensor labels =
        nn::Tensor::from_i32(labels_host, nn::Shape({B * H}), dev);

    auto forward = [&]() -> float {
      tape.clear();
      nn::autograd::TapeScope scope(tape);
      // [B,T,H*hd] -> [B,H,T,hd] -> drop the last head column -> [B*H, T]
      const nn::Tensor heads =
          x.reshape(nn::Shape({B, T, H, hd})).permute({0, 2, 1, 3});
      loss = nn::autograd::cross_entropy(
          heads.slice(-1, 0, 1).reshape(nn::Shape({B * H, T})), labels);
      return loss.item();
    };
    auto backward = [&]() { tape.backward(loss, true); };

    NN_CHECK(nn::test::gradCheck(x, forward, backward) < 2e-2f);
  }
}

NN_TEST(gradcheck_expand) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    nn::Pcg32 rng(32);
    nn::Tensor bias = nn::Tensor::randn({1, 4}, rng, 0.8f, dev);
    bias.set_requires_grad(true);

    nn::autograd::Tape tape;
    nn::Tensor loss;
    auto forward = [&]() -> float {
      tape.clear();
      nn::autograd::TapeScope scope(tape);
      const nn::Tensor wide = bias.expand(nn::Shape({3, 4}));
      loss = nn::autograd::sum_all(nn::autograd::mul(wide, wide));
      return loss.item();
    };
    auto backward = [&]() { tape.backward(loss, true); };

    NN_CHECK(nn::test::gradCheck(bias, forward, backward) < 2e-2f);

    // and the count is exact: d/db sum(3 copies of b) = 3
    nn::Tensor b2 = nn::Tensor::full({1, 4}, 2.0f, dev);
    b2.set_requires_grad(true);
    nn::autograd::Tape t2;
    nn::autograd::TapeScope scope(t2);
    nn::Tensor l2 = nn::autograd::sum_all(b2.expand(nn::Shape({3, 4})));
    t2.backward(l2);
    const nn::Tensor g = b2.grad().to(nn::Device::CPU);
    for (int i = 0; i < 4; ++i) NN_CHECK_CLOSE(g.host_data()[i], 3.0f, 0.0f);
  }
}

NN_TEST(view_axes_are_normalised_and_checked) {
  const nn::Tensor t = nn::Tensor::zeros({2, 3, 4}, nn::Device::CPU);

  NN_CHECK(t.transpose_view(-2, -1).shape() == nn::Shape({2, 4, 3}));
  NN_CHECK(t.transpose(-2, -1).shape() == nn::Shape({2, 4, 3}));
  NN_CHECK(t.slice_view(-1, 1, 2).shape() == nn::Shape({2, 3, 2}));
  NN_CHECK(t.slice(-3, 1, 1).shape() == nn::Shape({1, 3, 4}));
  NN_CHECK(t.permute_view({-1, 0, -2}).shape() == nn::Shape({4, 2, 3}));
  NN_CHECK(t.permute({-1, 0, -2}).shape() == nn::Shape({4, 2, 3}));

  NN_CHECK_THROWS(t.transpose(0, 3), std::invalid_argument);
  NN_CHECK_THROWS(t.transpose_view(-4, 0), std::invalid_argument);
  NN_CHECK_THROWS(t.slice(1, 2, 3), std::invalid_argument);      // past the end
  NN_CHECK_THROWS(t.slice_view(1, -1, 2), std::invalid_argument);
  NN_CHECK_THROWS(t.reshape(nn::Shape({5, 5})), std::invalid_argument);
  NN_CHECK_THROWS(t.permute({0, 1}), std::invalid_argument);       // wrong count
  NN_CHECK_THROWS(t.permute({0, 1, 1}), std::invalid_argument);    // repeated axis
  NN_CHECK_THROWS(t.permute({0, 1, -3}), std::invalid_argument);   // -3 is axis 0
  NN_CHECK_THROWS(t.permute_view({0, 1, 3}), std::invalid_argument);
  NN_CHECK_THROWS(t.permute_view({0, -4, 1}), std::invalid_argument);
}

NN_TEST(gradcheck_permute_with_negative_axes) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    nn::Pcg32 rng(34);
    nn::Tensor x = nn::Tensor::randn({2, 3, 4}, rng, 0.6f, dev);
    x.set_requires_grad(true);

    const nn::Tensor labels = nn::Tensor::from_i32({0, 3, 1, 2, 0, 3}, dev);

    nn::autograd::Tape tape;
    nn::Tensor loss;
    auto forward = [&]() -> float {
      tape.clear();
      nn::autograd::TapeScope scope(tape);
      // {-2, -1, 0} is the same 3-cycle as {1, 2, 0}: [2,3,4] -> [3,4,2]
      const nn::Tensor p = x.permute({-2, -1, 0});
      NN_CHECK(p.shape() == nn::Shape({3, 4, 2}));
      loss = nn::autograd::cross_entropy(p.reshape(nn::Shape({6, 4})), labels);
      return loss.item();
    };
    auto backward = [&]() { tape.backward(loss); };

    NN_CHECK(nn::test::gradCheck(x, forward, backward) < 2e-2f);
  }
}
