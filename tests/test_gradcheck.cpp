#include "test_harness.h"
#include "devices.h"

#include <nn/autograd/tape.h>
#include <nn/autograd/functions.h>

#include <array>

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
        nn::autograd::reshape(nn::autograd::permute(x, order), rows), labels);
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
      loss = nn::autograd::cross_entropy(nn::autograd::reshape(x, rows), labels);
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
        nn::autograd::reshape(nn::autograd::slice(x, 1, 1, 2), rows), labels);
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