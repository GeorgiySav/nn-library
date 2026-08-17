#include "test_harness.h"
#include "devices.h"

#include <nn/autograd/tape.h>
#include <nn/autograd/functions.h>

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
        nn::autograd::add_row_bias(
          nn::autograd::matmul(x, w1),
          b1
        )
      );

      nn::Tensor logits = nn::autograd::relu(
        nn::autograd::add_row_bias(
          nn::autograd::matmul(h1, w2),
          b2
        )
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
