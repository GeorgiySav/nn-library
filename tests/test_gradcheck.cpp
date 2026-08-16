#include "test_harness.h"

#include <nn/autograd/tape.h>
#include <nn/autograd/functions.h>

#include "gradcheck.h"

NN_TEST(gradcheck_two_layer_mlp) {
  nn::Pcg32 rng(1234);
  const int B = 4, D = 3, H = 5, C = 4;

  nn::Tensor x  = nn::Tensor::randn({B, D}, rng, 1.0f);
  nn::Tensor w1 = nn::Tensor::randn({D, H}, rng, 0.5f);
  nn::Tensor b1 = nn::Tensor::zeros({H});
  nn::Tensor w2 = nn::Tensor::randn({H, C}, rng, 0.5f);
  nn::Tensor b2 = nn::Tensor::zeros({C});

  for (nn::Tensor* p : {&w1, &b1, &w2, &b2})
    p->set_requires_grad(true);
  
  nn::Tensor labels(nn::Shape{B}, nn::Device::CPU, nn::DType::I32);
  for (int i{0}; i < B; ++i)
    labels.data_i32()[i] = i % C;

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