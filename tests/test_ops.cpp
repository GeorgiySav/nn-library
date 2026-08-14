#include "test_harness.h"

#include <nn/ops/ops.h>

NN_TEST(test_matmul) {
  nn::Tensor a = nn::Tensor::from({
    {1, 2, 3},
    {4, 5, 6}
  });
  nn::Tensor b = nn::Tensor::from({
    {7, 8},
    {9, 10},
    {11, 12}
  });

  nn::Tensor c = nn::ops::matmul(a, b);

  NN_CHECK(c.shape() == nn::Shape({2, 2}));
  NN_CHECK_CLOSE(c.data()[0], 58.0f, 1e-6);
  NN_CHECK_CLOSE(c.data()[1], 64.0f, 1e-6);
  NN_CHECK_CLOSE(c.data()[2], 139.0f, 1e-6);
  NN_CHECK_CLOSE(c.data()[3], 154.0f, 1e-6);


  nn::Tensor d = nn::Tensor::from({
    {7, 9, 11},
    {8, 10, 12}
  });

  nn::Tensor e = nn::ops::matmul(a, d, false, true);

  NN_CHECK(e.shape() == nn::Shape({2, 2}));
  NN_CHECK_CLOSE(e.data()[0], 58.0f, 1e-6);
  NN_CHECK_CLOSE(e.data()[1], 64.0f, 1e-6);
  NN_CHECK_CLOSE(e.data()[2], 139.0f, 1e-6);
  NN_CHECK_CLOSE(e.data()[3], 154.0f, 1e-6);
}

NN_TEST(test_softmax_ce) {
  nn::Pcg32 rng(1234);
  int M = 8, N = 10;

  nn::Tensor logits = nn::Tensor::randn(nn::Shape({M, N}), rng, 1.0f);
  nn::Tensor labels = nn::Tensor::zeros(nn::Shape({M}), nn::Device::CPU, nn::DType::I32);
  for (int i{0}; i < M; ++i) {
    labels.data_i32()[i] = static_cast<int32_t>(rng.next_uint32() % N);
  }

  nn::Tensor loss = nn::Tensor::zeros(nn::Shape({}), nn::Device::CPU, nn::DType::F32);
  nn::Tensor probs = nn::Tensor::zeros(nn::Shape({M, N}), nn::Device::CPU, nn::DType::F32);
  nn::ops::softmax_ce(logits, labels, loss, probs);

  NN_CHECK(loss.shape().rank() == 0);
  NN_CHECK(probs.shape() == nn::Shape({M, N}));
}