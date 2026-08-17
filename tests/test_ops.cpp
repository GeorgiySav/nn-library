#include "test_harness.h"
#include "devices.h"

#include <nn/ops/ops.h>

NN_TEST(test_matmul) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    nn::Tensor a = nn::Tensor::from({
      {1, 2, 3},
      {4, 5, 6}
    }, dev);
    nn::Tensor b = nn::Tensor::from({
      {7, 8},
      {9, 10},
      {11, 12}
    }, dev);

    nn::Tensor c = nn::ops::matmul(a, b);

    NN_CHECK(c.shape() == nn::Shape({2, 2}));
    NN_CHECK(c.device() == dev);

    const nn::Tensor hc = c.to(nn::Device::CPU);
    NN_CHECK_CLOSE(hc.host_data()[0], 58.0f, 1e-6);
    NN_CHECK_CLOSE(hc.host_data()[1], 64.0f, 1e-6);
    NN_CHECK_CLOSE(hc.host_data()[2], 139.0f, 1e-6);
    NN_CHECK_CLOSE(hc.host_data()[3], 154.0f, 1e-6);

    nn::Tensor d = nn::Tensor::from({
      {7, 9, 11},
      {8, 10, 12}
    }, dev);

    nn::Tensor e = nn::ops::matmul(a, d, false, true);

    NN_CHECK(e.shape() == nn::Shape({2, 2}));

    const nn::Tensor he = e.to(nn::Device::CPU);
    NN_CHECK_CLOSE(he.host_data()[0], 58.0f, 1e-6);
    NN_CHECK_CLOSE(he.host_data()[1], 64.0f, 1e-6);
    NN_CHECK_CLOSE(he.host_data()[2], 139.0f, 1e-6);
    NN_CHECK_CLOSE(he.host_data()[3], 154.0f, 1e-6);
  }
}

NN_TEST(test_matmul_rejects_device_mismatch) {
  // only meaningful once there is more than one device to mismatch
  if (nn::test::devices().size() < 2) return;

  nn::Tensor a = nn::Tensor::from({{1, 2}, {3, 4}}, nn::test::devices()[0]);
  nn::Tensor b = nn::Tensor::from({{1, 2}, {3, 4}}, nn::test::devices()[1]);
  NN_CHECK_THROWS(nn::ops::matmul(a, b), std::runtime_error);
}

NN_TEST(test_softmax_ce) {
  const int M = 8, N = 10;

  NN_TEST_FOR_EACH_DEVICE(dev) {
    nn::Pcg32 rng(1234);

    nn::Tensor logits = nn::Tensor::randn(nn::Shape({M, N}), rng, 1.0f, dev);

    std::vector<int32_t> labels_host(M);
    for (int i{0}; i < M; ++i) {
      labels_host[i] = static_cast<int32_t>(rng.next_uint32() % N);
    }
    nn::Tensor labels = nn::Tensor::from_i32(labels_host, nn::Shape({M}), dev);

    nn::Tensor loss  = nn::Tensor::zeros(nn::Shape({}), dev, nn::DType::F32);
    nn::Tensor probs = nn::Tensor::zeros(nn::Shape({M, N}), dev, nn::DType::F32);
    nn::ops::softmax_ce(logits, labels, loss, probs);

    NN_CHECK(loss.shape().rank() == 0);
    NN_CHECK(probs.shape() == nn::Shape({M, N}));
    NN_CHECK(probs.device() == dev);
  }
}
