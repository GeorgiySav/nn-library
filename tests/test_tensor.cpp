#include "test_harness.h"

#include <nn/core/tensor.h>

NN_TEST(test_tensor_zeros) {
  nn::Tensor t = nn::Tensor::zeros({2, 3});
  NN_CHECK(t.numel() == 6);
  NN_CHECK(t.defined());
  NN_CHECK(t.shape() == nn::Shape({2, 3}));
  NN_CHECK(t.device() == nn::Device::CPU);
  NN_CHECK(t.dtype() == nn::DType::F32);
  for (size_t i{0u}; i < t.numel(); ++i) {
    NN_CHECK(t.host_data()[i] == 0.0f);
  }
}

NN_TEST(test_tensor_full) {
  nn::Tensor t = nn::Tensor::full({2, 3}, 42.0f);
  NN_CHECK(t.numel() == 6);
  NN_CHECK(t.defined());
  NN_CHECK(t.shape() == nn::Shape({2, 3}));
  NN_CHECK(t.device() == nn::Device::CPU);
  NN_CHECK(t.dtype() == nn::DType::F32);
  for (size_t i{0u}; i < t.numel(); ++i) {
    NN_CHECK(t.host_data()[i] == 42.0f);
  }
}

NN_TEST(test_tensor_scalar) {
  nn::Tensor t = nn::Tensor::scalar(3.14f);
  NN_CHECK(t.numel() == 1);
  NN_CHECK(t.shape().rank() == 0);
  NN_CHECK(t.defined());
  NN_CHECK(t.shape() == nn::Shape({}));
  NN_CHECK(t.device() == nn::Device::CPU);
  NN_CHECK(t.dtype() == nn::DType::F32);
  NN_CHECK(t.item() == 3.14f);
}

NN_TEST(test_tensor_copy) {
  // shallow copy
  nn::Tensor t1 = nn::Tensor::full({2, 3}, 1.0f);
  nn::Tensor t2 = t1;
  NN_CHECK(t1.host_data() == t2.host_data());

  // deep copy
  nn::Tensor t3 = t1.clone();
  NN_CHECK(t1.host_data() != t3.host_data());
}

NN_TEST(test_tensor_cpu_write_read) {
  nn::Tensor t = nn::Tensor::zeros({2, 3});
  float* data = t.host_data();
  for (size_t i{0u}; i < t.numel(); ++i) {
    data[i] = static_cast<float>(i);
  }
  for (size_t i{0u}; i < t.numel(); ++i) {
    NN_CHECK(t.host_data()[i] == static_cast<float>(i));
  }
}

NN_TEST(test_tensor_randn) {
  nn::Pcg32 rng(1234);
  nn::Tensor t = nn::Tensor::randn({10000}, rng, 0.5f);
  // get stddev
  float mean = 0.0f;
  for (size_t i{0u}; i < t.numel(); ++i) {
    mean += t.host_data()[i];
  }
  mean /= t.numel();
  float var = 0.0f;
  for (size_t i{0u}; i < t.numel(); ++i) {
    float diff = t.host_data()[i] - mean;
    var += diff * diff;
  }
  var /= t.numel();
  float stddev = std::sqrt(var);
  NN_CHECK(std::abs(mean) < 0.05f);
  NN_CHECK(std::abs(stddev - 0.5f) < 0.05f);
}

NN_TEST(test_tensor_requires_grad) {
  nn::Tensor t = nn::Tensor::zeros({2, 3});
  NN_CHECK(!t.requires_grad());
  t.set_requires_grad(true);
  NN_CHECK(t.requires_grad());
  NN_CHECK(t.grad().shape() == t.shape());
  t.zero_grad();
  for (size_t i{0u}; i < t.grad().numel(); ++i) {
    NN_CHECK(t.grad().host_data()[i] == 0.0f);
  }
}