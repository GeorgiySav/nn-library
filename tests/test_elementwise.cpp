#include "test_harness.h"
#include "devices.h"

#include <nn/core/tensor.h>
#include <nn/kernels/kernel_api.h>

const nn::kernels::KernelTable& init(nn::Device d) {
  nn::kernels::init_kernels();
  const nn::kernels::KernelTable& kernels = nn::kernels::kernels(d);
  NN_CHECK(kernels.gemm != nullptr);
  NN_CHECK(kernels.unary != nullptr);
  NN_CHECK(kernels.unary_backward != nullptr);
  NN_CHECK(kernels.binary != nullptr);
  NN_CHECK(kernels.scalar != nullptr);
  NN_CHECK(kernels.axpy != nullptr);
  NN_CHECK(kernels.fill != nullptr);
  return kernels;
}

NN_TEST(test_fill) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    const auto& k = init(dev);

    nn::Tensor X = nn::Tensor::full({10}, -1.0f, dev);
    k.fill(nn::current_stream(dev), 42.0f, X.device_ptr(), X.numel());

    const nn::Tensor h = X.to(nn::Device::CPU);
    for (int64_t i{0}; i < h.numel(); ++i) {
      NN_CHECK(h.host_data()[i] == 42.0f);
    }
  }
}

NN_TEST(test_scale) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    const auto& k = init(dev);

    nn::Tensor X = nn::Tensor::from({1, 2, 3}, dev);
    // In place: the scalar family reads through a view and writes densely, so
    // passing the same pointer twice is scale.
    k.scalar(nn::current_stream(dev), nn::kernels::ScalarOp::MulScalar, 2.0f,
             X.device_ptr(), nn::view_of(X), X.device_ptr(), X.numel());

    const nn::Tensor h = X.to(nn::Device::CPU);
    NN_CHECK(h.host_data()[0] == 2.0f);
    NN_CHECK(h.host_data()[1] == 4.0f);
    NN_CHECK(h.host_data()[2] == 6.0f);
  }
}

NN_TEST(test_axpy) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    const auto& k = init(dev);

    nn::Tensor X = nn::Tensor::from({1, 2}, dev);
    nn::Tensor Y = nn::Tensor::from({10, 20}, dev);
    k.axpy(nn::current_stream(dev), 2.0f, X.device_ptr(), Y.device_ptr(), X.numel());

    const nn::Tensor h = Y.to(nn::Device::CPU);
    NN_CHECK(h.host_data()[0] == 12.0f);
    NN_CHECK(h.host_data()[1] == 24.0f);
  }
}

NN_TEST(test_add) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    const auto& k = init(dev);

    nn::Tensor A = nn::Tensor::from({1, 2}, dev);
    nn::Tensor B = nn::Tensor::from({10, 20}, dev);
    nn::Tensor C = nn::Tensor::zeros({2}, dev);
    k.binary(nn::current_stream(dev), nn::kernels::BinaryOp::Add,
             A.device_ptr(), nn::view_of(A), B.device_ptr(), nn::view_of(B),
             C.device_ptr(), A.numel());

    const nn::Tensor h = C.to(nn::Device::CPU);
    NN_CHECK(h.host_data()[0] == 11.0f);
    NN_CHECK(h.host_data()[1] == 22.0f);
  }
}

NN_TEST(test_relu) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    const auto& k = init(dev);

    nn::Tensor X = nn::Tensor::from({-1, 0, 1, 2}, dev);
    nn::Tensor Y = nn::Tensor::zeros({4}, dev);
    k.unary(nn::current_stream(dev), nn::kernels::UnaryOp::Relu,
            X.device_ptr(), nn::view_of(X), Y.device_ptr(), X.numel());

    const nn::Tensor h = Y.to(nn::Device::CPU);
    NN_CHECK(h.host_data()[0] == 0.0f);
    NN_CHECK(h.host_data()[1] == 0.0f);
    NN_CHECK(h.host_data()[2] == 1.0f);
    NN_CHECK(h.host_data()[3] == 2.0f);
  }
}

NN_TEST(test_relu_backward) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    const auto& k = init(dev);

    nn::Tensor X  = nn::Tensor::from({-1, 0, 1, 2}, dev);
    nn::Tensor gY = nn::Tensor::from({10, 20, 30, 40}, dev);
    nn::Tensor gX = nn::Tensor::zeros({4}, dev);
    // relu reads only x, so x stands in for the forward result here.
    k.unary_backward(nn::current_stream(dev), nn::kernels::UnaryOp::Relu,
                     X.device_ptr(), nn::view_of(X),
                     X.device_ptr(), nn::view_of(X),
                     gY.device_ptr(), nn::view_of(gY),
                     gX.device_ptr(), X.numel());

    const nn::Tensor h = gX.to(nn::Device::CPU);
    NN_CHECK(h.host_data()[0] == 0.0f);
    NN_CHECK(h.host_data()[1] == 0.0f);
    NN_CHECK(h.host_data()[2] == 30.0f);
    NN_CHECK(h.host_data()[3] == 40.0f);
  }
}
