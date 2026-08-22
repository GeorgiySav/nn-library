#include "test_harness.h"
#include "devices.h"

#include <nn/core/tensor.h>
#include <nn/kernels/kernel_api.h>

const nn::kernels::KernelTable& init(nn::Device d) {
  nn::kernels::init_kernels();
  const nn::kernels::KernelTable& kernels = nn::kernels::kernels(d);
  NN_CHECK(kernels.gemm != nullptr);
  NN_CHECK(kernels.add_row_bias != nullptr);
  NN_CHECK(kernels.col_sum != nullptr);
  NN_CHECK(kernels.relu != nullptr);
  NN_CHECK(kernels.relu_backward != nullptr);
  NN_CHECK(kernels.add != nullptr);
  NN_CHECK(kernels.scale != nullptr);
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
    k.scale(nn::current_stream(dev), 2.0f, X.device_ptr(), X.numel());

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
    k.add(nn::current_stream(dev), A.device_ptr(), B.device_ptr(), C.device_ptr(), A.numel());

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
    k.relu(nn::current_stream(dev), X.device_ptr(), Y.device_ptr(), X.numel());

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
    k.relu_backward(nn::current_stream(dev), X.device_ptr(), gY.device_ptr(),
                    gX.device_ptr(), X.numel());

    const nn::Tensor h = gX.to(nn::Device::CPU);
    NN_CHECK(h.host_data()[0] == 0.0f);
    NN_CHECK(h.host_data()[1] == 0.0f);
    NN_CHECK(h.host_data()[2] == 30.0f);
    NN_CHECK(h.host_data()[3] == 40.0f);
  }
}

NN_TEST(test_add_row_bias) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    const auto& k = init(dev);

    nn::Tensor X = nn::Tensor::full({2, 3}, 1.0f, dev);
    nn::Tensor b = nn::Tensor::from({10, 20, 30}, dev);
    nn::Tensor Y = nn::Tensor::zeros({2, 3}, dev);
    k.add_row_bias(nn::current_stream(dev), X.device_ptr(), b.device_ptr(),
                   Y.device_ptr(), 2, 3, /*sx=*/3);

    const nn::Tensor h = Y.to(nn::Device::CPU);
    NN_CHECK(h.host_data()[0] == 11.0f);
    NN_CHECK(h.host_data()[1] == 21.0f);
    NN_CHECK(h.host_data()[2] == 31.0f);
    NN_CHECK(h.host_data()[3] == 11.0f);
    NN_CHECK(h.host_data()[4] == 21.0f);
    NN_CHECK(h.host_data()[5] == 31.0f);
  }
}

NN_TEST(test_col_sum) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    const auto& k = init(dev);

    nn::Tensor X   = nn::Tensor::from({{1, 2, 3}, {4, 5, 6}}, dev);
    nn::Tensor out = nn::Tensor::zeros({3}, dev);
    k.col_sum(nn::current_stream(dev), X.device_ptr(), out.device_ptr(), 2, 3, /*sx=*/3);

    const nn::Tensor h = out.to(nn::Device::CPU);
    NN_CHECK(h.host_data()[0] == 5.0f);
    NN_CHECK(h.host_data()[1] == 7.0f);
    NN_CHECK(h.host_data()[2] == 9.0f);
  }
}
