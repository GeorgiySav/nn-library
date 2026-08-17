#include "test_harness.h"
#include "devices.h"

#include <nn/core/tensor.h>

NN_TEST(test_tensor_zeros) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    nn::Tensor t = nn::Tensor::zeros({2, 3}, dev);
    NN_CHECK(t.numel() == 6);
    NN_CHECK(t.defined());
    NN_CHECK(t.shape() == nn::Shape({2, 3}));
    NN_CHECK(t.device() == dev);
    NN_CHECK(t.dtype() == nn::DType::F32);

    const nn::Tensor h = t.to(nn::Device::CPU);
    for (int64_t i{0}; i < h.numel(); ++i) {
      NN_CHECK(h.host_data()[i] == 0.0f);
    }
  }
}

NN_TEST(test_tensor_full) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    nn::Tensor t = nn::Tensor::full({2, 3}, 42.0f, dev);
    NN_CHECK(t.numel() == 6);
    NN_CHECK(t.defined());
    NN_CHECK(t.shape() == nn::Shape({2, 3}));
    NN_CHECK(t.device() == dev);
    NN_CHECK(t.dtype() == nn::DType::F32);

    const nn::Tensor h = t.to(nn::Device::CPU);
    for (int64_t i{0}; i < h.numel(); ++i) {
      NN_CHECK(h.host_data()[i] == 42.0f);
    }
  }
}

NN_TEST(test_tensor_scalar) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    nn::Tensor t = nn::Tensor::scalar(3.14f, dev);
    NN_CHECK(t.numel() == 1);
    NN_CHECK(t.shape().rank() == 0);
    NN_CHECK(t.defined());
    NN_CHECK(t.shape() == nn::Shape({}));
    NN_CHECK(t.device() == dev);
    NN_CHECK(t.dtype() == nn::DType::F32);
    NN_CHECK(t.item() == 3.14f);
  }
}

NN_TEST(test_tensor_copy) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    // shallow copy -- comparing addresses, never dereferencing, so device_ptr
    // is the right accessor here even on a device tensor
    nn::Tensor t1 = nn::Tensor::full({2, 3}, 1.0f, dev);
    nn::Tensor t2 = t1;
    NN_CHECK(t1.device_ptr() == t2.device_ptr());

    // deep copy
    nn::Tensor t3 = t1.clone();
    NN_CHECK(t1.device_ptr() != t3.device_ptr());
    NN_CHECK(t3.device() == dev);

    const nn::Tensor h = t3.to(nn::Device::CPU);
    for (int64_t i{0}; i < h.numel(); ++i) {
      NN_CHECK(h.host_data()[i] == 1.0f);
    }
  }
}

NN_TEST(test_tensor_round_trip) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    std::vector<float> src(6);
    for (size_t i{0u}; i < src.size(); ++i) src[i] = static_cast<float>(i);

    nn::Tensor t = nn::Tensor::from(src, nn::Shape({2, 3}), dev);
    NN_CHECK(t.device() == dev);

    const nn::Tensor h = t.to(nn::Device::CPU);
    for (int64_t i{0}; i < h.numel(); ++i) {
      NN_CHECK(h.host_data()[i] == static_cast<float>(i));
    }
  }
}

NN_TEST(test_tensor_to_same_device_aliases) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    nn::Tensor t = nn::Tensor::full({4}, 7.0f, dev);
    nn::Tensor same = t.to(dev);
    NN_CHECK(same.device_ptr() == t.device_ptr());
  }
}

NN_TEST(test_tensor_randn) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    nn::Pcg32 rng(1234);
    nn::Tensor t = nn::Tensor::randn({10000}, rng, 0.5f, dev);
    NN_CHECK(t.device() == dev);

    const nn::Tensor h = t.to(nn::Device::CPU);
    float mean = 0.0f;
    for (int64_t i{0}; i < h.numel(); ++i) {
      mean += h.host_data()[i];
    }
    mean /= h.numel();
    float var = 0.0f;
    for (int64_t i{0}; i < h.numel(); ++i) {
      float diff = h.host_data()[i] - mean;
      var += diff * diff;
    }
    var /= h.numel();
    float stddev = std::sqrt(var);
    NN_CHECK(std::abs(mean) < 0.05f);
    NN_CHECK(std::abs(stddev - 0.5f) < 0.05f);
  }
}

NN_TEST(test_tensor_requires_grad) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    nn::Tensor t = nn::Tensor::zeros({2, 3}, dev);
    NN_CHECK(!t.requires_grad());
    t.set_requires_grad(true);
    NN_CHECK(t.requires_grad());
    NN_CHECK(t.grad().shape() == t.shape());
    // the gradient buffer must live on the same device as the parameter
    NN_CHECK(t.grad().device() == dev);

    t.zero_grad();
    const nn::Tensor g = t.grad().to(nn::Device::CPU);
    for (int64_t i{0}; i < g.numel(); ++i) {
      NN_CHECK(g.host_data()[i] == 0.0f);
    }
  }
}
