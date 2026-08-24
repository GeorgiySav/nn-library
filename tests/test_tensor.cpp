#include "test_harness.h"
#include "devices.h"

#include <stdexcept>

#include <nn/core/tensor.h>
#include <nn/autograd/functions.h>
#include <nn/nn/module.h>
#include <nn/ops/ops.h>

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

NN_TEST(test_tensor_arange) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    const nn::Tensor t = nn::Tensor::arange(5, dev);
    NN_CHECK(t.shape() == nn::Shape({5}));
    NN_CHECK(t.dtype() == nn::DType::I32);

    const nn::Tensor h = t.to(nn::Device::CPU);
    for (int i = 0; i < 5; ++i) NN_CHECK(h.host_data_i32()[i] == i);

    // n == 0 is a valid, empty range
    const nn::Tensor empty = nn::Tensor::arange(0, dev);
    NN_CHECK(empty.numel() == 0);

    // F32 for a plain numeric ramp
    const nn::Tensor f = nn::Tensor::arange(4, dev, nn::DType::F32).to(nn::Device::CPU);
    for (int i = 0; i < 4; ++i) NN_CHECK_CLOSE(f.host_data()[i], float(i), 0.0f);

    NN_CHECK_THROWS(nn::Tensor::arange(-1, dev), std::invalid_argument);
  }
}

NN_TEST(test_tensor_arange_feeds_embedding) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    nn::Pcg32 rng(9);
    nn::Embedding pos(8, 4, rng);
    pos.to(dev);

    const nn::Tensor idx = nn::Tensor::arange(8, dev);
    const nn::Tensor emb = pos(idx);
    NN_CHECK(emb.shape() == nn::Shape({8, 4}));
  }
}

NN_TEST(test_tril_mask) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    const int n = 5;
    const nn::Tensor m = nn::tril_mask(n, dev).to(nn::Device::CPU);
    NN_CHECK(m.shape() == nn::Shape({n, n}));

    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        const float want = (j <= i) ? 1.0f : 0.0f;
        NN_CHECK_CLOSE(m.host_data()[i * n + j], want, 0.0f);
      }
    }

    // a token may attend to itself
    for (int i = 0; i < n; ++i) NN_CHECK_CLOSE(m.host_data()[i * n + i], 1.0f, 0.0f);

    // n == 0 and n == 1 are valid edges
    NN_CHECK(nn::tril_mask(0, dev).numel() == 0);
    const nn::Tensor one = nn::tril_mask(1, dev).to(nn::Device::CPU);
    NN_CHECK_CLOSE(one.host_data()[0], 1.0f, 0.0f);

    NN_CHECK_THROWS(nn::tril_mask(-1, dev), std::invalid_argument);
  }
}

NN_TEST(tril_mask_composes_with_masked_fill_for_causal_attention) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    const int T = 4;
    nn::Tensor scores = nn::Tensor::from(
        {{1.0f, 2.0f, 3.0f, 4.0f},
         {1.0f, 2.0f, 3.0f, 4.0f},
         {1.0f, 2.0f, 3.0f, 4.0f},
         {1.0f, 2.0f, 3.0f, 4.0f}}, dev);

    const nn::Tensor causal = nn::tril_mask(T, dev);
    const nn::Tensor masked = nn::masked_fill(scores, 1.0f - causal, -1e9f);
    const nn::Tensor probs = nn::ops::softmax_rows(masked).to(nn::Device::CPU);

    for (int i = 0; i < T; ++i) {
      float row_sum = 0.0f;
      for (int j = 0; j < T; ++j) {
        const float p = probs.host_data()[i * T + j];
        row_sum += p;
        if (j > i) NN_CHECK_CLOSE(p, 0.0f, 1e-6f);   // future: exactly zero
        else       NN_CHECK(p > 0.0f);               // past and self: attended
      }
      NN_CHECK_CLOSE(row_sum, 1.0f, 1e-5f);
    }

    // row 0 can only attend to itself
    NN_CHECK_CLOSE(probs.host_data()[0], 1.0f, 1e-6f);
  }
}
