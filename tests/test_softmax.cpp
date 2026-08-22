#include "test_harness.h"
#include "devices.h"
#include "utilities.h"

#include <nn/core/rng.h>
#include <nn/core/tensor.h>
#include <nn/kernels/kernel_api.h>

nn::kernels::SoftmaxCeFn softmax_ce(nn::Device d) {
  nn::kernels::init_kernels();
  const auto& k = nn::kernels::kernels(d);
  NN_CHECK(k.softmax_ce != nullptr);
  return k.softmax_ce;
}

struct SoftmaxOut {
  nn::Tensor loss;
  nn::Tensor probs;
};

SoftmaxOut run_softmax_ce(const nn::Tensor& logits, const nn::Tensor& labels,
                          int M, int N, nn::Device d) {
  SoftmaxOut out{nn::Tensor::scalar(0.0f, d),
                 nn::Tensor::full({M, N}, -1.0f, d)};
  softmax_ce(d)(nn::current_stream(d), logits.device_ptr(), labels.device_ptr_i32(),
                out.loss.device_ptr(), out.probs.device_ptr(), M, N, /*sz=*/N);
  return out;
}

NN_TEST(test_uniform_logits) {
  const int M = 1, N = 4;

  NN_TEST_FOR_EACH_DEVICE(dev) {
    const nn::Tensor logits = nn::Tensor::full({M, N}, 0.5f, dev);
    const nn::Tensor labels = nn::Tensor::from_i32({2}, dev);

    const SoftmaxOut r = run_softmax_ce(logits, labels, M, N, dev);

    const nn::Tensor p = r.probs.to(nn::Device::CPU);
    for (int j{0}; j < N; ++j) {
      NN_CHECK_CLOSE(p.host_data()[j], 0.25f, 1e-6);
    }
    NN_CHECK_CLOSE(r.loss.item(), std::log(4.0f), 1e-6);
  }
}

NN_TEST(test_row_sum_to_one) {
  const int M = 8, N = 10;

  NN_TEST_FOR_EACH_DEVICE(dev) {
    nn::Pcg32 rng(1234);
    const nn::Tensor logits = nn::Tensor::randn({M, N}, rng, 1.0f, dev);
    const nn::Tensor labels = nn::Tensor::from_i32(std::vector<int32_t>(M, 0),
                                                   nn::Shape({M}), dev);

    const SoftmaxOut r = run_softmax_ce(logits, labels, M, N, dev);

    const nn::Tensor p = r.probs.to(nn::Device::CPU);
    for (int i{0}; i < M; ++i) {
      float row_sum = 0.0f;
      for (int j{0}; j < N; ++j) {
        row_sum += p.host_data()[i*N + j];
      }
      NN_CHECK_CLOSE(row_sum, 1.0f, 1e-6);
    }
  }
}

NN_TEST(test_numerical_stability) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    const nn::Tensor logits = nn::Tensor::from({1000.0f, 1000.0f, 1000.0f}, dev);
    const nn::Tensor labels = nn::Tensor::from_i32({1}, dev);

    const SoftmaxOut r = run_softmax_ce(logits, labels, 1, 3, dev);

    const nn::Tensor p = r.probs.to(nn::Device::CPU);
    for (int j{0}; j < 3; ++j) {
      NN_CHECK_CLOSE(p.host_data()[j], 1.0f/3.0f, 1e-6);
    }
    NN_CHECK_CLOSE(r.loss.item(), std::log(3.0f), 1e-6);

    const nn::Tensor logits2 = nn::Tensor::from({-1000.0f, -1000.0f, -1000.0f}, dev);
    const nn::Tensor labels2 = nn::Tensor::from_i32({2}, dev);

    const SoftmaxOut r2 = run_softmax_ce(logits2, labels2, 1, 3, dev);

    const nn::Tensor p2 = r2.probs.to(nn::Device::CPU);
    for (int j{0}; j < 3; ++j) {
      NN_CHECK_CLOSE(p2.host_data()[j], 1.0f/3.0f, 1e-6);
    }
    NN_CHECK_CLOSE(r2.loss.item(), std::log(3.0f), 1e-6);
  }
}

NN_TEST(test_confident_correct_prediction) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    const nn::Tensor logits = nn::Tensor::from({0.0f, 100.0f, 0.0f}, dev);
    const nn::Tensor labels = nn::Tensor::from_i32({1}, dev);

    const SoftmaxOut r = run_softmax_ce(logits, labels, 1, 3, dev);

    const nn::Tensor p = r.probs.to(nn::Device::CPU);
    NN_CHECK_CLOSE(p.host_data()[1], 1.0f, 1e-6);
    NN_CHECK_CLOSE(r.loss.item(), 0.0f, 1e-6);
  }
}

NN_TEST(test_confident_incorrect_prediction) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    const nn::Tensor logits = nn::Tensor::from({0.0f, 100.0f, 0.0f}, dev);
    const nn::Tensor labels = nn::Tensor::from_i32({0}, dev);

    const SoftmaxOut r = run_softmax_ce(logits, labels, 1, 3, dev);

    const nn::Tensor p = r.probs.to(nn::Device::CPU);
    NN_CHECK_CLOSE(p.host_data()[1], 1.0f, 1e-6);
    NN_CHECK_CLOSE(r.loss.item(), 100.0f, 1e-6);
  }
}

NN_TEST(test_mean_not_sum) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    const nn::Tensor logits = nn::Tensor::from({
      {0.0f, 100.0f, 0.0f},
      {0.0f, 100.0f, 0.0f},
      {0.0f, 100.0f, 0.0f},
      {0.0f, 100.0f, 0.0f}
    }, dev);
    const nn::Tensor labels = nn::Tensor::from_i32({0, 0, 0, 0}, dev);

    const SoftmaxOut r = run_softmax_ce(logits, labels, 4, 3, dev);

    const nn::Tensor p = r.probs.to(nn::Device::CPU);
    NN_CHECK_CLOSE(p.host_data()[1], 1.0f, 1e-6);
    NN_CHECK_CLOSE(r.loss.item(), 100.0f, 1e-6);
  }
}

NN_TEST(test_backward_against_finite_difference) {
  const int M = 3, N = 5;

  NN_TEST_FOR_EACH_DEVICE(dev) {
    nn::Pcg32 rng(1234);
    std::vector<float> logits_host(size_t(M)*N);
    fill_random(logits_host, rng);

    std::vector<int32_t> labels_host(M);
    for (int i{0}; i < M; ++i) {
      labels_host[i] = static_cast<int32_t>(rng.next_uint32() % N);
    }

    const nn::Tensor logits = nn::Tensor::from(logits_host, nn::Shape({M, N}), dev);
    const nn::Tensor labels = nn::Tensor::from_i32(labels_host, nn::Shape({M}), dev);

    const SoftmaxOut r = run_softmax_ce(logits, labels, M, N, dev);

    const auto& k = nn::kernels::kernels(dev);
    NN_CHECK(k.softmax_ce_backward != nullptr);
    const nn::Tensor g_loss = nn::Tensor::scalar(1.0f, dev);
    nn::Tensor g_logits = nn::Tensor::full({M, N}, -1.0f, dev);
    k.softmax_ce_backward(nn::current_stream(dev), r.probs.device_ptr(),
                          labels.device_ptr_i32(), g_loss.device_ptr(),
                          g_logits.device_ptr(), M, N, /*sp=*/N);

    const nn::Tensor g = g_logits.to(nn::Device::CPU);

    const float h = 1e-3f;
    for (int i{0}; i < M*N; ++i) {
      std::vector<float> plus = logits_host, minus = logits_host;
      plus[i]  += h;
      minus[i] -= h;

      const SoftmaxOut rp = run_softmax_ce(
        nn::Tensor::from(plus, nn::Shape({M, N}), dev), labels, M, N, dev);
      const SoftmaxOut rm = run_softmax_ce(
        nn::Tensor::from(minus, nn::Shape({M, N}), dev), labels, M, N, dev);

      const float finite_diff = (rp.loss.item() - rm.loss.item()) / (2.0f * h);
      NN_CHECK_CLOSE(finite_diff, g.host_data()[i], 2e-2f);
    }
  }
}

NN_TEST(test_gradient_rows_sum_to_zero) {
  const int M = 4, N = 6;

  NN_TEST_FOR_EACH_DEVICE(dev) {
    nn::Pcg32 rng(1234);
    const nn::Tensor logits = nn::Tensor::randn({M, N}, rng, 1.0f, dev);

    std::vector<int32_t> labels_host(M);
    for (int i{0}; i < M; ++i) {
      labels_host[i] = static_cast<int32_t>(rng.next_uint32() % N);
    }
    const nn::Tensor labels = nn::Tensor::from_i32(labels_host, nn::Shape({M}), dev);

    const SoftmaxOut r = run_softmax_ce(logits, labels, M, N, dev);

    const auto& k = nn::kernels::kernels(dev);
    NN_CHECK(k.softmax_ce_backward != nullptr);
    const nn::Tensor g_loss = nn::Tensor::scalar(1.0f, dev);
    nn::Tensor g_logits = nn::Tensor::full({M, N}, -1.0f, dev);
    k.softmax_ce_backward(nn::current_stream(dev), r.probs.device_ptr(),
                          labels.device_ptr_i32(), g_loss.device_ptr(),
                          g_logits.device_ptr(), M, N, /*sp=*/N);

    const nn::Tensor g = g_logits.to(nn::Device::CPU);
    for (int i{0}; i < M; ++i) {
      float row_sum = 0.0f;
      for (int j{0}; j < N; ++j) {
        row_sum += g.host_data()[i*N + j];
      }
      NN_CHECK_CLOSE(row_sum, 0.0f, 1e-6);
    }
  }
}
