#include "test_harness.h"
#include "devices.h"
#include "utilities.h"

#include <nn/autograd/tape.h>
#include <nn/autograd/functions.h>
#include <nn/core/rng.h>
#include <nn/core/tensor.h>
#include <nn/ops/ops.h>

#include <cmath>
#include <stdexcept>
#include <vector>

#include "gradcheck.h"

namespace {

struct Weighted {
  nn::Tensor loss;
  nn::Tensor probs;
};

Weighted run_weighted(const nn::Tensor& logits, const nn::Tensor& labels,
                      const nn::Tensor& weights) {
  Weighted out{nn::Tensor::scalar(0.0f, logits.device()),
               nn::Tensor(logits.shape(), logits.device(), logits.dtype())};
  nn::ops::softmax_ce_weighted(logits, labels, weights, out.loss, out.probs);
  return out;
}

}  // namespace

NN_TEST(weighted_ones_matches_unweighted_forward_and_backward) {
  const int M = 4, N = 6;

  NN_TEST_FOR_EACH_DEVICE(dev) {
    nn::Pcg32 rng(11);
    const nn::Tensor logits = nn::Tensor::randn({M, N}, rng, 1.0f, dev);

    std::vector<int32_t> labels_host(M);
    for (int i{0}; i < M; ++i) labels_host[i] = static_cast<int32_t>(rng.next_uint32() % N);
    const nn::Tensor labels = nn::Tensor::from_i32(labels_host, nn::Shape({M}), dev);
    const nn::Tensor ones = nn::Tensor::full({M}, 1.0f, dev);

    nn::Tensor loss_ref = nn::Tensor::scalar(0.0f, dev);
    nn::Tensor probs_ref(logits.shape(), dev, logits.dtype());
    nn::ops::softmax_ce(logits, labels, loss_ref, probs_ref);

    const Weighted w = run_weighted(logits, labels, ones);

    NN_CHECK_CLOSE(w.loss.item(), loss_ref.item(), 1e-6);

    const nn::Tensor pw = w.probs.to(nn::Device::CPU);
    const nn::Tensor pr = probs_ref.to(nn::Device::CPU);
    for (int i{0}; i < M * N; ++i) {
      NN_CHECK_CLOSE(pw.host_data()[i], pr.host_data()[i], 1e-6);
    }

    const nn::Tensor g_loss = nn::Tensor::scalar(1.0f, dev);
    const nn::Tensor g_ref = nn::ops::softmax_ce_backward(probs_ref, labels, g_loss);
    const nn::Tensor g_w = nn::ops::softmax_ce_weighted_backward(w.probs, labels, ones, g_loss);

    const nn::Tensor g_ref_h = g_ref.to(nn::Device::CPU);
    const nn::Tensor g_w_h = g_w.to(nn::Device::CPU);
    for (int i{0}; i < M * N; ++i) {
      NN_CHECK_CLOSE(g_w_h.host_data()[i], g_ref_h.host_data()[i], 1e-6);
    }
  }
}

NN_TEST(weighted_matches_per_row_reference) {
  const int M = 4, N = 5;

  NN_TEST_FOR_EACH_DEVICE(dev) {
    nn::Pcg32 rng(22);
    std::vector<float> logits_host(size_t(M) * N);
    fill_random(logits_host, rng);
    const nn::Tensor logits = nn::Tensor::from(logits_host, nn::Shape({M, N}), dev);

    std::vector<int32_t> labels_host(M);
    std::vector<float> weights_host(M);
    for (int i{0}; i < M; ++i) {
      labels_host[i] = static_cast<int32_t>(rng.next_uint32() % N);
      weights_host[i] = 0.25f + 2.0f * (rng.next_uint32() % 1000) / 1000.0f;
    }
    const nn::Tensor labels = nn::Tensor::from_i32(labels_host, nn::Shape({M}), dev);
    const nn::Tensor weights = nn::Tensor::from(weights_host, nn::Shape({M}), dev);

    // Ground truth: the slow per-row workaround being replaced -- one
    // unweighted softmax_ce call per row, combined by hand.
    float reference_loss = 0.0f;
    for (int i{0}; i < M; ++i) {
      const nn::Tensor row_logits =
          nn::Tensor::from(std::vector<float>(logits_host.begin() + i * N,
                                              logits_host.begin() + (i + 1) * N),
                           nn::Shape({1, N}), dev);
      const nn::Tensor row_label = nn::Tensor::from_i32({labels_host[i]}, dev);

      nn::Tensor row_loss = nn::Tensor::scalar(0.0f, dev);
      nn::Tensor row_probs(row_logits.shape(), dev, row_logits.dtype());
      nn::ops::softmax_ce(row_logits, row_label, row_loss, row_probs);

      reference_loss += weights_host[i] * row_loss.item();
    }
    reference_loss /= static_cast<float>(M);

    const Weighted w = run_weighted(logits, labels, weights);
    NN_CHECK_CLOSE(w.loss.item(), reference_loss, 1e-4);
  }
}

NN_TEST(gradcheck_cross_entropy_weighted) {
  const int M = 4, N = 5;

  NN_TEST_FOR_EACH_DEVICE(dev) {
    nn::Pcg32 rng(33);
    nn::Tensor logits = nn::Tensor::randn({M, N}, rng, 1.0f, dev);
    logits.set_requires_grad(true);

    std::vector<int32_t> labels_host(M);
    std::vector<float> weights_host(M);
    for (int i{0}; i < M; ++i) {
      labels_host[i] = static_cast<int32_t>(rng.next_uint32() % N);
      weights_host[i] = 0.1f + 3.0f * (rng.next_uint32() % 1000) / 1000.0f;
    }
    const nn::Tensor labels = nn::Tensor::from_i32(labels_host, nn::Shape({M}), dev);
    const nn::Tensor weights = nn::Tensor::from(weights_host, nn::Shape({M}), dev);

    nn::autograd::Tape tape;
    nn::Tensor loss;

    auto forward = [&]() -> float {
      tape.clear();
      nn::autograd::TapeScope scope(tape);
      loss = nn::autograd::cross_entropy(logits, labels, weights);
      return loss.item();
    };
    auto backward = [&]() { tape.backward(loss); };

    NN_CHECK(nn::test::gradCheck(logits, forward, backward) < 2e-2f);
  }
}

NN_TEST(weighted_forward_rejects_bad_weight_shape) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    const int M = 3, N = 4;
    const nn::Tensor logits = nn::Tensor::full({M, N}, 0.1f, dev);
    const nn::Tensor labels = nn::Tensor::from_i32({0, 1, 2}, dev);

    nn::Tensor loss = nn::Tensor::scalar(0.0f, dev);
    nn::Tensor probs(logits.shape(), dev, logits.dtype());

    // wrong rank
    const nn::Tensor weights_2d = nn::Tensor::full({M, 1}, 1.0f, dev);
    NN_CHECK_THROWS(nn::ops::softmax_ce_weighted(logits, labels, weights_2d, loss, probs),
                    std::invalid_argument);

    // wrong length
    const nn::Tensor weights_short = nn::Tensor::full({M - 1}, 1.0f, dev);
    NN_CHECK_THROWS(nn::ops::softmax_ce_weighted(logits, labels, weights_short, loss, probs),
                    std::invalid_argument);
  }
}

NN_TEST(weighted_backward_rejects_bad_weight_shape) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    const int M = 3, N = 4;
    const nn::Tensor probs = nn::Tensor::full({M, N}, 0.25f, dev);
    const nn::Tensor labels = nn::Tensor::from_i32({0, 1, 2}, dev);
    const nn::Tensor g_loss = nn::Tensor::scalar(1.0f, dev);

    const nn::Tensor weights_2d = nn::Tensor::full({M, 1}, 1.0f, dev);
    NN_CHECK_THROWS(nn::ops::softmax_ce_weighted_backward(probs, labels, weights_2d, g_loss),
                    std::invalid_argument);

    const nn::Tensor weights_long = nn::Tensor::full({M + 1}, 1.0f, dev);
    NN_CHECK_THROWS(nn::ops::softmax_ce_weighted_backward(probs, labels, weights_long, g_loss),
                    std::invalid_argument);
  }
}
