#include "test_harness.h"
#include "utilities.h"

#include <nn/core/rng.h>
#include <nn/kernels/kernel_api.h>

nn::kernels::SoftmaxCeFn softmax_ce() {
  nn::kernels::init_kernels();
  const auto& k = nn::kernels::kernels(nn::Device::CPU);
  NN_CHECK(k.softmax_ce != nullptr);
  return k.softmax_ce;
}

NN_TEST(test_uniform_logits) {
  int M = 1, N = 4;
  std::vector<float> logits(M*N, 0.5f);
  std::vector<int32_t> labels{2};
  std::vector<float> probs(M*N, -1.0f);

  float loss = 0.0f;
  softmax_ce()(logits.data(), labels.data(), &loss, probs.data(), M, N);

  for (int j{0}; j < N; ++j) {
    NN_CHECK_CLOSE(probs[j], 0.25f, 1e-6);
  }
  NN_CHECK_CLOSE(loss, std::log(4.0f), 1e-6);
}

NN_TEST(test_row_sum_to_one) {
  nn::Pcg32 rng(1234);
  std::vector<float> logits(8*10, -1.0f);
  fill_random(logits, rng);
  std::vector<int32_t> labels(8, 0);
  std::vector<float> probs(8*10, -1.0f);
  float loss = 0.0f;
  softmax_ce()(logits.data(), labels.data(), &loss, probs.data(), 8, 10);
  for (int i{0}; i < 8; ++i) {
    float row_sum = 0.0f;
    for (int j{0}; j < 10; ++j) {
      row_sum += probs[i*10 + j];
    }
    NN_CHECK_CLOSE(row_sum, 1.0f, 1e-6);
  }
}

NN_TEST(test_numerical_stability) {
  std::vector<float> logits{1000.0f, 1000.0f, 1000.0f};
  std::vector<int32_t> labels{1};
  std::vector<float> probs(3, -1.0f);
  float loss = 0.0f;
  softmax_ce()(logits.data(), labels.data(), &loss, probs.data(), 1, 3);
  for (int j{0}; j < 3; ++j) {
    NN_CHECK_CLOSE(probs[j], 1.0f/3.0f, 1e-6);
  }
  NN_CHECK_CLOSE(loss, std::log(3.0f), 1e-6);

  std::vector<float> logits2{-1000.0f, -1000.0f, -1000.0f};
  std::vector<int32_t> labels2{2};
  std::vector<float> probs2(3, -1.0f);
  float loss2 = 0.0f;
  softmax_ce()(logits2.data(), labels2.data(), &loss2, probs2.data(), 1, 3);
  for (int j{0}; j < 3; ++j) {
    NN_CHECK_CLOSE(probs2[j], 1.0f/3.0f, 1e-6);
  }
  NN_CHECK_CLOSE(loss2, std::log(3.0f), 1e-6);
}

NN_TEST(test_confident_correct_prediction) {
  std::vector<float> logits{0.0f, 100.0f, 0.0f};
  std::vector<int32_t> labels{1};
  std::vector<float> probs(3, -1.0f);
  float loss = 0.0f;
  softmax_ce()(logits.data(), labels.data(), &loss, probs.data(), 1, 3);
  NN_CHECK_CLOSE(probs[1], 1.0f, 1e-6);
  NN_CHECK_CLOSE(loss, 0.0f, 1e-6);
}

NN_TEST(test_confident_incorrect_prediction) {
  std::vector<float> logits{0.0f, 100.0f, 0.0f};
  std::vector<int32_t> labels{0};
  std::vector<float> probs(3, -1.0f);
  float loss = 0.0f;
  softmax_ce()(logits.data(), labels.data(), &loss, probs.data(), 1, 3);
  NN_CHECK_CLOSE(probs[1], 1.0f, 1e-6);
  NN_CHECK_CLOSE(loss, 100.0f, 1e-6);
}

NN_TEST(test_mean_not_sum) {
  std::vector<float> logits{
    0.0f, 100.0f, 0.0f,
    0.0f, 100.0f, 0.0f,
    0.0f, 100.0f, 0.0f,
    0.0f, 100.0f, 0.0f
  };
  std::vector<int32_t> labels{0, 0, 0, 0};
  std::vector<float> probs(3*4, -1.0f);
  float loss = 0.0f;
  softmax_ce()(logits.data(), labels.data(), &loss, probs.data(), 4, 3);
  NN_CHECK_CLOSE(probs[1], 1.0f, 1e-6);
  NN_CHECK_CLOSE(loss, 100.0f, 1e-6);
}

NN_TEST(test_backward_against_finite_difference) {
  // random [3, 5] logits and random labels, perturb each of the 15
  // logits by h=1e-3 and compare (loss(+h) - loss(-h)) / (2*h)
  // against the analytic g_logits with tolerance 2e-2

  nn::Pcg32 rng(1234);
  int M = 3, N = 5;
  std::vector<float> logits(M*N, -1.0f);
  fill_random(logits, rng);
  std::vector<int32_t> labels(M, -1);
  for (int i{0}; i < M; ++i) {
    labels[i] = static_cast<int32_t>(rng.next_uint32() % N);
  }
  std::vector<float> probs(M*N, -1.0f);
  float loss = 0.0f;
  softmax_ce()(logits.data(), labels.data(), &loss, probs.data(), M, N);

  std::vector<float> g_logits(M*N, -1.0f);
  const auto& k = nn::kernels::kernels(nn::Device::CPU);
  NN_CHECK(k.softmax_ce_backward != nullptr);
  float g_loss = 1.0f;
  k.softmax_ce_backward(probs.data(), labels.data(), &g_loss, g_logits.data(), M, N);

  const float h = 1e-3f;
  for (int i{0}; i < M*N; ++i) {
    std::vector<float> logits_plus_h = logits;
    std::vector<float> logits_minus_h = logits;
    logits_plus_h[i] += h;
    logits_minus_h[i] -= h;

    std::vector<float> probs_plus_h(M*N, -1.0f);
    float loss_plus_h = 0.0f;
    softmax_ce()(logits_plus_h.data(), labels.data(), &loss_plus_h, probs_plus_h.data(), M, N);

    std::vector<float> probs_minus_h(M*N, -1.0f);
    float loss_minus_h = 0.0f;
    softmax_ce()(logits_minus_h.data(), labels.data(), &loss_minus_h, probs_minus_h.data(), M, N);

    const float finite_diff = (loss_plus_h - loss_minus_h) / (2.0f * h);
    NN_CHECK_CLOSE(finite_diff, g_logits[i], 2e-2f);
  }
}

NN_TEST(test_gradient_rows_sum_to_zero) {
  nn::Pcg32 rng(1234);
  int M = 4, N = 6;
  std::vector<float> logits(M*N, -1.0f);
  fill_random(logits, rng);
  std::vector<int32_t> labels(M, -1);
  for (int i{0}; i < M; ++i) {
    labels[i] = static_cast<int32_t>(rng.next_uint32() % N);
  }
  std::vector<float> probs(M*N, -1.0f);
  float loss = 0.0f;
  softmax_ce()(logits.data(), labels.data(), &loss, probs.data(), M, N);

  std::vector<float> g_logits(M*N, -1.0f);
  const auto& k = nn::kernels::kernels(nn::Device::CPU);
  NN_CHECK(k.softmax_ce_backward != nullptr);
  float g_loss = 1.0f;
  k.softmax_ce_backward(probs.data(), labels.data(), &g_loss, g_logits.data(), M, N);

  for (int i{0}; i < M; ++i) {
    float row_sum = 0.0f;
    for (int j{0}; j < N; ++j) {
      row_sum += g_logits[i*N + j];
    }
    NN_CHECK_CLOSE(row_sum, 0.0f, 1e-6);
  }
}