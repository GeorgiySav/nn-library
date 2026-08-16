#include "naive_kernels.h"

#include <cmath>
#include <cassert>

namespace nn::kernels {

void naive_softmax_ce(const float* logits, const int32_t* labels,
                      float* loss_out, float* probs, int M, int N) {
  /*
  logits: [M, N]
  labels: [M] (int32)
  *loss_out: mean cross-entropy over the batch
  probs: [M, N] (softmax probabilities)
  g_logits = g_loss * (probs - one_hot(labels)) / M

  probs[i, j] = exp(z[i, j] - m_i) / sum_k exp(z[i, k] - m_i), m_i = max_k z[i, k]
  loss        = -(1/M) * sum_i log(probs[i, labels[i]])
  */

  assert(M >= 0 && N > 0);
  float total_loss = 0.0f;

  for (int i{0}; i < M; ++i) {
    const float* z = logits + static_cast<int64_t>(i) * N;
    float* p = probs + static_cast<int64_t>(i) * N;

    const int32_t label = labels[i];
    assert(label >= 0 && label < N);

    float m = z[0];
    for (int j{1}; j < N; ++j) {
      if (z[j] > m) {
        m = z[j];
      }
    }
    
    float s = 0.0f;
    for (int j{0}; j < N; ++j) {
      float exp_z = std::exp(z[j] - m);
      p[j] = exp_z;
      s += exp_z;
    }

    const float inv_s = 1.0f / s;
    for (int j{0}; j < N; ++j) {
      p[j] *= inv_s;
    }

    total_loss -= (z[label] - m) - std::log(s);
  }

  *loss_out = (M > 0) ? total_loss / static_cast<float>(M) : 0.0f;
}

void naive_softmax_ce_backward(const float* probs, const int32_t* labels,
                               const float* g_loss, float* g_logits, int M, int N) {
  /*
  probs: [M, N] (softmax probabilities)
  labels: [M] (int32)
  g_loss: gradient of the loss w.r.t. the output
  g_logits: [M, N] (gradient of the loss w.r.t. the logits)
  */
  assert(M >= 0 && N > 0);

  const float scale = (M > 0) ? *g_loss / static_cast<float>(M) : 0.0f;

  for (int i{0}; i < M; ++i) {
    const float* p = probs + static_cast<int64_t>(i) * N;
    float* g = g_logits + static_cast<int64_t>(i) * N;

    const int32_t label = labels[i];
    assert(label >= 0 && label < N);

    for (int j{0}; j < N; ++j) {
      g[j] = scale * p[j];
    }
    g[label] -= scale;
  }
}

}