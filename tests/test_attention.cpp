// scaled_dot_product_attention: composed out of matmul/softmax/dropout/
// masked_fill, so these tests check the composition itself -- shapes,
// batching/broadcast, the "keep" mask convention, causal masking and
// gradients -- against independent host-side reference computations rather
// than against the library's own primitives.

#include "test_harness.h"
#include "devices.h"
#include "gradcheck.h"

#include <algorithm>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <vector>

#include <nn/autograd/functions.h>
#include <nn/autograd/tape.h>
#include <nn/core/rng.h>
#include <nn/core/tensor.h>

namespace {

std::vector<float> host_of(const nn::Tensor& t) {
  const nn::Tensor h = t.pack().to(nn::Device::CPU);
  return std::vector<float>(h.host_data(), h.host_data() + h.numel());
}

nn::Tensor make(const std::vector<float>& v, nn::Shape s, nn::Device d) {
  return nn::Tensor::from(v, s, d);
}

// Independent ground truth: a plain triple-nested-loop attention over flat
// row-major [Tq,Dk]/[Tk,Dk]/[Tk,Dv] host buffers. `keep` is optional and uses
// the same "1 = attend" convention as the library (row-major [Tq,Tk]).
std::vector<float> naive_sdpa(const std::vector<float>& q, const std::vector<float>& k,
                              const std::vector<float>& v, int Tq, int Tk, int Dk, int Dv,
                              bool causal, const std::vector<float>* keep) {
  std::vector<float> out(size_t(Tq) * Dv, 0.0f);
  const float scale = 1.0f / std::sqrt(float(Dk));

  for (int i = 0; i < Tq; ++i) {
    std::vector<float> scores(Tk);
    for (int j = 0; j < Tk; ++j) {
      float dot = 0.0f;
      for (int d = 0; d < Dk; ++d) dot += q[size_t(i) * Dk + d] * k[size_t(j) * Dk + d];
      scores[j] = dot * scale;

      const bool blocked_causal = causal && (j > i);
      const bool blocked_mask = keep && ((*keep)[size_t(i) * Tk + j] == 0.0f);
      if (blocked_causal || blocked_mask) scores[j] = -1e30f;
    }

    const float m = *std::max_element(scores.begin(), scores.end());
    std::vector<float> p(Tk);
    float sum = 0.0f;
    for (int j = 0; j < Tk; ++j) { p[j] = std::exp(scores[j] - m); sum += p[j]; }
    for (int j = 0; j < Tk; ++j) p[j] /= sum;

    for (int d = 0; d < Dv; ++d) {
      float acc = 0.0f;
      for (int j = 0; j < Tk; ++j) acc += p[j] * v[size_t(j) * Dv + d];
      out[size_t(i) * Dv + d] = acc;
    }
  }
  return out;
}

// A fixed, non-uniform, order-one weight for whatever shape the op produced
// -- same role as in test_reductions.cpp's gradient checks.
nn::Tensor weights_like(const nn::Tensor& y) {
  std::vector<float> w(size_t(y.numel()));
  for (size_t i = 0; i < w.size(); ++i) w[i] = 0.7f + 0.9f * std::sin(1.7f * float(i));
  return nn::Tensor::from(w, y.shape(), y.device());
}

float grad_error(nn::Tensor& param, const std::function<nn::Tensor()>& f, int checks = 10) {
  nn::autograd::Tape tape;
  nn::Tensor loss;
  auto loss_fn = [&]() -> float {
    tape.clear();
    nn::autograd::TapeScope scope(tape);
    const nn::Tensor y = f();
    loss = nn::autograd::sum_all(nn::autograd::mul(y, weights_like(y)));
    return loss.item();
  };
  auto backward_fn = [&]() { tape.backward(loss, true); };
  return nn::test::gradCheck(param, loss_fn, backward_fn, checks);
}

}  // namespace

NN_TEST(sdpa_matches_naive_reference_no_mask) {
  const int Tq = 5, Tk = 6, Dk = 4, Dv = 3;

  NN_TEST_FOR_EACH_DEVICE(dev) {
    nn::Pcg32 rng(1);
    std::vector<float> qh(size_t(Tq) * Dk), kh(size_t(Tk) * Dk), vh(size_t(Tk) * Dv);
    for (auto& x : qh) x = rng.next_normal();
    for (auto& x : kh) x = rng.next_normal();
    for (auto& x : vh) x = rng.next_normal();

    const nn::Tensor q = make(qh, nn::Shape({Tq, Dk}), dev);
    const nn::Tensor k = make(kh, nn::Shape({Tk, Dk}), dev);
    const nn::Tensor v = make(vh, nn::Shape({Tk, Dv}), dev);

    const nn::Tensor out = nn::scaled_dot_product_attention(q, k, v, nn::Tensor());
    NN_CHECK(out.shape() == nn::Shape({Tq, Dv}));

    const std::vector<float> want = naive_sdpa(qh, kh, vh, Tq, Tk, Dk, Dv, false, nullptr);
    const std::vector<float> got = host_of(out);
    for (size_t i = 0; i < want.size(); ++i) NN_CHECK_CLOSE(got[i], want[i], 1e-4);
  }
}

NN_TEST(sdpa_causal_matches_naive_reference) {
  const int T = 5, Dk = 4, Dv = 3;

  NN_TEST_FOR_EACH_DEVICE(dev) {
    nn::Pcg32 rng(2);
    std::vector<float> qh(size_t(T) * Dk), kh(size_t(T) * Dk), vh(size_t(T) * Dv);
    for (auto& x : qh) x = rng.next_normal();
    for (auto& x : kh) x = rng.next_normal();
    for (auto& x : vh) x = rng.next_normal();

    const nn::Tensor q = make(qh, nn::Shape({T, Dk}), dev);
    const nn::Tensor k = make(kh, nn::Shape({T, Dk}), dev);
    const nn::Tensor v = make(vh, nn::Shape({T, Dv}), dev);

    const nn::Tensor out = nn::scaled_dot_product_attention(
        q, k, v, nn::Tensor(), /*dropout_p=*/0.0f, /*is_causal=*/true);

    const std::vector<float> want = naive_sdpa(qh, kh, vh, T, T, Dk, Dv, true, nullptr);
    const std::vector<float> got = host_of(out);
    for (size_t i = 0; i < want.size(); ++i) NN_CHECK_CLOSE(got[i], want[i], 1e-4);
  }
}

NN_TEST(sdpa_causal_output_is_independent_of_future_values) {
  // A property test that doesn't depend on the reference implementation
  // above: perturbing k/v at position j must not change the output at any
  // query position i < j, if masking is genuinely causal.
  const int T = 6, Dk = 3, Dv = 2;

  NN_TEST_FOR_EACH_DEVICE(dev) {
    nn::Pcg32 rng(3);
    std::vector<float> qh(size_t(T) * Dk), kh(size_t(T) * Dk), vh(size_t(T) * Dv);
    for (auto& x : qh) x = rng.next_normal();
    for (auto& x : kh) x = rng.next_normal();
    for (auto& x : vh) x = rng.next_normal();

    auto run = [&](const std::vector<float>& kk, const std::vector<float>& vv) {
      const nn::Tensor q = make(qh, nn::Shape({T, Dk}), dev);
      const nn::Tensor k = make(kk, nn::Shape({T, Dk}), dev);
      const nn::Tensor v = make(vv, nn::Shape({T, Dv}), dev);
      return host_of(nn::scaled_dot_product_attention(
          q, k, v, nn::Tensor(), 0.0f, /*is_causal=*/true));
    };

    const std::vector<float> base = run(kh, vh);

    std::vector<float> kh2 = kh, vh2 = vh;
    const int future = T - 1;
    for (int d = 0; d < Dk; ++d) kh2[size_t(future) * Dk + d] += 5.0f;
    for (int d = 0; d < Dv; ++d) vh2[size_t(future) * Dv + d] += 5.0f;
    const std::vector<float> perturbed = run(kh2, vh2);

    for (int i = 0; i < future; ++i) {
      for (int d = 0; d < Dv; ++d) {
        NN_CHECK_CLOSE(base[size_t(i) * Dv + d], perturbed[size_t(i) * Dv + d], 1e-5);
      }
    }
    // the last row can (and generically will) change, since it may attend to itself
  }
}

NN_TEST(sdpa_mask_blocking_a_key_matches_removing_it) {
  const int Tq = 4, Tk = 5, Dk = 3, Dv = 2;
  const int blocked = 2;

  NN_TEST_FOR_EACH_DEVICE(dev) {
    nn::Pcg32 rng(4);
    std::vector<float> qh(size_t(Tq) * Dk), kh(size_t(Tk) * Dk), vh(size_t(Tk) * Dv);
    for (auto& x : qh) x = rng.next_normal();
    for (auto& x : kh) x = rng.next_normal();
    for (auto& x : vh) x = rng.next_normal();

    std::vector<float> keep(size_t(Tq) * Tk, 1.0f);
    for (int i = 0; i < Tq; ++i) keep[size_t(i) * Tk + blocked] = 0.0f;

    const nn::Tensor q = make(qh, nn::Shape({Tq, Dk}), dev);
    const nn::Tensor k = make(kh, nn::Shape({Tk, Dk}), dev);
    const nn::Tensor v = make(vh, nn::Shape({Tk, Dv}), dev);
    const nn::Tensor mask = make(keep, nn::Shape({Tq, Tk}), dev);

    const nn::Tensor out = nn::scaled_dot_product_attention(q, k, v, mask);

    // Ground truth: attention with column `blocked` deleted entirely.
    std::vector<float> kh_reduced, vh_reduced;
    for (int j = 0; j < Tk; ++j) {
      if (j == blocked) continue;
      kh_reduced.insert(kh_reduced.end(), kh.begin() + size_t(j) * Dk, kh.begin() + size_t(j + 1) * Dk);
      vh_reduced.insert(vh_reduced.end(), vh.begin() + size_t(j) * Dv, vh.begin() + size_t(j + 1) * Dv);
    }
    const std::vector<float> want = naive_sdpa(qh, kh_reduced, vh_reduced, Tq, Tk - 1, Dk, Dv, false, nullptr);
    const std::vector<float> got = host_of(out);
    for (size_t i = 0; i < want.size(); ++i) NN_CHECK_CLOSE(got[i], want[i], 1e-4);
  }
}

NN_TEST(sdpa_batched_matches_per_slice_reference) {
  const int B = 2, H = 2, Tq = 4, Tk = 4, Dk = 3, Dv = 2;

  NN_TEST_FOR_EACH_DEVICE(dev) {
    nn::Pcg32 rng(5);
    std::vector<float> qh(size_t(B) * H * Tq * Dk), kh(size_t(B) * H * Tk * Dk),
        vh(size_t(B) * H * Tk * Dv);
    for (auto& x : qh) x = rng.next_normal();
    for (auto& x : kh) x = rng.next_normal();
    for (auto& x : vh) x = rng.next_normal();

    const nn::Tensor q = make(qh, nn::Shape({B, H, Tq, Dk}), dev);
    const nn::Tensor k = make(kh, nn::Shape({B, H, Tk, Dk}), dev);
    const nn::Tensor v = make(vh, nn::Shape({B, H, Tk, Dv}), dev);

    const nn::Tensor out = nn::scaled_dot_product_attention(
        q, k, v, nn::Tensor(), 0.0f, /*is_causal=*/true);
    NN_CHECK(out.shape() == nn::Shape({B, H, Tq, Dv}));
    const std::vector<float> got = host_of(out);

    for (int b = 0; b < B; ++b) {
      for (int h = 0; h < H; ++h) {
        const size_t qoff = (size_t(b) * H + h) * Tq * Dk;
        const size_t koff = (size_t(b) * H + h) * Tk * Dk;
        const size_t voff = (size_t(b) * H + h) * Tk * Dv;
        const size_t ooff = (size_t(b) * H + h) * Tq * Dv;

        const std::vector<float> qs(qh.begin() + qoff, qh.begin() + qoff + size_t(Tq) * Dk);
        const std::vector<float> ks(kh.begin() + koff, kh.begin() + koff + size_t(Tk) * Dk);
        const std::vector<float> vs(vh.begin() + voff, vh.begin() + voff + size_t(Tk) * Dv);

        const std::vector<float> want = naive_sdpa(qs, ks, vs, Tq, Tk, Dk, Dv, true, nullptr);
        for (size_t i = 0; i < want.size(); ++i) NN_CHECK_CLOSE(got[ooff + i], want[i], 1e-4);
      }
    }
  }
}

NN_TEST(sdpa_dropout_is_a_noop_unless_training) {
  const int Tq = 4, Tk = 4, Dk = 3, Dv = 2;

  NN_TEST_FOR_EACH_DEVICE(dev) {
    nn::Pcg32 rng(6);
    std::vector<float> qh(size_t(Tq) * Dk), kh(size_t(Tk) * Dk), vh(size_t(Tk) * Dv);
    for (auto& x : qh) x = rng.next_normal();
    for (auto& x : kh) x = rng.next_normal();
    for (auto& x : vh) x = rng.next_normal();

    const nn::Tensor q = make(qh, nn::Shape({Tq, Dk}), dev);
    const nn::Tensor k = make(kh, nn::Shape({Tk, Dk}), dev);
    const nn::Tensor v = make(vh, nn::Shape({Tk, Dv}), dev);

    const std::vector<float> no_dropout =
        host_of(nn::scaled_dot_product_attention(q, k, v, nn::Tensor(), 0.0f, false, false));
    const std::vector<float> high_p_not_training =
        host_of(nn::scaled_dot_product_attention(q, k, v, nn::Tensor(), 0.9f, false, /*training=*/false));

    for (size_t i = 0; i < no_dropout.size(); ++i) {
      NN_CHECK_CLOSE(no_dropout[i], high_p_not_training[i], 1e-6);
    }
  }
}

NN_TEST(gradcheck_sdpa_causal_and_mask) {
  const int Tq = 4, Tk = 4, Dk = 3, Dv = 3;

  NN_TEST_FOR_EACH_DEVICE(dev) {
    nn::Pcg32 rng(7);
    nn::Tensor q = nn::Tensor::randn({Tq, Dk}, rng, 0.7f, dev);
    nn::Tensor k = nn::Tensor::randn({Tk, Dk}, rng, 0.7f, dev);
    nn::Tensor v = nn::Tensor::randn({Tk, Dv}, rng, 0.7f, dev);
    q.set_requires_grad(true);
    k.set_requires_grad(true);
    v.set_requires_grad(true);

    auto forward = [&] {
      return nn::scaled_dot_product_attention(q, k, v, nn::Tensor(), 0.0f, /*is_causal=*/true);
    };

    NN_CHECK(grad_error(q, forward) < 3e-2f);
    NN_CHECK(grad_error(k, forward) < 3e-2f);
    NN_CHECK(grad_error(v, forward) < 3e-2f);
  }
}

NN_TEST(sdpa_rejects_bad_shapes) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    const nn::Tensor q = nn::Tensor::full({4, 3}, 0.1f, dev);
    const nn::Tensor k = nn::Tensor::full({4, 3}, 0.1f, dev);
    const nn::Tensor v = nn::Tensor::full({4, 2}, 0.1f, dev);

    // rank mismatch
    NN_CHECK_THROWS(
        nn::scaled_dot_product_attention(nn::Tensor::full({1, 4, 3}, 0.1f, dev), k, v, nn::Tensor()),
        std::invalid_argument);

    // head-dim mismatch between q and k
    NN_CHECK_THROWS(
        nn::scaled_dot_product_attention(q, nn::Tensor::full({4, 5}, 0.1f, dev), v, nn::Tensor()),
        std::invalid_argument);

    // sequence-length mismatch between k and v
    NN_CHECK_THROWS(
        nn::scaled_dot_product_attention(q, k, nn::Tensor::full({5, 2}, 0.1f, dev), nn::Tensor()),
        std::invalid_argument);

    // is_causal requires Tq == Tk
    NN_CHECK_THROWS(
        nn::scaled_dot_product_attention(nn::Tensor::full({3, 3}, 0.1f, dev), k, v, nn::Tensor(),
                                         0.0f, /*is_causal=*/true),
        std::invalid_argument);
  }
}
