// Axis reductions, softmax, layer norm, embedding and masked_fill: the ops
// added on top of the elementwise family.

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
#include <nn/ops/ops.h>

namespace {

std::vector<float> host_of(const nn::Tensor& t) {
  const nn::Tensor h = t.pack().to(nn::Device::CPU);
  return std::vector<float>(h.host_data(), h.host_data() + h.numel());
}

std::vector<int32_t> host_of_i32(const nn::Tensor& t) {
  const nn::Tensor h = t.pack().to(nn::Device::CPU);
  return std::vector<int32_t>(h.host_data_i32(), h.host_data_i32() + h.numel());
}

nn::Tensor ramp(nn::Shape s, nn::Device d, float start = -1.3f, float step = 0.17f) {
  std::vector<float> v(size_t(s.numel()));
  for (size_t i = 0; i < v.size(); ++i) v[i] = start + step * float(i);
  return nn::Tensor::from(v, s, d);
}

// A fixed, non-uniform, order-one weight for whatever shape the op produced.
// Deterministic, so the two finite-difference evaluations see the same one.
nn::Tensor weights_like(const nn::Tensor& y) {
  std::vector<float> w(size_t(y.numel()));
  for (size_t i = 0; i < w.size(); ++i) w[i] = 0.7f + 0.9f * std::sin(1.7f * float(i));
  return nn::Tensor::from(w, y.shape(), y.device());
}

// L = sum(op(x) * w). Weighting matters twice over: it sends a varying
// gradient into the backward, so an op that ignored its incoming gradient
// cannot pass, and it keeps dL/dx order one. The obvious alternative,
// sum(y * y), fails the second: on a softmax it produces gradients around
// 1e-3 against a loss of order 1, and a central difference at h = 1e-3 then
// carries more float noise than signal.
float grad_error(nn::Tensor& param, const std::function<nn::Tensor()>& f,
                 int checks = 10) {
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

NN_TEST(sum_over_an_axis_matches_a_hand_sum) {
  const int A = 3, B = 4, C = 5;

  NN_TEST_FOR_EACH_DEVICE(dev) {
    const nn::Tensor x = ramp(nn::Shape({A, B, C}), dev);
    const std::vector<float> h = host_of(x);
    auto at = [&](int a, int b, int c) { return h[(size_t(a) * B + b) * C + c]; };

    {   // axis 0
      const std::vector<float> got = host_of(nn::ops::sum_dim(x, 0, false));
      NN_CHECK(nn::ops::sum_dim(x, 0, false).shape() == nn::Shape({B, C}));
      NN_CHECK(nn::ops::sum_dim(x, 0, true).shape() == nn::Shape({1, B, C}));
      for (int b = 0; b < B; ++b)
        for (int c = 0; c < C; ++c) {
          float want = 0.0f;
          for (int a = 0; a < A; ++a) want += at(a, b, c);
          NN_CHECK_CLOSE(got[size_t(b) * C + c], want, 1e-5);
        }
    }
    {   // axis 1
      const std::vector<float> got = host_of(nn::ops::sum_dim(x, 1, false));
      NN_CHECK(nn::ops::sum_dim(x, 1, false).shape() == nn::Shape({A, C}));
      for (int a = 0; a < A; ++a)
        for (int c = 0; c < C; ++c) {
          float want = 0.0f;
          for (int b = 0; b < B; ++b) want += at(a, b, c);
          NN_CHECK_CLOSE(got[size_t(a) * C + c], want, 1e-5);
        }
    }
    {   // the last axis, addressed from the right
      const std::vector<float> got = host_of(nn::ops::sum_dim(x, -1, false));
      NN_CHECK(nn::ops::sum_dim(x, -1, false).shape() == nn::Shape({A, B}));
      for (int a = 0; a < A; ++a)
        for (int b = 0; b < B; ++b) {
          float want = 0.0f;
          for (int c = 0; c < C; ++c) want += at(a, b, c);
          NN_CHECK_CLOSE(got[size_t(a) * B + b], want, 1e-5);
        }
    }

    NN_CHECK_THROWS(nn::ops::sum_dim(x, 3, false), std::invalid_argument);
    NN_CHECK_THROWS(nn::ops::sum_dim(x, -4, false), std::invalid_argument);
  }
}

NN_TEST(mean_is_the_sum_over_the_extent) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    const nn::Tensor x = ramp(nn::Shape({3, 4, 5}), dev);

    for (int axis = 0; axis < 3; ++axis) {
      const int n = x.shape().dim(axis);
      const std::vector<float> m = host_of(nn::ops::mean_dim(x, axis, false));
      const std::vector<float> s = host_of(nn::ops::sum_dim(x, axis, false));
      for (size_t i = 0; i < m.size(); ++i) NN_CHECK_CLOSE(m[i], s[i] / float(n), 1e-5);
    }

    const float total = nn::ops::sum_all(x).item();
    NN_CHECK_CLOSE(nn::ops::mean_all(x).item(), total / float(x.numel()), 1e-5);
  }
}

// The reduction reads through strides, so a slice must not need materialising.
NN_TEST(sum_over_an_axis_absorbs_strides) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    const nn::Tensor wide = ramp(nn::Shape({4, 9}), dev);
    const nn::Tensor view = wide.slice_view(1, 2, 5);
    NN_CHECK(!view.is_contiguous());

    for (int axis = 0; axis < 2; ++axis) {
      const std::vector<float> a = host_of(nn::ops::sum_dim(view, axis, false));
      const std::vector<float> b = host_of(nn::ops::sum_dim(view.pack(), axis, false));
      for (size_t i = 0; i < a.size(); ++i) NN_CHECK_CLOSE(a[i], b[i], 1e-6);
    }
  }
}

NN_TEST(variance_matches_a_hand_computation) {
  const int R = 4, N = 6;

  NN_TEST_FOR_EACH_DEVICE(dev) {
    nn::Pcg32 rng(7);
    const nn::Tensor x = nn::Tensor::randn({R, N}, rng, 1.0f, dev);
    const std::vector<float> h = host_of(x);

    for (bool unbiased : {true, false}) {
      const std::vector<float> got = host_of(nn::autograd::var(x, 1, false, unbiased));
      for (int r = 0; r < R; ++r) {
        float mean = 0.0f;
        for (int c = 0; c < N; ++c) mean += h[size_t(r) * N + c];
        mean /= float(N);
        float ss = 0.0f;
        for (int c = 0; c < N; ++c) {
          const float d = h[size_t(r) * N + c] - mean;
          ss += d * d;
        }
        NN_CHECK_CLOSE(got[size_t(r)], ss / float(unbiased ? N - 1 : N), 1e-4);
      }
    }

    // stddev is just its square root, and a length-1 axis has no unbiased
    // estimate to give.
    const std::vector<float> sd = host_of(nn::autograd::stddev(x, 1));
    const std::vector<float> v = host_of(nn::autograd::var(x, 1));
    for (size_t i = 0; i < sd.size(); ++i) NN_CHECK_CLOSE(sd[i], std::sqrt(v[i]), 1e-5);

    const nn::Tensor thin = nn::Tensor::zeros({3, 1}, dev);
    NN_CHECK_THROWS(nn::autograd::var(thin, 1, false, true), std::invalid_argument);
  }
}

NN_TEST(gradcheck_sum_mean_var) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    nn::Pcg32 rng(3);

    for (int axis : {0, 1, -1}) {
      for (bool keepdim : {false, true}) {
        nn::Tensor x = nn::Tensor::randn({4, 5}, rng, 0.7f, dev);
        x.set_requires_grad(true);
        NN_CHECK(grad_error(x, [&] { return nn::autograd::sum(x, axis, keepdim); }) < 2e-2f);
      }
    }

    nn::Tensor xm = nn::Tensor::randn({4, 5}, rng, 0.7f, dev);
    xm.set_requires_grad(true);
    NN_CHECK(grad_error(xm, [&] { return nn::autograd::mean(xm, 1); }) < 2e-2f);

    nn::Tensor xv = nn::Tensor::randn({4, 5}, rng, 0.7f, dev);
    xv.set_requires_grad(true);
    NN_CHECK(grad_error(xv, [&] { return nn::autograd::var(xv, 1); }) < 2e-2f);

    nn::Tensor xa = nn::Tensor::randn({4, 5}, rng, 0.7f, dev);
    xa.set_requires_grad(true);
    NN_CHECK(grad_error(xa, [&] { return nn::autograd::mean(xa); }) < 2e-2f);
  }
}

NN_TEST(softmax_rows_are_a_distribution) {
  const int M = 5, N = 7;

  NN_TEST_FOR_EACH_DEVICE(dev) {
    nn::Pcg32 rng(11);
    const nn::Tensor x = nn::Tensor::randn({M, N}, rng, 2.0f, dev);
    const nn::Tensor p = nn::ops::softmax_rows(x);
    NN_CHECK(p.shape() == x.shape());

    const std::vector<float> hp = host_of(p);
    const std::vector<float> hx = host_of(x);
    for (int i = 0; i < M; ++i) {
      float sum = 0.0f;
      for (int j = 0; j < N; ++j) {
        NN_CHECK(hp[size_t(i) * N + j] > 0.0f);
        sum += hp[size_t(i) * N + j];
      }
      NN_CHECK_CLOSE(sum, 1.0f, 1e-6);

      // against a host softmax of the same row
      float m = hx[size_t(i) * N];
      for (int j = 1; j < N; ++j) m = std::fmax(m, hx[size_t(i) * N + j]);
      float denom = 0.0f;
      for (int j = 0; j < N; ++j) denom += std::exp(hx[size_t(i) * N + j] - m);
      for (int j = 0; j < N; ++j) {
        NN_CHECK_CLOSE(hp[size_t(i) * N + j],
                       std::exp(hx[size_t(i) * N + j] - m) / denom, 1e-5);
      }
    }
  }
}

NN_TEST(softmax_is_shift_invariant_and_stable) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    const nn::Tensor big = nn::Tensor::from({{1000.0f, 1000.0f, 1000.0f}}, dev);
    const std::vector<float> p = host_of(nn::ops::softmax_rows(big));
    for (float v : p) NN_CHECK_CLOSE(v, 1.0f / 3.0f, 1e-6);

    const nn::Tensor small = nn::Tensor::from({{-1000.0f, -1000.0f, -999.0f}}, dev);
    const std::vector<float> q = host_of(nn::ops::softmax_rows(small));
    float sum = 0.0f;
    for (float v : q) { NN_CHECK(std::isfinite(v)); sum += v; }
    NN_CHECK_CLOSE(sum, 1.0f, 1e-6);
    NN_CHECK(q[2] > q[0]);
  }
}

// Softmax is over the last axis whatever the rank, and a row stride is passed
// down rather than materialised.
NN_TEST(softmax_handles_rank3_and_row_strides) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    nn::Pcg32 rng(13);
    const nn::Tensor x = nn::Tensor::randn({2, 3, 4}, rng, 1.0f, dev);
    const std::vector<float> p = host_of(nn::ops::softmax_rows(x));
    for (int r = 0; r < 6; ++r) {
      float sum = 0.0f;
      for (int c = 0; c < 4; ++c) sum += p[size_t(r) * 4 + c];
      NN_CHECK_CLOSE(sum, 1.0f, 1e-6);
    }

    const nn::Tensor wide = nn::Tensor::randn({5, 9}, rng, 1.0f, dev);
    const nn::Tensor view = wide.slice_view(1, 3, 4);
    NN_CHECK(!view.is_contiguous());
    const std::vector<float> a = host_of(nn::ops::softmax_rows(view));
    const std::vector<float> b = host_of(nn::ops::softmax_rows(view.pack()));
    for (size_t i = 0; i < a.size(); ++i) NN_CHECK_CLOSE(a[i], b[i], 1e-6);
  }
}

NN_TEST(gradcheck_softmax) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    nn::Pcg32 rng(17);
    nn::Tensor x = nn::Tensor::randn({4, 6}, rng, 1.0f, dev);
    x.set_requires_grad(true);
    NN_CHECK(grad_error(x, [&] { return nn::autograd::softmax(x); }) < 2e-2f);
  }
}

NN_TEST(embedding_gathers_rows) {
  const int V = 6, D = 3;

  NN_TEST_FOR_EACH_DEVICE(dev) {
    const nn::Tensor w = ramp(nn::Shape({V, D}), dev, 0.0f, 1.0f);
    const nn::Tensor idx = nn::Tensor::from_i32({4, 0, 2, 2}, dev);

    const nn::Tensor out = nn::ops::embedding(w, idx);
    NN_CHECK(out.shape() == nn::Shape({4, D}));

    const std::vector<float> hw = host_of(w);
    const std::vector<float> ho = host_of(out);
    const int rows[] = {4, 0, 2, 2};
    for (int i = 0; i < 4; ++i)
      for (int d = 0; d < D; ++d)
        NN_CHECK(ho[size_t(i) * D + d] == hw[size_t(rows[i]) * D + d]);

    // A rank-2 index tensor gets the feature axis appended, not flattened away.
    const nn::Tensor idx2 = nn::Tensor::from_i32(std::vector<int32_t>{1, 3, 5, 0, 1, 2},
                                                 nn::Shape({2, 3}), dev);
    NN_CHECK(nn::ops::embedding(w, idx2).shape() == nn::Shape({2, 3, D}));
  }
}

NN_TEST(embedding_backward_accumulates_repeats) {
  const int V = 5, D = 2;

  NN_TEST_FOR_EACH_DEVICE(dev) {
    // index 1 appears three times, so its row collects three gradients
    const nn::Tensor idx = nn::Tensor::from_i32({1, 3, 1, 1}, dev);
    const nn::Tensor g = ramp(nn::Shape({4, D}), dev, 1.0f, 1.0f);

    const nn::Tensor gw = nn::ops::embedding_backward(g, idx, V);
    NN_CHECK(gw.shape() == nn::Shape({V, D}));

    const std::vector<float> hg = host_of(g);
    const std::vector<float> hw = host_of(gw);

    for (int d = 0; d < D; ++d) {
      const float want = hg[size_t(0) * D + d] + hg[size_t(2) * D + d] + hg[size_t(3) * D + d];
      NN_CHECK_CLOSE(hw[size_t(1) * D + d], want, 1e-6);
      NN_CHECK_CLOSE(hw[size_t(3) * D + d], hg[size_t(1) * D + d], 1e-6);
    }
    // Rows nothing pointed at stay zero.
    for (int v : {0, 2, 4})
      for (int d = 0; d < D; ++d) NN_CHECK(hw[size_t(v) * D + d] == 0.0f);
  }
}

// A bad index is a wrong number, not a wild read or write.
NN_TEST(embedding_skips_out_of_range_indices) {
  const int V = 4, D = 2;

  NN_TEST_FOR_EACH_DEVICE(dev) {
    const nn::Tensor w = ramp(nn::Shape({V, D}), dev, 1.0f, 1.0f);
    const nn::Tensor idx = nn::Tensor::from_i32({-1, 2, 99}, dev);

    const std::vector<float> out = host_of(nn::ops::embedding(w, idx));
    NN_CHECK(out[0] == 0.0f && out[1] == 0.0f);
    NN_CHECK(out[4] == 0.0f && out[5] == 0.0f);
    const std::vector<float> hw = host_of(w);
    NN_CHECK(out[2] == hw[4] && out[3] == hw[5]);

    const nn::Tensor gw = nn::ops::embedding_backward(ramp(nn::Shape({3, D}), dev), idx, V);
    const std::vector<float> hgw = host_of(gw);
    for (int v : {0, 1, 3})
      for (int d = 0; d < D; ++d) NN_CHECK(hgw[size_t(v) * D + d] == 0.0f);
  }
}

NN_TEST(gradcheck_embedding) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    nn::Pcg32 rng(19);
    nn::Tensor w = nn::Tensor::randn({6, 4}, rng, 0.5f, dev);
    w.set_requires_grad(true);
    const nn::Tensor idx = nn::Tensor::from_i32({3, 0, 3, 5, 1}, dev);

    NN_CHECK(grad_error(w, [&] { return nn::autograd::embedding(w, idx); }, 20) < 2e-2f);
  }
}

NN_TEST(layer_norm_normalises_the_last_axis) {
  const int R = 4, N = 8;

  NN_TEST_FOR_EACH_DEVICE(dev) {
    nn::Pcg32 rng(23);
    const nn::Tensor x = nn::Tensor::randn({R, N}, rng, 3.0f, dev);
    const nn::Tensor w = nn::Tensor::full({N}, 1.0f, dev);
    const nn::Tensor b = nn::Tensor::zeros({N}, dev);

    const std::vector<float> y = host_of(nn::autograd::layer_norm(x, w, b));
    for (int r = 0; r < R; ++r) {
      float mean = 0.0f;
      for (int c = 0; c < N; ++c) mean += y[size_t(r) * N + c];
      mean /= float(N);
      float ss = 0.0f;
      for (int c = 0; c < N; ++c) {
        const float d = y[size_t(r) * N + c] - mean;
        ss += d * d;
      }
      NN_CHECK_CLOSE(mean, 0.0f, 1e-5);
      NN_CHECK_CLOSE(ss / float(N), 1.0f, 1e-4);   // biased variance, as scaled
    }

    // weight and bias are applied per feature, after the normalisation
    const nn::Tensor w2 = nn::Tensor::full({N}, 2.0f, dev);
    const nn::Tensor b2 = nn::Tensor::full({N}, 5.0f, dev);
    const std::vector<float> y2 = host_of(nn::autograd::layer_norm(x, w2, b2));
    for (size_t i = 0; i < y.size(); ++i) NN_CHECK_CLOSE(y2[i], 2.0f * y[i] + 5.0f, 1e-4);
  }
}

NN_TEST(gradcheck_layer_norm) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    nn::Pcg32 rng(29);
    nn::Tensor x = nn::Tensor::randn({3, 5}, rng, 1.0f, dev);
    nn::Tensor w = nn::Tensor::randn({5}, rng, 0.5f, dev);
    nn::Tensor b = nn::Tensor::randn({5}, rng, 0.5f, dev);
    x.set_requires_grad(true);
    w.set_requires_grad(true);
    b.set_requires_grad(true);

    auto forward = [&] { return nn::autograd::layer_norm(x, w, b); };
    NN_CHECK(grad_error(x, forward) < 3e-2f);
    NN_CHECK(grad_error(w, forward) < 2e-2f);
    NN_CHECK(grad_error(b, forward) < 2e-2f);
  }
}

NN_TEST(masked_fill_replaces_and_blocks_the_gradient) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    const nn::Tensor x = ramp(nn::Shape({2, 3}), dev, 1.0f, 1.0f);
    const nn::Tensor mask = nn::Tensor::from({{0.0f, 1.0f, 0.0f}, {1.0f, 0.0f, 1.0f}}, dev);

    const std::vector<float> y = host_of(nn::autograd::masked_fill(x, mask, -9.0f));
    const std::vector<float> hx = host_of(x);
    const std::vector<float> hm = host_of(mask);
    for (size_t i = 0; i < y.size(); ++i) {
      NN_CHECK_CLOSE(y[i], (hm[i] != 0.0f) ? -9.0f : hx[i], 1e-6);
    }

    // The gradient has to stop at a masked position: those outputs no longer
    // depend on x at all.
    nn::Tensor p = ramp(nn::Shape({2, 3}), dev, 1.0f, 1.0f);
    p.set_requires_grad(true);
    nn::autograd::Tape tape;
    nn::Tensor loss;
    {
      nn::autograd::TapeScope scope(tape);
      loss = nn::autograd::sum_all(nn::autograd::masked_fill(p, mask, -9.0f));
    }
    tape.backward(loss);

    const std::vector<float> g = host_of(p.grad());
    for (size_t i = 0; i < g.size(); ++i) {
      NN_CHECK_CLOSE(g[i], (hm[i] != 0.0f) ? 0.0f : 1.0f, 1e-6);
    }
  }
}

// A mask built with the comparison ops, which is how a causal mask is written.
NN_TEST(masked_fill_composes_with_a_predicate) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    const nn::Tensor scores = nn::Tensor::from({{1.0f, 2.0f, 3.0f}}, dev);
    const nn::Tensor position = nn::Tensor::from({{0.0f, 1.0f, 2.0f}}, dev);

    // everything strictly after index 0 is masked out
    const nn::Tensor mask = nn::ops::scalar(nn::kernels::ScalarOp::GtScalar, position, 0.5f);
    const nn::Tensor filled = nn::autograd::masked_fill(scores, mask, -1e9f);
    const std::vector<float> p = host_of(nn::ops::softmax_rows(filled));

    NN_CHECK_CLOSE(p[0], 1.0f, 1e-6);
    NN_CHECK_CLOSE(p[1], 0.0f, 1e-6);
    NN_CHECK_CLOSE(p[2], 0.0f, 1e-6);
  }
}

// An I32 view has to be read from its own offset, not from the base of its
// storage -- the TensorView handed to the kernel carries no offset of its own,
// exactly as for float. Indices are the only I32 tensors in the library, so
// embedding is where a slice of one actually shows up.
NN_TEST(i32_views_read_from_their_own_offset) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    const nn::Tensor all = nn::Tensor::from_i32({0, 1, 2, 3, 4, 5}, dev);
    const nn::Tensor tail = all.slice_view(0, 2, 3);          // {2, 3, 4}

    const nn::Tensor packed = tail.pack().to(nn::Device::CPU);
    NN_CHECK(packed.host_data_i32()[0] == 2);
    NN_CHECK(packed.host_data_i32()[1] == 3);
    NN_CHECK(packed.host_data_i32()[2] == 4);

    const nn::Tensor w = ramp(nn::Shape({6, 2}), dev, 0.0f, 1.0f);
    const std::vector<float> got = host_of(nn::ops::embedding(w, tail));
    const std::vector<float> hw = host_of(w);
    for (int i = 0; i < 3; ++i)
      for (int d = 0; d < 2; ++d)
        NN_CHECK(got[size_t(i) * 2 + d] == hw[size_t(i + 2) * 2 + d]);
  }
}

// The float half of the same bug: to() and clone() read through raw(), and a
// slice along the outermost axis is still contiguous, so nothing else would
// have caught an offset being dropped there.
NN_TEST(contiguous_views_with_an_offset_copy_from_their_own_start) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    const nn::Tensor all = ramp(nn::Shape({6, 2}), dev, 0.0f, 1.0f);
    const nn::Tensor tail = all.slice_view(0, 3, 2);     // rows 3 and 4
    NN_CHECK(tail.is_contiguous());
    NN_CHECK(tail.offset() == 6);

    const std::vector<float> viaTo = host_of(tail.to(nn::Device::CPU));
    const std::vector<float> viaClone = host_of(tail.clone());
    for (int i = 0; i < 4; ++i) {
      NN_CHECK(viaTo[size_t(i)] == 6.0f + float(i));
      NN_CHECK(viaClone[size_t(i)] == 6.0f + float(i));
    }
  }
}

NN_TEST(cat_joins_and_splits_back_apart) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    const nn::Tensor a = ramp(nn::Shape({2, 3}), dev, 0.0f, 1.0f);   // 0..5
    const nn::Tensor b = ramp(nn::Shape({2, 2}), dev, 100.0f, 1.0f); // 100..103
    const nn::Tensor c = ramp(nn::Shape({2, 1}), dev, 200.0f, 1.0f); // 200..201

    const nn::Tensor joined = nn::cat({a, b, c}, 1);
    NN_CHECK(joined.shape() == nn::Shape({2, 6}));

    const std::vector<float> h = host_of(joined);
    const std::vector<float> want{0, 1, 2, 100, 101, 200,
                                  3, 4, 5, 102, 103, 201};
    for (size_t i = 0; i < want.size(); ++i) NN_CHECK(h[i] == want[i]);

    // slicing the pieces back out is the inverse
    const std::vector<float> back = host_of(joined.slice_view(1, 3, 2));
    const std::vector<float> hb = host_of(b);
    for (size_t i = 0; i < hb.size(); ++i) NN_CHECK(back[i] == hb[i]);

    // along the outermost axis too
    const nn::Tensor stacked = nn::cat({a, a}, 0);
    NN_CHECK(stacked.shape() == nn::Shape({4, 3}));

    NN_CHECK_THROWS(nn::cat({a, b}, 0), std::invalid_argument);   // differ off-axis
    NN_CHECK_THROWS(nn::cat({}, 0), std::invalid_argument);
  }
}

// Each operand gets exactly the window of the gradient it contributed, and an
// operand that appears twice collects both.
NN_TEST(gradcheck_cat) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    nn::Pcg32 rng(53);
    nn::Tensor a = nn::Tensor::randn({3, 2}, rng, 0.6f, dev);
    nn::Tensor b = nn::Tensor::randn({3, 4}, rng, 0.6f, dev);
    a.set_requires_grad(true);
    b.set_requires_grad(true);

    auto forward = [&] { return nn::cat({a, b}, 1); };
    NN_CHECK(grad_error(a, forward) < 2e-2f);
    NN_CHECK(grad_error(b, forward) < 2e-2f);

    nn::Tensor t = nn::Tensor::randn({2, 3}, rng, 0.6f, dev);
    t.set_requires_grad(true);
    NN_CHECK(grad_error(t, [&] { return nn::cat({t, t}, 0); }) < 2e-2f);
  }
}

NN_TEST(test_topk) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    nn::Pcg32 rng(61);
    const int M = 4, N = 7, K = 3;
    const nn::Tensor x = nn::Tensor::randn({M, N}, rng, 1.0f, dev);

    nn::Tensor values, indices;
    nn::ops::topk_rows(x, K, values, indices);

    NN_CHECK(values.shape() == nn::Shape({M, K}));
    NN_CHECK(indices.shape() == nn::Shape({M, K}));

    const std::vector<float> hx = host_of(x);
    const std::vector<float> hv = host_of(values);
    const std::vector<int32_t> hi = host_of_i32(indices);

    for (int i = 0; i < M; ++i) {
      std::vector<std::pair<float, int>> row;
      for (int j = 0; j < N; ++j) row.emplace_back(hx[size_t(i) * N + j], j);
      std::sort(row.begin(), row.end(), std::greater<>());
      for (int k = 0; k < K; ++k) {
        NN_CHECK_CLOSE(hv[size_t(i) * K + k], row[k].first, 1e-5);
        NN_CHECK(hi[size_t(i) * K + k] == row[k].second);
      }
    }
  }
}

// multinomial takes any nonnegative row, not just a normalised distribution --
// topk_rows' raw values feed it directly, no softmax renormalisation needed.
NN_TEST(multinomial_never_draws_a_zero_weight_column) {
  const int M = 3, N = 5;

  NN_TEST_FOR_EACH_DEVICE(dev) {
    std::vector<float> host(size_t(M) * N, 0.0f);
    for (int i = 0; i < M; ++i) {
      host[size_t(i) * N + i % N] = 1.0f;
      host[size_t(i) * N + (i + 2) % N] = 3.0f;   // unnormalised: doesn't sum to 1
    }
    const nn::Tensor weights = nn::Tensor::from(host, nn::Shape({M, N}), dev);

    nn::Pcg32 rng(71);
    for (int trial = 0; trial < 50; ++trial) {
      const std::vector<int32_t> got = host_of_i32(nn::ops::multinomial(weights, rng));
      NN_CHECK(int(got.size()) == M);
      for (int i = 0; i < M; ++i) {
        const int j = got[size_t(i)];
        NN_CHECK(j == i % N || j == (i + 2) % N);
      }
    }
  }
}

NN_TEST(multinomial_rejects_bad_input) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    nn::Pcg32 rng(72);
    const nn::Tensor rank1 = nn::Tensor::from({1.0f, 2.0f, 3.0f}, dev);
    NN_CHECK_THROWS(nn::ops::multinomial(rank1, rng), std::invalid_argument);

    const nn::Tensor all_zero = nn::Tensor::from({{0.0f, 0.0f, 0.0f}}, dev);
    NN_CHECK_THROWS(nn::ops::multinomial(all_zero, rng), std::invalid_argument);
  }
}

// The motivating case: turning multinomial's pick (an index into topk_rows'
// [M, K] values) back into topk_rows' matching [M, K] indices -- the actual
// vocab id sampled.
NN_TEST(gather_rows_maps_a_local_pick_back_through_topk_indices) {
  const int M = 4, N = 6, K = 3;

  NN_TEST_FOR_EACH_DEVICE(dev) {
    nn::Pcg32 rng(81);
    const nn::Tensor logits = nn::Tensor::randn({M, N}, rng, 1.0f, dev);
    const nn::Tensor probs = nn::ops::softmax_rows(logits);   // multinomial needs nonnegative weights

    nn::Tensor values, indices;
    nn::ops::topk_rows(probs, K, values, indices);

    const nn::Tensor local = nn::ops::multinomial(values, rng);   // [M], in [0, K)
    const nn::Tensor vocab_id = nn::ops::gather_rows(indices, local);
    NN_CHECK(vocab_id.shape() == nn::Shape({M}));
    NN_CHECK(vocab_id.dtype() == nn::DType::I32);

    const std::vector<int32_t> hl = host_of_i32(local);
    const std::vector<int32_t> hidx = host_of_i32(indices);
    const std::vector<int32_t> got = host_of_i32(vocab_id);
    for (int i = 0; i < M; ++i) {
      NN_CHECK(got[size_t(i)] == hidx[size_t(i) * K + size_t(hl[size_t(i)])]);
    }
  }
}

NN_TEST(gather_rows_gathers_float_rows_and_rejects_bad_input) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    const nn::Tensor src = nn::Tensor::from({{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}}, dev);
    const nn::Tensor idx = nn::Tensor::from_i32({2, 0}, dev);

    const std::vector<float> got = host_of(nn::ops::gather_rows(src, idx));
    NN_CHECK_CLOSE(got[0], 3.0f, 0.0f);
    NN_CHECK_CLOSE(got[1], 4.0f, 0.0f);

    NN_CHECK_THROWS(nn::ops::gather_rows(src, nn::Tensor::from_i32({0, 1, 2}, dev)),
                    std::invalid_argument);
    NN_CHECK_THROWS(nn::ops::gather_rows(src, nn::Tensor::from_i32({0, 5}, dev)),
                    std::invalid_argument);
  }
}
