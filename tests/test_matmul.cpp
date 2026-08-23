#include "test_harness.h"
#include "devices.h"
#include "gradcheck.h"

#include <cmath>
#include <functional>
#include <stdexcept>
#include <vector>

#include <nn/autograd/functions.h>
#include <nn/autograd/tape.h>
#include <nn/core/rng.h>
#include <nn/core/tensor.h>
#include <nn/nn/module.h>
#include <nn/ops/ops.h>

namespace {

std::vector<float> host_of(const nn::Tensor& t) {
  const nn::Tensor h = t.pack().to(nn::Device::CPU);
  return std::vector<float>(h.host_data(), h.host_data() + h.numel());
}

nn::Tensor spread(nn::Shape s, nn::Device d, int seed) {
  std::vector<float> v(size_t(s.numel()));
  for (size_t i = 0; i < v.size(); ++i) {
    v[i] = std::sin(0.7f * float(i) + float(seed)) * 1.3f + 0.2f;
  }
  return nn::Tensor::from(v, s, d);
}

// Matrix i out of a contiguous [batch..., r, c] tensor.
nn::Tensor matrix_at(const nn::Tensor& t, int64_t i) {
  const int r = t.shape().rank();
  const int rows = t.shape().dim(r - 2), cols = t.shape().dim(r - 1);
  return t.pack()
          .reshape_view(nn::Shape({int(t.numel() / (int64_t(rows) * cols)), rows, cols}))
          .slice_view(0, i, 1)
          .reshape_view(nn::Shape({rows, cols}));
}

// Everything but an operand's last two axes, as a list of extents.
std::vector<int> batch_dims(const nn::Tensor& t) {
  std::vector<int> d;
  for (int i = 0; i < t.shape().rank() - 2; ++i) d.push_back(t.shape().dim(i));
  return d;
}

// Which of an operand's matrices batch element `i` of the output uses. Aligned
// from the right like every other broadcast, and an axis of extent 1 stays at
// index 0 however far the output has walked along it.
int64_t operand_matrix_index(int64_t i, const std::vector<int>& out_dims,
                             const std::vector<int>& own_dims) {
  const size_t lead = out_dims.size() - own_dims.size();
  int64_t rem = i, own = 0, place = 1;
  for (size_t k = out_dims.size(); k-- > 0;) {
    const int64_t idx = rem % out_dims[k];
    rem /= out_dims[k];
    if (k < lead) continue;
    const int own_extent = own_dims[k - lead];
    own += (own_extent == 1 ? 0 : idx) * place;
    place *= own_extent;
  }
  return own;
}

// The reference every case below is checked against: one plain 2-D matmul per
// batch element, each operand contributing whichever matrix broadcasting says.
void check_against_loop(const nn::Tensor& a, const nn::Tensor& b,
                        bool transA, bool transB, const char* what) {
  const nn::Tensor got = nn::ops::matmul(a, b, transA, transB);

  const int gr = got.shape().rank();
  const int64_t per = int64_t(got.shape().dim(gr - 2)) * got.shape().dim(gr - 1);
  const int64_t nb = got.numel() / per;

  const std::vector<int> out_dims = batch_dims(got);
  const std::vector<int> a_dims = batch_dims(a), b_dims = batch_dims(b);
  const std::vector<float> flat = host_of(got);

  for (int64_t i = 0; i < nb; ++i) {
    const nn::Tensor want = nn::ops::matmul(
        matrix_at(a, operand_matrix_index(i, out_dims, a_dims)),
        matrix_at(b, operand_matrix_index(i, out_dims, b_dims)),
        transA, transB);
    const std::vector<float> w = host_of(want);
    for (int64_t j = 0; j < per; ++j) {
      const float got_v = flat[size_t(i * per + j)], want_v = w[size_t(j)];
      if (std::fabs(got_v - want_v) > 1e-4f * std::fmax(1.0f, std::fabs(want_v))) {
        nn::test::report(__FILE__, __LINE__, std::string(what) + ": matrix " +
            std::to_string(i) + " element " + std::to_string(j) + ": " +
            std::to_string(got_v) + " vs " + std::to_string(want_v));
        return;
      }
    }
  }
}

nn::Tensor weights_like(const nn::Tensor& y) {
  std::vector<float> w(size_t(y.numel()));
  for (size_t i = 0; i < w.size(); ++i) w[i] = 0.6f + 0.8f * std::sin(1.3f * float(i));
  return nn::Tensor::from(w, y.shape(), y.device());
}

// h is deliberately larger than the 1e-3 default. A matmul is linear in each
// operand, so a central difference is exact for any step and a bigger one costs
// nothing in truncation error -- while at 1e-3 a gradient component of 4e-4
// moves the loss by 8e-7 against a loss of order 1, which is about three bits
// above float epsilon. At that point the two backends disagree simply because
// they sum in a different order.
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
  return nn::test::gradCheck(param, loss_fn, backward_fn, checks, /*h=*/1e-2f);
}

}  // namespace

NN_TEST(batched_matmul_matches_a_loop) {
  const int B = 3, H = 2, M = 5, K = 4, N = 6;

  NN_TEST_FOR_EACH_DEVICE(dev) {
    const nn::Tensor a = spread(nn::Shape({B, H, M, K}), dev, 1);
    const nn::Tensor b = spread(nn::Shape({B, H, K, N}), dev, 2);

    const nn::Tensor out = nn::ops::matmul(a, b);
    NN_CHECK(out.shape() == nn::Shape({B, H, M, N}));
    check_against_loop(a, b, false, false, "rank 4");

    // rank 3, and a batch of exactly one
    check_against_loop(spread(nn::Shape({4, M, K}), dev, 3),
                       spread(nn::Shape({4, K, N}), dev, 4), false, false, "rank 3");
    check_against_loop(spread(nn::Shape({1, M, K}), dev, 5),
                       spread(nn::Shape({1, K, N}), dev, 6), false, false, "batch 1");
  }
}

NN_TEST(batched_matmul_handles_every_transpose) {
  const int B = 3, M = 5, K = 4, N = 6;

  NN_TEST_FOR_EACH_DEVICE(dev) {
    for (int ta = 0; ta < 2; ++ta) {
      for (int tb = 0; tb < 2; ++tb) {
        const nn::Tensor a = ta ? spread(nn::Shape({B, K, M}), dev, 7)
                                : spread(nn::Shape({B, M, K}), dev, 7);
        const nn::Tensor b = tb ? spread(nn::Shape({B, N, K}), dev, 8)
                                : spread(nn::Shape({B, K, N}), dev, 8);
        const std::string what = std::string("transA=") + std::to_string(ta) +
                                 " transB=" + std::to_string(tb);
        check_against_loop(a, b, ta != 0, tb != 0, what.c_str());
      }
    }
  }
}

// A weight with no batch axes of its own reaches every matrix through a stride
// of zero, so nothing is copied to broadcast it.
NN_TEST(matmul_broadcasts_batch_axes) {
  const int B = 3, H = 2, T = 5, d = 4, K = 6;

  NN_TEST_FOR_EACH_DEVICE(dev) {
    // [B,H,T,d] x [d,K] -- this one takes the fold path, not the batched one
    check_against_loop(spread(nn::Shape({B, H, T, d}), dev, 9),
                       spread(nn::Shape({d, K}), dev, 10), false, false, "fold");

    // [T,d] x [B,H,d,K] -- only B has batch axes, so A is broadcast
    check_against_loop(spread(nn::Shape({T, d}), dev, 11),
                       spread(nn::Shape({B, H, d, K}), dev, 12), false, false, "A shared");

    // [B,1,T,d] x [B,H,d,K] -- a size-1 axis stretched against a real one
    check_against_loop(spread(nn::Shape({B, 1, T, d}), dev, 13),
                       spread(nn::Shape({B, H, d, K}), dev, 14), false, false, "stretch");

    NN_CHECK_THROWS(nn::ops::matmul(spread(nn::Shape({2, T, d}), dev, 1),
                                    spread(nn::Shape({3, d, K}), dev, 2)),
                    std::invalid_argument);
  }
}

// The fold path has to produce the same numbers as reshaping by hand, and it
// has to be what actually runs -- a [B,T,C] x [C,K] is one GEMM, not B*T.
NN_TEST(matmul_folds_a_batched_operand_against_a_plain_weight) {
  const int B = 4, T = 7, C = 5, K = 3;

  NN_TEST_FOR_EACH_DEVICE(dev) {
    const nn::Tensor x = spread(nn::Shape({B, T, C}), dev, 15);
    const nn::Tensor w = spread(nn::Shape({C, K}), dev, 16);

    const nn::Tensor got = nn::ops::matmul(x, w);
    NN_CHECK(got.shape() == nn::Shape({B, T, K}));

    const nn::Tensor by_hand =
        nn::ops::matmul(x.reshape_view(nn::Shape({B * T, C})), w).reshape_view(nn::Shape({B, T, K}));

    const std::vector<float> a = host_of(got), b = host_of(by_hand);
    for (size_t i = 0; i < a.size(); ++i) NN_CHECK_CLOSE(a[i], b[i], 1e-6);
  }
}

// The attention layout: [B,T,C] -> [B,T,H,hd] -> permute -> [B,H,T,hd]. The two
// batch axes are then not evenly spaced, so a single batch stride cannot reach
// the matrices and matmul says so rather than reading the wrong memory.
NN_TEST(matmul_rejects_a_batch_it_cannot_address) {
  const int B = 2, T = 6, H = 4, hd = 3;

  NN_TEST_FOR_EACH_DEVICE(dev) {
    const nn::Tensor x = spread(nn::Shape({B, T, H * hd}), dev, 17);
    const int order[4] = {0, 2, 1, 3};
    const nn::Tensor q = x.reshape_view(nn::Shape({B, T, H, hd}))
                          .permute_view(std::span<const int>(order, 4));
    NN_CHECK(q.shape() == nn::Shape({B, H, T, hd}));
    NN_CHECK(q.stride(0) != H * q.stride(1));    // the two batch axes do not merge

    const nn::Tensor kt = q.transpose_view(2, 3);     // [B,H,hd,T]
    NN_CHECK_THROWS(nn::ops::matmul(q, kt), std::invalid_argument);

    // One pack fixes it, and then the answer matches the loop.
    const nn::Tensor qc = q.pack();
    check_against_loop(qc, qc.transpose_view(2, 3).pack(), false, false, "packed");
  }
}

// gridDim.z is capped at 65535, so a batch past that has to loop rather than
// silently compute only the first 65535 matrices.
NN_TEST(batched_matmul_survives_a_batch_larger_than_the_z_grid) {
  const int64_t batch = 66000;   // just past the cap, so a few blocks wrap twice
  const int M = 2, K = 2, N = 2;

  NN_TEST_FOR_EACH_DEVICE(dev) {
    const nn::Tensor a = spread(nn::Shape({int(batch), M, K}), dev, 18);
    const nn::Tensor b = spread(nn::Shape({int(batch), K, N}), dev, 19);
    const nn::Tensor got = nn::ops::matmul(a, b);
    NN_CHECK(got.numel() == batch * M * N);

    // spot-check the first, a middle and the last matrix
    const std::vector<float> flat = host_of(got);
    for (int64_t i : {int64_t(0), batch / 2, batch - 1}) {
      const std::vector<float> want = host_of(nn::ops::matmul(matrix_at(a, i), matrix_at(b, i)));
      for (size_t j = 0; j < want.size(); ++j) {
        NN_CHECK_CLOSE(flat[size_t(i * M * N) + j], want[j], 1e-5);
      }
    }
  }
}

NN_TEST(linear_accepts_a_batched_input) {
  const int B = 3, T = 5, C = 6, K = 4;

  NN_TEST_FOR_EACH_DEVICE(dev) {
    nn::Pcg32 rng(2);
    nn::Linear fc(C, K, rng);
    fc.to(dev);

    const nn::Tensor x = spread(nn::Shape({B, T, C}), dev, 20);
    const nn::Tensor y = fc(x);
    NN_CHECK(y.shape() == nn::Shape({B, T, K}));

    // the same weights applied to the flattened batch
    const nn::Tensor flat = fc(x.reshape_view(nn::Shape({B * T, C})));
    const std::vector<float> a = host_of(y), b = host_of(flat);
    for (size_t i = 0; i < a.size(); ++i) NN_CHECK_CLOSE(a[i], b[i], 1e-6);
  }
}

NN_TEST(gradcheck_batched_matmul) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    nn::Pcg32 rng(3);

    {   // both operands batched
      nn::Tensor a = nn::Tensor::randn({2, 3, 4}, rng, 0.6f, dev);
      nn::Tensor b = nn::Tensor::randn({2, 4, 3}, rng, 0.6f, dev);
      a.set_requires_grad(true);
      b.set_requires_grad(true);
      auto f = [&] { return nn::autograd::matmul(a, b); };
      NN_CHECK(grad_error(a, f) < 2e-2f);
      NN_CHECK(grad_error(b, f) < 2e-2f);
    }
    {   // the fold path: a batched activation against a plain weight
      nn::Tensor x = nn::Tensor::randn({2, 5, 4}, rng, 0.6f, dev);
      nn::Tensor w = nn::Tensor::randn({4, 3}, rng, 0.6f, dev);
      x.set_requires_grad(true);
      w.set_requires_grad(true);
      auto f = [&] { return nn::autograd::matmul(x, w); };
      NN_CHECK(grad_error(x, f) < 2e-2f);
      NN_CHECK(grad_error(w, f) < 2e-2f);   // the hand-folded weight gradient
    }
    {   // a broadcast batch axis, so the gradient has to be summed back down
      nn::Tensor a = nn::Tensor::randn({1, 5, 4}, rng, 0.6f, dev);
      nn::Tensor b = nn::Tensor::randn({3, 4, 2}, rng, 0.6f, dev);
      a.set_requires_grad(true);
      b.set_requires_grad(true);
      auto f = [&] { return nn::autograd::matmul(a, b); };
      NN_CHECK(grad_error(a, f) < 2e-2f);
      NN_CHECK(grad_error(b, f) < 2e-2f);
    }
  }
}

NN_TEST(gradcheck_attention) {
  const int B = 2, T = 4, H = 2, hd = 3, C = H * hd;

  NN_TEST_FOR_EACH_DEVICE(dev) {
    nn::Pcg32 rng(5);
    nn::Tensor x = nn::Tensor::randn({B, T, C}, rng, 0.5f, dev);
    nn::Tensor wq = nn::Tensor::randn({C, C}, rng, 0.5f, dev);
    nn::Tensor wk = nn::Tensor::randn({C, C}, rng, 0.5f, dev);
    nn::Tensor wv = nn::Tensor::randn({C, C}, rng, 0.5f, dev);
    x.set_requires_grad(true);
    wq.set_requires_grad(true);
    wk.set_requires_grad(true);
    wv.set_requires_grad(true);

    auto heads = [&](const nn::Tensor& t) {
      return t.reshape(nn::Shape({B, T, H, hd})).permute({0, 2, 1, 3}).contiguous();
    };

    auto attention = [&]() {
      const nn::Tensor q = heads(nn::autograd::matmul(x, wq));   // [B,H,T,hd]
      const nn::Tensor k = heads(nn::autograd::matmul(x, wk));
      const nn::Tensor v = heads(nn::autograd::matmul(x, wv));

      const nn::Tensor scores =
          nn::autograd::matmul(q, k.t().contiguous()) *
          (1.0f / std::sqrt(float(hd)));
      return nn::autograd::matmul(scores.softmax(), v);          // [B,H,T,hd]
    };

    NN_CHECK(attention().shape() == nn::Shape({B, H, T, hd}));
    NN_CHECK(grad_error(wq, attention) < 3e-2f);
    NN_CHECK(grad_error(wv, attention) < 3e-2f);
    NN_CHECK(grad_error(x, attention) < 3e-2f);
  }
}
