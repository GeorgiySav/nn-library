#include "test_harness.h"
#include "devices.h"

#include <stdexcept>
#include <vector>

#include <nn/core/rng.h>
#include <nn/core/tensor.h>
#include <nn/autograd/functions.h>
#include <nn/autograd/tape.h>
#include <nn/ops/ops.h>

namespace {

// Compares two tensors element by element on the host. Reports through the
// harness with the caller's line, so a failure points at the assertion rather
// than at this helper.
void check_same(const nn::Tensor& got, const nn::Tensor& want, float tol,
                const char* what, const char* file, int line) {
  if (got.shape() != want.shape()) {
    nn::test::report(file, line, std::string(what) + ": shape mismatch");
    return;
  }
  const nn::Tensor g = got.contiguous().to(nn::Device::CPU);
  const nn::Tensor w = want.contiguous().to(nn::Device::CPU);
  const int64_t n = g.numel();
  for (int64_t i = 0; i < n; ++i) {
    const float a = g.host_data()[i], b = w.host_data()[i];
    const float d = std::fabs(a - b);
    const float scale = std::fmax(1e-8f, std::fmax(std::fabs(a), std::fabs(b)));
    if (d > tol && d / scale > tol) {
      nn::test::report(file, line, std::string(what) + ": element " +
          std::to_string(i) + ": " + std::to_string(a) + " vs " + std::to_string(b));
      return;   // one report per call, not one per element
    }
  }
}

}  // namespace

#define NN_CHECK_SAME(got, want, tol) \
  check_same((got), (want), (tol), #got " vs " #want, __FILE__, __LINE__)

NN_TEST(contiguity_of_dense_tensors) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    NN_CHECK(nn::Tensor::zeros({4}, dev).is_contiguous());
    NN_CHECK(nn::Tensor::zeros({3, 5}, dev).is_contiguous());
    NN_CHECK(nn::Tensor::zeros({2, 3, 4, 5}, dev).is_contiguous());
    NN_CHECK(nn::Tensor::zeros({1, 7, 1}, dev).is_contiguous());   // extent-1 axes
  }
}

NN_TEST(permute_preserves_rank_and_reorders_dims) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    const nn::Tensor t = nn::Tensor::zeros({2, 3, 4}, dev);
    const int order[3] = {1, 2, 0};                 // a 3-cycle, not self-inverse
    const nn::Tensor p = t.permute(std::span<const int>(order, 3));

    NN_CHECK(p.shape().rank() == 3);                // rank 0 before the fix
    NN_CHECK(p.shape() == nn::Shape({3, 4, 2}));
    NN_CHECK(p.numel() == 24);
    NN_CHECK(!p.is_contiguous());
    NN_CHECK(p.stride(0) == 4 && p.stride(1) == 1 && p.stride(2) == 12);
  }
}

NN_TEST(views_report_contiguity_correctly) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    const nn::Tensor t = nn::Tensor::zeros({2, 3, 4}, dev);
    NN_CHECK(t.is_contiguous());
    NN_CHECK(t.slice(0, 0, 1).is_contiguous());     // outer slice: still dense
    NN_CHECK(!t.slice(2, 1, 2).is_contiguous());    // inner slice: not
    NN_CHECK(!t.transpose(1, 2).is_contiguous());
    NN_CHECK(t.transpose(1, 2).transpose(1, 2).is_contiguous());   // back again
    NN_CHECK(t.transpose(1, 2).contiguous().is_contiguous());
  }
}

NN_TEST(contiguous_materialises_the_right_elements) {
  const int B = 2, T = 5, H = 3, dh = 4;

  NN_TEST_FOR_EACH_DEVICE(dev) {
    std::vector<float> host(size_t(B) * T * H * dh);
    for (size_t i = 0; i < host.size(); ++i) host[i] = float(i);

    // [B,T,H,dh] -> [B,H,T,dh]: the attention permute
    const nn::Tensor base = nn::Tensor::from(host, nn::Shape({B, T, H, dh}), dev);
    const int order[4] = {0, 2, 1, 3};
    const nn::Tensor v = base.permute(std::span<const int>(order, 4));
    NN_CHECK(v.shape() == nn::Shape({B, H, T, dh}));

    const nn::Tensor packed = v.contiguous().to(nn::Device::CPU);
    NN_CHECK(packed.is_contiguous());

    int64_t k = 0;
    for (int b = 0; b < B; ++b)
      for (int h = 0; h < H; ++h)
        for (int t = 0; t < T; ++t)
          for (int c = 0; c < dh; ++c) {
            const float expect = host[((size_t(b) * T + t) * H + h) * dh + c];
            NN_CHECK_CLOSE(packed.host_data()[k++], expect, 0.0f);
          }
  }
}

NN_TEST(reshape_of_a_non_contiguous_view_copies) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    std::vector<float> host(24);
    for (size_t i = 0; i < host.size(); ++i) host[i] = float(i);

    const nn::Tensor t = nn::Tensor::from(host, nn::Shape({2, 3, 4}), dev);
    const nn::Tensor r = t.transpose(1, 2).reshape(nn::Shape({24}));

    NN_CHECK(r.is_contiguous());
    const nn::Tensor h = r.to(nn::Device::CPU);
    // transposed order: b, then c (was axis 2), then t (was axis 1)
    int64_t k = 0;
    for (int b = 0; b < 2; ++b)
      for (int c = 0; c < 4; ++c)
        for (int tt = 0; tt < 3; ++tt)
          NN_CHECK_CLOSE(h.host_data()[k++], host[size_t(b) * 12 + tt * 4 + c], 0.0f);
  }
}

NN_TEST(elementwise_ops_agree_on_a_view_and_its_copy) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    std::vector<float> host(6 * 8);
    for (size_t i = 0; i < host.size(); ++i) host[i] = float(i) * 0.25f - 5.0f;
    const nn::Tensor base = nn::Tensor::from(host, nn::Shape({6, 8}), dev);

    const nn::Tensor v = base.transpose(0, 1);          // [8,6], strides (1,8)
    const nn::Tensor c = v.contiguous();
    NN_CHECK(!v.is_contiguous());
    NN_CHECK(c.is_contiguous());

    NN_CHECK_SAME(nn::ops::relu(v), nn::ops::relu(c), 1e-6f);
    NN_CHECK_SAME(nn::ops::add(v, c), nn::ops::add(c, c), 1e-6f);
    NN_CHECK_SAME(nn::ops::add(c, v), nn::ops::add(c, c), 1e-6f);
    NN_CHECK_SAME(nn::ops::relu_backward(v, c), nn::ops::relu_backward(c, c), 1e-6f);
  }
}

NN_TEST(relu_on_a_permuted_rank4_view) {
  const int B = 2, T = 5, H = 3, dh = 4;

  NN_TEST_FOR_EACH_DEVICE(dev) {
    std::vector<float> host(size_t(B) * T * H * dh);
    for (size_t i = 0; i < host.size(); ++i) host[i] = float((i * 41) % 97) / 97.0f - 0.5f;
    const nn::Tensor base = nn::Tensor::from(host, nn::Shape({B, T, H, dh}), dev);

    const int order[4] = {0, 2, 1, 3};
    const nn::Tensor v = base.permute(std::span<const int>(order, 4));
    NN_CHECK_SAME(nn::ops::relu(v), nn::ops::relu(v.contiguous()), 1e-6f);
  }
}

NN_TEST(row_reductions_absorb_a_row_stride) {
  const int M = 7, N = 5, pad = 3;

  NN_TEST_FOR_EACH_DEVICE(dev) {
    std::vector<float> host(size_t(M) * (N + pad));
    for (size_t i = 0; i < host.size(); ++i) host[i] = float((i * 13) % 17) - 8.0f;
    const nn::Tensor wide = nn::Tensor::from(host, nn::Shape({M, N + pad}), dev);

    const nn::Tensor x = wide.slice(1, 0, N);
    NN_CHECK(x.stride(0) == N + pad && x.stride(1) == 1);
    NN_CHECK(!x.is_contiguous());
    const nn::Tensor packed = x.contiguous();

    NN_CHECK_SAME(nn::ops::argmax_rows(x), nn::ops::argmax_rows(packed), 0.0f);
    NN_CHECK_SAME(nn::ops::softmax_rows(x), nn::ops::softmax_rows(packed), 1e-6f);

    const nn::Tensor bias = nn::Tensor::from({1.0f, -2.0f, 3.0f, -4.0f, 5.0f}, dev);
    NN_CHECK_SAME(nn::ops::add(x, bias), nn::ops::add(packed, bias), 1e-6f);
  }
}

NN_TEST(softmax_ce_absorbs_a_row_stride) {
  const int M = 7, N = 5, pad = 3;

  NN_TEST_FOR_EACH_DEVICE(dev) {
    std::vector<float> host(size_t(M) * (N + pad));
    for (size_t i = 0; i < host.size(); ++i) host[i] = float((i * 13) % 17) / 17.0f;
    const nn::Tensor wide = nn::Tensor::from(host, nn::Shape({M, N + pad}), dev);
    const nn::Tensor z = wide.slice(1, 0, N);
    const nn::Tensor packed = z.contiguous();

    const nn::Tensor labels = nn::Tensor::from_i32({0, 4, 2, 1, 3, 0, 4}, dev);

    nn::Tensor loss_v = nn::Tensor::zeros({}, dev);
    nn::Tensor probs_v(z.shape(), dev, nn::DType::F32);
    nn::ops::softmax_ce(z, labels, loss_v, probs_v);

    nn::Tensor loss_p = nn::Tensor::zeros({}, dev);
    nn::Tensor probs_p(packed.shape(), dev, nn::DType::F32);
    nn::ops::softmax_ce(packed, labels, loss_p, probs_p);

    NN_CHECK_CLOSE(loss_v.item(), loss_p.item(), 1e-6f);
    NN_CHECK_SAME(probs_v, probs_p, 1e-6f);

    const nn::Tensor g_loss = nn::Tensor::scalar(1.0f, dev);
    const nn::Tensor cpu_probs = probs_p.to(nn::Device::CPU);
    std::vector<float> padded(size_t(M) * (N + pad), 999.0f);
    for (int i = 0; i < M; ++i) {
      for (int j = 0; j < N; ++j) {
        padded[size_t(i) * (N + pad) + j] = cpu_probs.host_data()[size_t(i) * N + j];
      }
    }
    const nn::Tensor probs_strided =
        nn::Tensor::from(padded, nn::Shape({M, N + pad}), dev).slice(1, 0, N);
    NN_CHECK(!probs_strided.is_contiguous());

    NN_CHECK_SAME(nn::ops::softmax_ce_backward(probs_strided, labels, g_loss),
                  nn::ops::softmax_ce_backward(probs_p, labels, g_loss), 1e-6f);
  }
}

NN_TEST(matmul_absorbs_a_row_stride) {
  const int T = 24, H = 3, dh = 8, d = H * dh, pad = 5;

  NN_TEST_FOR_EACH_DEVICE(dev) {
    std::vector<float> ha(size_t(T) * d);
    for (size_t i = 0; i < ha.size(); ++i) ha[i] = float((i * 37) % 101) / 101.0f - 0.5f;
    const nn::Tensor big = nn::Tensor::from(ha, nn::Shape({T, d}), dev);

    std::vector<float> hb(size_t(d) * (T + pad));
    for (size_t i = 0; i < hb.size(); ++i) hb[i] = float((i * 53) % 89) / 89.0f - 0.5f;
    const nn::Tensor wide = nn::Tensor::from(hb, nn::Shape({d, T + pad}), dev);

    for (int h = 0; h < H; ++h) {
      const nn::Tensor q = big.slice(1, int64_t(h) * dh, dh);          // [T,dh], lda = d
      const nn::Tensor r = wide.slice(0, int64_t(h) * dh, dh).slice(1, 0, T);  // [dh,T]
      NN_CHECK(q.stride(0) == d && q.stride(1) == 1 && !q.is_contiguous());
      NN_CHECK(r.stride(0) == T + pad && r.stride(1) == 1 && !r.is_contiguous());

      const nn::Tensor pq = q.contiguous();
      const nn::Tensor pr = r.contiguous();

      NN_CHECK_SAME(nn::ops::matmul(q, r,  false, false),
                    nn::ops::matmul(pq, pr, false, false), 1e-4f);
      NN_CHECK_SAME(nn::ops::matmul(q, q,  false, true),
                    nn::ops::matmul(pq, pq, false, true), 1e-4f);
      NN_CHECK_SAME(nn::ops::matmul(q, q,  true,  false),
                    nn::ops::matmul(pq, pq, true,  false), 1e-4f);
      NN_CHECK_SAME(nn::ops::matmul(r, q,  true,  true),
                    nn::ops::matmul(pr, pq, true,  true), 1e-4f);
    }
  }
}

NN_TEST(matmul_into_writes_only_its_own_columns) {
  const int T = 20, H = 3, dh = 8, d = H * dh;
  const int h = 1;                                     // the middle head

  NN_TEST_FOR_EACH_DEVICE(dev) {
    std::vector<float> ha(size_t(T) * dh), hb(size_t(dh) * dh);
    for (size_t i = 0; i < ha.size(); ++i) ha[i] = float((i * 29) % 61) / 61.0f - 0.5f;
    for (size_t i = 0; i < hb.size(); ++i) hb[i] = float((i * 17) % 43) / 43.0f - 0.5f;
    const nn::Tensor a = nn::Tensor::from(ha, nn::Shape({T, dh}), dev);
    const nn::Tensor b = nn::Tensor::from(hb, nn::Shape({dh, dh}), dev);

    nn::Tensor wide = nn::Tensor::full({T, d}, -7.0f, dev);
    nn::Tensor slot = wide.slice(1, int64_t(h) * dh, dh);
    NN_CHECK(slot.stride(0) == d && slot.stride(1) == 1);

    nn::ops::matmul_into(slot, a, b, false, false);     // [T,dh] x [dh,dh]
    NN_CHECK_SAME(slot, nn::ops::matmul(a, b, false, false), 1e-4f);

    const nn::Tensor hw = wide.to(nn::Device::CPU);
    for (int t = 0; t < T; ++t) {
      for (int c = 0; c < d; ++c) {
        if (c >= h * dh && c < (h + 1) * dh) continue;
        NN_CHECK_CLOSE(hw.host_data()[size_t(t) * d + c], -7.0f, 0.0f);
      }
    }
  }
}

NN_TEST(non_unit_innermost_stride_is_rejected) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    const nn::Tensor base = nn::Tensor::zeros({8, 8}, dev);
    const nn::Tensor bad = base.transpose(0, 1);        // innermost stride 8

    NN_CHECK_THROWS(nn::ops::argmax_rows(bad), std::invalid_argument);
    NN_CHECK_THROWS(nn::ops::matmul(bad, base, false, false), std::invalid_argument);
    NN_CHECK_THROWS(nn::ops::matmul(base, bad, false, false), std::invalid_argument);

    nn::Tensor dst = base.slice(1, 0, 4);               // non-contiguous destination
    NN_CHECK_THROWS(nn::ops::scale_inplace(dst, 2.0f), std::invalid_argument);
    NN_CHECK_THROWS(nn::ops::fill_inplace(dst, 1.0f), std::invalid_argument);
    NN_CHECK_THROWS(nn::ops::add_inplace(dst, dst), std::invalid_argument);
  }
}

NN_TEST(sum_all_does_not_stagnate_at_2_to_the_24) {
  const int64_t n = (int64_t(1) << 24) + 1;

  NN_TEST_FOR_EACH_DEVICE(dev) {
    const nn::Tensor x = nn::Tensor::full({int(n)}, 1.0f, dev);
    NN_CHECK_CLOSE(nn::ops::sum_all(x).item(), 16777217.0f, 1e-7f);
  }
}

NN_TEST(sum_all_is_deterministic) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    nn::Pcg32 rng(1234);
    const nn::Tensor x = nn::Tensor::randn({1000, 1000}, rng, 1.0f, dev);

    const float first = nn::ops::sum_all(x).item();
    for (int i = 0; i < 20; ++i) {
      NN_CHECK(nn::ops::sum_all(x).item() == first);
    }
  }
}

NN_TEST(sum_all_absorbs_strides) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    std::vector<float> host(7 * 8);
    for (size_t i = 0; i < host.size(); ++i) host[i] = float(i) - 20.0f;
    const nn::Tensor wide = nn::Tensor::from(host, nn::Shape({7, 8}), dev);

    // A slice drops columns, so a kernel that ignored strides would read a
    // different set of elements and still return a plausible number.
    const nn::Tensor v = wide.slice(1, 0, 5);
    NN_CHECK(!v.is_contiguous());

    float expect = 0.0f;
    for (int r = 0; r < 7; ++r)
      for (int c = 0; c < 5; ++c) expect += host[size_t(r) * 8 + c];

    NN_CHECK_CLOSE(nn::ops::sum_all(v).item(), expect, 1e-6f);
    NN_CHECK_CLOSE(nn::ops::sum_all(v).item(),
                   nn::ops::sum_all(v.contiguous()).item(), 1e-6f);

    // and a 3D permute, so the decode has to get more than one axis right
    const nn::Tensor cube = nn::Tensor::from(host, nn::Shape({2, 4, 7}), dev);
    const int order[3] = {2, 0, 1};
    const nn::Tensor p = cube.permute(std::span<const int>(order, 3));
    NN_CHECK_CLOSE(nn::ops::sum_all(p).item(), nn::ops::sum_all(cube).item(), 1e-6f);
  }
}

NN_TEST(sum_all_edge_shapes) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    NN_CHECK_CLOSE(nn::ops::sum_all(nn::Tensor::full({1}, 3.5f, dev)).item(), 3.5f, 1e-7f);
    NN_CHECK_CLOSE(nn::ops::sum_all(nn::Tensor::zeros({64, 64}, dev)).item(), 0.0f, 1e-7f);
    // fewer elements than the fixed 1024-block grid: most blocks contribute 0
    NN_CHECK_CLOSE(nn::ops::sum_all(nn::Tensor::full({10}, 2.0f, dev)).item(), 20.0f, 1e-7f);
  }
}

NN_TEST(fill_from_broadcasts_a_device_scalar) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    const nn::Tensor x = nn::Tensor::full({5, 5}, 2.0f, dev);
    const nn::Tensor total = nn::ops::sum_all(x);          // 50.0, never on the host

    nn::Tensor dst = nn::Tensor::zeros({3, 4}, dev);
    nn::ops::fill_from(dst, total);

    const nn::Tensor h = dst.to(nn::Device::CPU);
    for (int64_t i = 0; i < dst.numel(); ++i) {
      NN_CHECK_CLOSE(h.host_data()[i], 50.0f, 1e-6f);
    }

    NN_CHECK_THROWS(nn::ops::fill_from(dst, x), std::invalid_argument);   // not a scalar
    nn::Tensor view = nn::Tensor::zeros({4, 4}, dev).slice(1, 0, 2);
    NN_CHECK_THROWS(nn::ops::fill_from(view, total), std::invalid_argument);
  }
}

NN_TEST(sum_all_backward_fills_and_accumulates) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    nn::Tensor x = nn::Tensor::full({2, 3}, 4.0f, dev);
    x.set_requires_grad(true);

    for (int pass = 1; pass <= 2; ++pass) {
      nn::autograd::Tape tape;
      nn::autograd::TapeScope scope(tape);
      nn::Tensor s = nn::autograd::sum_all(x);
      NN_CHECK_CLOSE(s.item(), 24.0f, 1e-6f);
      tape.backward(s);

      const nn::Tensor g = x.grad().to(nn::Device::CPU);
      for (int64_t i = 0; i < g.numel(); ++i) {
        NN_CHECK_CLOSE(g.host_data()[i], float(pass), 1e-6f);
      }
    }
  }
}

NN_TEST(offset_of_matches_the_full_modulo_form) {
  std::vector<float> storage(4096);
  for (size_t i = 0; i < storage.size(); ++i) storage[i] = float(i);

  auto gather = [&](const nn::Tensor& v) {
    std::vector<float> out(size_t(v.numel()));
    for (int64_t i = 0; i < v.numel(); ++i) {
      int64_t rem = i, off = v.offset();
      for (int a = v.shape().rank() - 1; a >= 0; --a) {
        off += (rem % v.shape().dim(a)) * v.stride(a);
        rem /= v.shape().dim(a);
      }
      out[size_t(i)] = storage[size_t(off)];
    }
    return out;
  };

  NN_TEST_FOR_EACH_DEVICE(dev) {
    const nn::Tensor base = nn::Tensor::from(storage, nn::Shape({4096}), dev);

    std::vector<nn::Tensor> views;
    views.push_back(base.slice(0, 7, 100));                        // rank 1, offset
    views.push_back(base.reshape(nn::Shape({64, 64})).transpose(0, 1));
    views.push_back(base.reshape(nn::Shape({8, 16, 32})).slice(2, 3, 20));
    {
      const int order[3] = {2, 0, 1};
      views.push_back(base.reshape(nn::Shape({8, 16, 32}))
                          .permute(std::span<const int>(order, 3)));
    }
    {
      const int order[4] = {0, 2, 1, 3};
      views.push_back(base.reshape(nn::Shape({4, 8, 16, 8}))
                          .permute(std::span<const int>(order, 4)));
    }
    views.push_back(base.reshape(nn::Shape({2, 4, 8, 8, 8})).slice(3, 1, 5));
    views.push_back(base.slice(0, 0, 5).expand(nn::Shape({3, 7, 5})));  // stride 0
    views.push_back(base.slice(0, 11, 1));                              // single element

    for (size_t k = 0; k < views.size(); ++k) {
      const nn::Tensor& v = views[k];
      const std::vector<float> want = gather(v);
      const nn::Tensor got = v.contiguous().to(nn::Device::CPU);

      if (got.numel() != int64_t(want.size())) {
        nn::test::report(__FILE__, __LINE__, "view " + std::to_string(k) + ": wrong size");
        continue;
      }
      for (size_t i = 0; i < want.size(); ++i) {
        if (got.host_data()[i] != want[i]) {
          nn::test::report(__FILE__, __LINE__,
              "view " + std::to_string(k) + " element " + std::to_string(i) + " of " +
              std::to_string(want.size()) + ": " + std::to_string(got.host_data()[i]) +
              " vs " + std::to_string(want[i]));
          break;
        }
      }
    }
  }
}
