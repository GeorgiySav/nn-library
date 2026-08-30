#include <nn/autograd/functions.h>

#include <nn/ops/ops.h>

#include "autograd_common.h"

namespace nn::autograd {

namespace {

// Every axis but the last folded into one row index. Used only on gradients
// inside a backward, where nothing is being recorded.
Tensor as_matrix(const Tensor& t) {
  const int r = t.shape().rank();
  const int last = t.shape().dim(r - 1);
  return t.reshape_view(Shape({int(t.numel() / last), last}));
}

}  // namespace

Tensor matmul(const Tensor& x, const Tensor& w, bool transA, bool transB) {
  Tensor out = ops::matmul(x, w, transA, transB);

  record_op(out, "matmul",
    [x, w, transA, transB](const Tensor& g, std::span<Tensor> g_in) {
      g_in[0] = transA
          ? ops::sum_to(ops::matmul(w, g, transB, true), x.shape())
          : ops::sum_to(ops::matmul(g, w, false, !transB), x.shape());

      if (w.shape().rank() == 2 && x.shape().rank() > 2 && !transA) {
        g_in[1] = transB ? ops::matmul(as_matrix(g), as_matrix(x), true, false)
                         : ops::matmul(as_matrix(x), as_matrix(g), true, false);
      } else if (!transA) {
        g_in[1] = transB
            ? ops::sum_to(ops::matmul(g, x, true, false), w.shape())
            : ops::sum_to(ops::matmul(x, g, true, false), w.shape());
      } else {
        g_in[1] = transB
            ? ops::sum_to(ops::matmul(g, x, true, true), w.shape())
            : ops::sum_to(ops::matmul(x, g, false, false), w.shape());
      }
    }, x, w);

  return out;
}

}  // namespace nn::autograd
