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

Tensor matmul(const Tensor& x, const Tensor& w) {
  Tensor out = ops::matmul(x, w);

  record_op(out, "matmul",
    [x, w](const Tensor& g, std::span<Tensor> g_in) {
      // g @ W^T. When W has no batch axes this folds to a single GEMM on its
      // own, the same way the forward did.
      g_in[0] = ops::sum_to(ops::matmul(g, w, false, true), x.shape());

      // X^T @ g. The transpose blocks the fold rule inside ops::matmul, so a
      // batched X would give one small GEMM per batch element and then a
      // reduction. Folding both operands by hand turns that back into the
      // single [K, B*T] x [B*T, N] GEMM it should be -- this is the weight
      // gradient of every Linear in the model, so it is worth the special case.
      if (w.shape().rank() == 2 && x.shape().rank() > 2) {
        g_in[1] = ops::matmul(as_matrix(x), as_matrix(g), true, false);
      } else {
        g_in[1] = ops::sum_to(ops::matmul(x, g, true, false), w.shape());
      }
    }, x, w);

  return out;
}

}  // namespace nn::autograd
