#include <nn/autograd/functions.h>

#include <stdexcept>
#include <utility>

#include <nn/ops/ops.h>

#include "autograd_common.h"

namespace nn::autograd {

Tensor sum_all(const Tensor& x) {
  Tensor out = ops::sum_all(x);

  if (Tape* tape = active_tape(); tape && x.requires_grad()) {
    const int id = tape->record(
      [x](const Tensor& g_out, std::span<Tensor> g_in) {
        Tensor g(x.shape(), x.device(), x.dtype());
        ops::fill_from(g, g_out);
        g_in[0] = std::move(g);
      },
      {tape->node_for(x)},
      "sum_all"
    );
    tape->set_producer(out, id);
  }

  return out;
}

Tensor sum(const Tensor& x, int dim, bool keepdim) {
  const int d = x.shape().resolve_dim(dim, "sum");
  Tensor out = ops::sum_dim(x, d, keepdim);

  record_op(out, "sum",
    [sx = x.shape(), d, keepdim](const Tensor& g, std::span<Tensor> g_in) {
      // the gradient is just g broadcast back over the summed axis
      Shape kept = sx;
      kept.set_dim(d, 1);
      g_in[0] = (keepdim ? g : g.reshape_view(kept)).expand_view(sx);
    }, x);

  return out;
}

Tensor mean(const Tensor& x) {
  const int64_t n = x.numel();
  return scalar(ops::ScalarOp::MulScalar, sum_all(x), 1.0f / float(n));
}

Tensor mean(const Tensor& x, int dim, bool keepdim) {
  const int d = x.shape().resolve_dim(dim, "mean");
  const int n = x.shape().dim(d);
  return scalar(ops::ScalarOp::MulScalar, sum(x, d, keepdim), 1.0f / float(n));
}

Tensor var(const Tensor& x, int dim, bool keepdim, bool unbiased) {
  const int d = x.shape().resolve_dim(dim, "var");
  const int n = x.shape().dim(d);
  const int denom = unbiased ? (n - 1) : n;
  if (denom <= 0) {
    throw std::invalid_argument("var: axis is too short for an unbiased estimate");
  }

  const Tensor centred = binary(ops::BinaryOp::Sub, x, mean(x, d, /*keepdim=*/true));
  const Tensor sq = binary(ops::BinaryOp::Mul, centred, centred);
  return scalar(ops::ScalarOp::MulScalar, sum(sq, d, keepdim), 1.0f / float(denom));
}

Tensor stddev(const Tensor& x, int dim, bool keepdim, bool unbiased) {
  return unary(ops::UnaryOp::Sqrt, var(x, dim, keepdim, unbiased));
}

}  // namespace nn::autograd
