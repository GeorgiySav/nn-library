#include <nn/autograd/functions.h>

#include <nn/core/rng.h>
#include <nn/ops/ops.h>

#include "autograd_common.h"

namespace nn::autograd {

Tensor cross_entropy(const Tensor& logits, const Tensor& labels) {
  Tensor loss = Tensor::scalar(0.0f, logits.device(), logits.dtype());
  Tensor probs(logits.shape(), logits.device(), logits.dtype());
  ops::softmax_ce(logits, labels, loss, probs);

  record_op(loss, "cross_entropy",
    [probs, labels](const Tensor& g, std::span<Tensor> g_in) {
      g_in[0] = ops::softmax_ce_backward(probs, labels, g);
    },
  logits);

  return loss;
}

Tensor softmax(const Tensor& x) {
  Tensor out = ops::softmax_rows(x);

  record_op(out, "softmax",
    [out](const Tensor& g, std::span<Tensor> g_in) {
      g_in[0] = ops::softmax_rows_backward(out, g);
    }, x);

  return out;
}

Tensor dropout(const Tensor& x, float p, bool training) {
  if (!training || p == 0.0f) return x;

  const uint64_t seed = random_seed();
  const uint64_t offset = reserve_random(x.numel());

  Tensor out = ops::dropout(x, p, seed, offset);

  record_op(out, "dropout",
    [p, seed, offset](const Tensor& g, std::span<Tensor> g_in) {
      g_in[0] = ops::dropout(g, p, seed, offset);
    }, x);

  return out;
}

Tensor embedding(const Tensor& weight, const Tensor& idx) {
  Tensor out = ops::embedding(weight, idx);

  record_op(out, "embedding",
    [idx, V = weight.shape().dim(0)](const Tensor& g, std::span<Tensor> g_in) {
      g_in[0] = ops::embedding_backward(g, idx, V);
    }, weight);

  return out;
}

// Composed out of the ops above, so it records nothing of its own -- the tape
// gets one node per step and the chain rule does the rest.
Tensor layer_norm(const Tensor& x, const Tensor& weight, const Tensor& bias, float eps) {
  const int last = x.shape().rank() - 1;
  const Tensor centred = binary(ops::BinaryOp::Sub, x, mean(x, last, /*keepdim=*/true));
  const Tensor v = mean(binary(ops::BinaryOp::Mul, centred, centred), last, /*keepdim=*/true);
  const Tensor inv = unary(ops::UnaryOp::Rsqrt,
                           scalar(ops::ScalarOp::AddScalar, v, eps));

  Tensor out = binary(ops::BinaryOp::Mul, centred, inv);
  if (weight.defined()) out = binary(ops::BinaryOp::Mul, out, weight);
  if (bias.defined())   out = binary(ops::BinaryOp::Add, out, bias);
  return out;
}

// Likewise composed: keep * x + value * mask. The mask itself carries no
// gradient, so the two scalar ops run through nn::ops and stay off the tape.
Tensor masked_fill(const Tensor& x, const Tensor& mask, float value) {
  const Tensor keep = ops::scalar(ops::ScalarOp::RsubScalar, mask, 1.0f);
  const Tensor fill = ops::scalar(ops::ScalarOp::MulScalar, mask, value);
  return binary(ops::BinaryOp::Add, binary(ops::BinaryOp::Mul, x, keep), fill);
}

}  // namespace nn::autograd
