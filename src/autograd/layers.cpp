#include <nn/autograd/functions.h>

#include <cmath>
#include <stdexcept>

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

Tensor cross_entropy(const Tensor& logits, const Tensor& labels, const Tensor& weights) {
  Tensor loss = Tensor::scalar(0.0f, logits.device(), logits.dtype());
  Tensor probs(logits.shape(), logits.device(), logits.dtype());
  ops::softmax_ce_weighted(logits, labels, weights, loss, probs);

  record_op(loss, "cross_entropy_weighted",
    [probs, labels, weights](const Tensor& g, std::span<Tensor> g_in) {
      g_in[0] = ops::softmax_ce_weighted_backward(probs, labels, weights, g);
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

Tensor rms_norm(const Tensor& x, const Tensor& weight, float eps) {
  const int last = x.shape().rank() - 1;
  const Tensor v = mean(binary(ops::BinaryOp::Mul, x, x), last, /*keepdim=*/true);
  const Tensor inv = unary(ops::UnaryOp::Rsqrt,
                           scalar(ops::ScalarOp::AddScalar, v, eps));

  Tensor out = binary(ops::BinaryOp::Mul, x, inv);
  if (weight.defined()) out = binary(ops::BinaryOp::Mul, out, weight);
  return out;
}

Tensor masked_fill(const Tensor& x, const Tensor& mask, float value) {
  const Tensor keep = ops::scalar(ops::ScalarOp::RsubScalar, mask, 1.0f);
  const Tensor fill = ops::scalar(ops::ScalarOp::MulScalar, mask, value);
  return binary(ops::BinaryOp::Add, binary(ops::BinaryOp::Mul, x, keep), fill);
}

Tensor scaled_dot_product_attention(const Tensor& q, const Tensor& k, const Tensor& v,
                                    const Tensor& mask, float dropout_p, bool is_causal,
                                    bool training) {
  const int r = q.shape().rank();
  if (r < 2 || k.shape().rank() != r || v.shape().rank() != r) {
    throw std::invalid_argument(
        "scaled_dot_product_attention: q, k, v must all have the same rank (>= 2)");
  }
  const int dk = q.shape().dim(r - 1);
  if (k.shape().dim(r - 1) != dk) {
    throw std::invalid_argument(
        "scaled_dot_product_attention: q and k must have the same head dimension");
  }
  const int Tq = q.shape().dim(r - 2);
  const int Tk = k.shape().dim(r - 2);
  if (v.shape().dim(r - 2) != Tk) {
    throw std::invalid_argument(
        "scaled_dot_product_attention: k and v must have the same sequence length");
  }
  if (is_causal && Tq != Tk) {
    throw std::invalid_argument(
        "scaled_dot_product_attention: is_causal requires equal query and key length");
  }

  Tensor scores = mul_scalar(matmul(q, k, false, true), 1.0f / std::sqrt(float(dk)));

  if (is_causal || mask.defined()) {
    Tensor keep = is_causal ? tril_mask(Tq, q.device()) : Tensor();
    if (mask.defined()) {
      keep = keep.defined() ? ops::binary(ops::BinaryOp::Mul, mask, keep) : mask;
    }
    const Tensor block = ops::scalar(ops::ScalarOp::RsubScalar, keep, 1.0f);
    scores = masked_fill(scores, block, -1e9f);
  }

  const Tensor probs = dropout(softmax(scores), dropout_p, training);
  return matmul(probs, v);
}

}  // namespace nn::autograd
