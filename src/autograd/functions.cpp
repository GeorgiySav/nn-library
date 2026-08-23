#include <nn/autograd/functions.h>

#include <nn/autograd/tape.h>
#include <nn/ops/ops.h>
#include <initializer_list>
#include <stdexcept>

#include <nn/core/small_vec.h>

namespace nn::autograd {

namespace {

template<class F, class... Ts>
void record_op(Tensor& out,
               const char* name,
               F&& fn,
               const Ts&... ins) {
  Tape* tape = active_tape();
  if (!tape) return;

  NodeInputs ids;
  bool any_tracked = false;

  for (int id : {tape->node_for(ins)...}) {
    ids.push_back(id);
    any_tracked = any_tracked || (id >= 0);
  }

  if (!any_tracked) return; // inference

  tape->set_producer(out, tape->record(std::forward<F>(fn), ids, name));
}

}

Tensor unary(ops::UnaryOp op, const Tensor& x) {
  Tensor out = ops::unary(op, x);

  record_op(out, kernels::unary_op_name(op),
    [op, x, out](const Tensor& g, std::span<Tensor> g_in) {
      g_in[0] = ops::unary_backward(op, x, out, g);
    }, x);

  return out;
}

Tensor binary(ops::BinaryOp op, const Tensor& a, const Tensor& b) {
  Tensor out = ops::binary(op, a, b);

  record_op(out, kernels::binary_op_name(op),
    [op, a, b, out](const Tensor& g, std::span<Tensor> g_in) {
      g_in[0] = ops::binary_backward(op, 0, a, b, out, g);
      g_in[1] = ops::binary_backward(op, 1, a, b, out, g);
    }, a, b);

  return out;
}

Tensor scalar(ops::ScalarOp op, const Tensor& x, float k) {
  Tensor out = ops::scalar(op, x, k);

  record_op(out, kernels::scalar_op_name(op),
    [op, k, x, out](const Tensor& g, std::span<Tensor> g_in) {
      g_in[0] = ops::scalar_backward(op, x, out, g, k);
    }, x);

  return out;
}

// The named wrappers, generated from the same lists as the kernels.
#define NN_UNARY(Name, method, fwd, bwd) \
  Tensor method(const Tensor& x) { return unary(ops::UnaryOp::Name, x); }
#include <nn/kernels/unary_ops.def>
#undef NN_UNARY

#define NN_BINARY(Name, method, fwd, da, db) \
  Tensor method(const Tensor& a, const Tensor& b) { return binary(ops::BinaryOp::Name, a, b); }
#include <nn/kernels/binary_ops.def>
#undef NN_BINARY

#define NN_SCALAR(Name, method, fwd, bwd) \
  Tensor method(const Tensor& x, float k) { return scalar(ops::ScalarOp::Name, x, k); }
#include <nn/kernels/scalar_ops.def>
#undef NN_SCALAR

namespace {

// Every axis but the last folded into one row index. Used only on gradients
// inside a backward, where nothing is being recorded.
Tensor as_matrix(const Tensor& t) {
  const int r = t.shape().rank();
  const int last = t.shape().dim(r - 1);
  return t.reshape(Shape({int(t.numel() / last), last}));
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

Tensor embedding(const Tensor& weight, const Tensor& idx) {
  Tensor out = ops::embedding(weight, idx);

  record_op(out, "embedding",
    [idx, V = weight.shape().dim(0)](const Tensor& g, std::span<Tensor> g_in) {
      g_in[0] = ops::embedding_backward(g, idx, V);
    }, weight);

  return out;
}

Tensor contiguous(const Tensor& x) {
  if (x.is_contiguous()) return x;

  Tensor out = x.contiguous();
  record_op(out, "contiguous",
    [](const Tensor& g, std::span<Tensor> g_in) {
      g_in[0] = g;
    }, x);

  return out;
}

Tensor permute(const Tensor& x, std::span<const int> order) {
  Tensor out = x.permute(order);

  if (Tape* tape = active_tape(); tape && x.requires_grad()) {
    // inverse[order[i]] = i
    SmallVec<int, 8> inv(order.size());
    for (int i{0}; i < int(order.size()); ++i) inv[order[i]] = i;

    const int id = tape->record(
      [inv](const Tensor& g_out, std::span<Tensor> g_in) mutable {
        g_in[0] = g_out.permute(inv.span());
      },
      {tape->node_for(x)},
      "permute"
    );
    tape->set_producer(out, id);
  }

  return out;
}

Tensor transpose(const Tensor& x, int a, int b) {
  const int r = x.shape().rank();
  const int ia = ops::normalise_dim(a, r, "transpose");
  const int ib = ops::normalise_dim(b, r, "transpose");

  int order[kMaxShapeRank];
  for (int i = 0; i < r; ++i) order[i] = i;
  order[ia] = ib;
  order[ib] = ia;
  return permute(x, std::span<const int>(order, r));
}

Tensor reshape(const Tensor& x, std::span<const int> shape) {
  Tensor out = x.reshape(shape);

  if (Tape* tape = active_tape(); tape && x.requires_grad()) {
    const int id = tape->record(
      [x](const Tensor& g_out, std::span<Tensor> g_in) {
        g_in[0] = g_out.reshape(x.shape());
      },
      {tape->node_for(x)},
      "reshape"
    );
    tape->set_producer(out, id);
  }

  return out;
}

Tensor slice(const Tensor& x, int axis, int64_t start, int64_t len) {
  Tensor out = x.slice(axis, start, len);

  if (Tape* tape = active_tape(); tape && x.requires_grad()) {
    const int id = tape->record(
      [x, axis, start, len](const Tensor& g_out, std::span<Tensor> g_in) {
        // Zero everywhere the slice did not reach, g_out inside the window.
        // The window is a view sharing storage with g, so unpack writes
        // through its strides into g.
        Tensor g = Tensor::zeros(x.shape(), x.device(), x.dtype());
        Tensor window = g.slice(axis, start, len);
        ops::unpack(window, g_out);
        g_in[0] = std::move(g);
      },
      {tape->node_for(x)},
      "slice"
    );
    tape->set_producer(out, id);
  }

  return out;
}

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
  const int d = ops::normalise_dim(dim, x.shape().rank(), "sum");
  Tensor out = ops::sum_dim(x, d, keepdim);

  record_op(out, "sum",
    [sx = x.shape(), d, keepdim](const Tensor& g, std::span<Tensor> g_in) {
      Shape kept = sx;
      kept.set_dim(d, 1);
      g_in[0] = (keepdim ? g : g.reshape(kept)).expand(sx);
    }, x);

  return out;
}

Tensor mean(const Tensor& x) {
  const int64_t n = x.numel();
  return scalar(ops::ScalarOp::MulScalar, sum_all(x), 1.0f / float(n));
}

Tensor mean(const Tensor& x, int dim, bool keepdim) {
  const int d = ops::normalise_dim(dim, x.shape().rank(), "mean");
  const int n = x.shape().dim(d);
  return scalar(ops::ScalarOp::MulScalar, sum(x, d, keepdim), 1.0f / float(n));
}

Tensor var(const Tensor& x, int dim, bool keepdim, bool unbiased) {
  const int d = ops::normalise_dim(dim, x.shape().rank(), "var");
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

Tensor cat(std::span<const Tensor> parts, int dim) {
  if (parts.empty()) throw std::invalid_argument("cat: nothing to concatenate");
  if (int(parts.size()) > kMaxNodeInputs) {
    throw std::invalid_argument("cat: at most " + std::to_string(kMaxNodeInputs) +
                                " tensors at a time (the tape records a node's "
                                "inputs inline)");
  }

  const Shape& first = parts[0].shape();
  const int d = ops::normalise_dim(dim, first.rank(), "cat");

  int total = 0;
  for (const Tensor& p : parts) {
    if (p.shape().rank() != first.rank()) {
      throw std::invalid_argument("cat: " + p.shape().str() + " and " + first.str() +
                                  " have different ranks");
    }
    for (int i = 0; i < first.rank(); ++i) {
      if (i != d && p.shape().dim(i) != first.dim(i)) {
        throw std::invalid_argument("cat: " + p.shape().str() + " and " + first.str() +
                                    " differ away from axis " + std::to_string(d));
      }
    }
    if (p.device() != parts[0].device() || p.dtype() != parts[0].dtype()) {
      throw std::invalid_argument("cat: operands differ in device or dtype");
    }
    total += p.shape().dim(d);
  }

  Shape out_shape = first;
  out_shape.set_dim(d, total);
  Tensor out(out_shape, parts[0].device(), parts[0].dtype());

  SmallVec<int64_t, kMaxNodeInputs> starts;
  int64_t offset = 0;
  for (const Tensor& p : parts) {
    const int64_t len = p.shape().dim(d);
    Tensor window = out.slice(d, offset, len);
    ops::unpack(window, p);
    starts.push_back(offset);
    offset += len;
  }

  Tape* tape = active_tape();
  if (!tape) return out;

  NodeInputs ids;
  bool any_tracked = false;
  for (const Tensor& p : parts) {
    const int id = tape->node_for(p);
    ids.push_back(id);
    any_tracked = any_tracked || (id >= 0);
  }
  if (!any_tracked) return out;

  SmallVec<int64_t, kMaxNodeInputs> lens;
  for (const Tensor& p : parts) lens.push_back(p.shape().dim(d));

  tape->set_producer(out, tape->record(
      [d, starts, lens](const Tensor& g, std::span<Tensor> g_in) mutable {
        for (int i = 0; i < starts.size(); ++i) {
          g_in[i] = g.slice(d, starts[i], lens[i]);
        }
      },
      ids, "cat"));

  return out;
}

Tensor cat(std::initializer_list<Tensor> parts, int dim) {
  return cat(std::span<const Tensor>(parts.begin(), parts.size()), dim);
}

Tensor masked_fill(const Tensor& x, const Tensor& mask, float value) {
  const Tensor keep = ops::scalar(ops::ScalarOp::RsubScalar, mask, 1.0f);
  const Tensor fill = ops::scalar(ops::ScalarOp::MulScalar, mask, value);
  return binary(ops::BinaryOp::Add, binary(ops::BinaryOp::Mul, x, keep), fill);
}

}

namespace nn {

#define NN_UNARY(Name, method, fwd, bwd) \
  Tensor Tensor::method() const { return autograd::unary(kernels::UnaryOp::Name, *this); }
#include <nn/kernels/unary_ops.def>
#undef NN_UNARY

#define NN_BINARY(Name, method, fwd, da, db)                     \
  Tensor Tensor::method(const Tensor& other) const {             \
    return autograd::binary(kernels::BinaryOp::Name, *this, other); }
#include <nn/kernels/binary_ops.def>
#undef NN_BINARY

#define NN_SCALAR(Name, method, fwd, bwd) \
  Tensor Tensor::method(float k) const { return autograd::scalar(kernels::ScalarOp::Name, *this, k); }
#include <nn/kernels/scalar_ops.def>
#undef NN_SCALAR

Tensor Tensor::pow(float e) const { return pow_scalar(e); }
Tensor Tensor::mm(const Tensor& other) const { return autograd::matmul(*this, other); }
Tensor Tensor::t() const { return autograd::transpose(*this, -2, -1); }
Tensor Tensor::softmax() const { return autograd::softmax(*this); }

Tensor Tensor::sum() const { return autograd::sum_all(*this); }
Tensor Tensor::sum(int dim, bool keepdim) const { return autograd::sum(*this, dim, keepdim); }
Tensor Tensor::mean() const { return autograd::mean(*this); }
Tensor Tensor::mean(int dim, bool keepdim) const { return autograd::mean(*this, dim, keepdim); }
Tensor Tensor::var(int dim, bool keepdim, bool unbiased) const {
  return autograd::var(*this, dim, keepdim, unbiased);
}

Tensor operator+(const Tensor& a, const Tensor& b) { return autograd::add(a, b); }
Tensor operator-(const Tensor& a, const Tensor& b) { return autograd::sub(a, b); }
Tensor operator*(const Tensor& a, const Tensor& b) { return autograd::mul(a, b); }
Tensor operator/(const Tensor& a, const Tensor& b) { return autograd::div(a, b); }

Tensor operator+(const Tensor& a, float k) { return autograd::add_scalar(a, k); }
Tensor operator+(float k, const Tensor& a) { return autograd::add_scalar(a, k); }
Tensor operator-(const Tensor& a, float k) { return autograd::add_scalar(a, -k); }
Tensor operator-(float k, const Tensor& a) { return autograd::rsub_scalar(a, k); }
Tensor operator*(const Tensor& a, float k) { return autograd::mul_scalar(a, k); }
Tensor operator*(float k, const Tensor& a) { return autograd::mul_scalar(a, k); }
Tensor operator/(const Tensor& a, float k) { return autograd::mul_scalar(a, 1.0f / k); }
Tensor operator/(float k, const Tensor& a) { return autograd::rdiv_scalar(a, k); }

Tensor operator-(const Tensor& a) { return autograd::neg(a); }

}
