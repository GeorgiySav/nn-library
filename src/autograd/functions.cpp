#include <nn/autograd/functions.h>

#include <nn/autograd/tape.h>
#include <nn/ops/ops.h>
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
  
  SmallVec<int, 3> ids;
  bool any_tracked = false;

  for (int id : {tape->node_for(ins)...}) {
    ids.push_back(id);
    any_tracked = any_tracked || (id >= 0);
  }

  if (!any_tracked) return; // inference

  tape->set_producer(out, tape->record(std::forward<F>(fn), ids, name));
}

}

Tensor matmul(const Tensor& x, const Tensor& w) {
  Tensor out = ops::matmul(x, w);

  record_op(out, "matmul",
    [x, w](const Tensor& g, std::span<Tensor> g_in) {
      g_in[0] = ops::matmul(g, w, false, true); // g @ W^T
      g_in[1] = ops::matmul(x, g, true, false); // X^T @ g
    }, x, w);

  return out;
}

Tensor add(const Tensor& a, const Tensor& b) {
  Tensor out = ops::add(a, b);

  record_op(out, "add",
    [sa = a.shape(), sb = b.shape()](const Tensor& g, std::span<Tensor> g_in) {
      g_in[0] = ops::sum_to(g, sa);
      g_in[1] = ops::sum_to(g, sb);
    }, a, b);

  return out;
}

Tensor relu(const Tensor& x) {
  Tensor out = ops::relu(x);

  record_op(out, "relu",
    [out](const Tensor& g, std::span<Tensor> g_in) {
      g_in[0] = ops::relu_backward(out, g);
    }, x);

  return out;
}

Tensor add_row_bias(const Tensor& x, const Tensor& b) {
  Tensor out = ops::add_row_bias(x, b);

  record_op(out, "add_row_bias",
    [](const Tensor& g, std::span<Tensor> g_in) {
      g_in[0] = g.clone();       // identity w.r.t x
      g_in[1] = ops::col_sum(g); // bias is broadcast
    }, x, b);
  
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
        // The window is a view sharing storage with g, so copy_into writes
        // through its strides into g.
        Tensor g = Tensor::zeros(x.shape(), x.device(), x.dtype());
        Tensor window = g.slice(axis, start, len);
        ops::copy_into(window, g_out);
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

}
