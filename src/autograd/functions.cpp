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

}