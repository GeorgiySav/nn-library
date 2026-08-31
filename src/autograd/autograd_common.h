#pragma once

#include <initializer_list>
#include <utility>

#include <nn/autograd/tape.h>
#include <nn/core/tensor.h>

namespace nn::autograd {

// Attaches a backward to `out` on the active tape, or does nothing if there is
// no tape (inference) or none of the inputs is on it. Every differentiable op
// in this directory ends with a call to this.
template <class F, class... Ts>
[[maybe_unused]] static void record_op(Tensor& out,
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

  if (!any_tracked) return;  // inference

  tape->set_producer(out, tape->record(std::forward<F>(fn), ids, name));
}

}  // namespace nn::autograd
