#include <nn/autograd/tape.h>

#include <stdexcept>

#include <nn/ops/ops.h>

namespace nn::autograd {

static thread_local Tape* active_tape_ = nullptr;

TapeScope::TapeScope(Tape& t) {
  prev_tape_ = active_tape_;
  active_tape_ = &t;
}

TapeScope::~TapeScope() {
  active_tape_ = prev_tape_;
}

Tape* active_tape() {
  return active_tape_;
}

NoGradScope::NoGradScope() {
  prev_tape_ = active_tape_;
  active_tape_ = nullptr;
}

NoGradScope::~NoGradScope() {
  active_tape_ = prev_tape_;
}

bool grad_enabled() {
  return active_tape_ != nullptr;
}

int Tape::record(BackwardFn fn, SmallVec<int, 3> inputs, const char* name) {
  int id = static_cast<int>(nodes_.size());
  for (int input_id : inputs) {
    if (!(input_id == -1 || (input_id >= 0 && input_id < id))) {
      throw std::invalid_argument("input node id is out of range");
    }
  }
  nodes_.push_back({fn, inputs, nullptr, name});
  return id;
}

int Tape::record_leaf(Tensor* param) {
  if (param->meta() &&
      param->meta()->node_id >= 0 &&
      param->meta()->node_id < static_cast<int>(nodes_.size()) &&
      nodes_[param->meta()->node_id].leaf == param) {
    return param->meta()->node_id;
  }

  int id = static_cast<int>(nodes_.size());
  nodes_.push_back({nullptr, {}, param, "leaf"});
  param->ensure_meta().node_id = id;
  return id;
}

void Tape::backward(const Tensor& loss) {
  if (loss.shape().rank() != 0) {
    throw std::invalid_argument("loss must be a scalar tensor");
  }

  if (!loss.meta() || loss.meta()->node_id < 0) {
    throw std::invalid_argument("loss tensor is not tracked by the tape");
  }

  grads_.assign(nodes_.size(), Tensor{});

  grads_[loss.meta()->node_id] = Tensor::scalar(1.0f);

  for (int i{static_cast<int>(nodes_.size()) - 1}; i >= 0; --i) {
    if (!grads_[i].defined()) {
      continue;
    }
    if (nodes_[i].leaf) {
      nn::ops::add_inplace(nodes_[i].leaf->grad(), grads_[i]);
    }
    else {
      std::vector<Tensor> g_in(nodes_[i].inputs.size());

      nodes_[i].backward(grads_[i], g_in);

      for (int k{0}; k < static_cast<int>(nodes_[i].inputs.size()); ++k) {
        int input_id = nodes_[i].inputs[k];
        if (input_id < 0 || !g_in[k].defined()) {
          continue;
        }
        if (!grads_[input_id].defined()) {
          grads_[input_id] = std::move(g_in[k]);
        }
        else {
          nn::ops::add_inplace(grads_[input_id], g_in[k]);
        }
      }
    }
    grads_[i] = Tensor{};
  }

  //clear();
}

void Tape::clear() {
  nodes_.clear();
  grads_.clear();
}

int Tape::size() const {
  return static_cast<int>(nodes_.size());
}

}