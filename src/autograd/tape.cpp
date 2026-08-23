#include <nn/autograd/tape.h>

#include <atomic>
#include <stdexcept>
#include <unordered_map>

#include <nn/ops/ops.h>

namespace nn::autograd {

static thread_local Tape* active_tape_ = nullptr;

// epoch -> tape
static thread_local std::unordered_map<uint64_t, Tape*> live_tapes_;

static uint64_t next_epoch() {
  static std::atomic<uint64_t> counter{0};
  return ++counter;
}

Tape* Tape::by_epoch(uint64_t epoch) {
  const auto it = live_tapes_.find(epoch);
  return (it == live_tapes_.end()) ? nullptr : it->second;
}

TapeScope::TapeScope(Tape& t) {
  if (active_tape_ && active_tape_ != &t) {
    throw std::logic_error("nested TapeScope: a different tape is already active");
  }
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

Tape::Tape() : epoch_(next_epoch()) { live_tapes_[epoch_] = this; }

Tape::~Tape() {
  clear();
  live_tapes_.erase(epoch_);
}

int Tape::node_for(const Tensor& t) {
  AutogradMeta* m = t.meta();

  if (!m) return -1;
  if (owns(*m)) return m->node_id;
  if (!m->requires_grad) return -1;

  return record_leaf(t.ensure_meta_shared());
}

void Tape::set_producer(Tensor& out, int id) {
  AutogradMeta& m = out.ensure_meta();
  m.requires_grad = true;
  bind(m, id);
}

int Tape::record_leaf(std::shared_ptr<AutogradMeta> m) {
  if (!m) throw std::invalid_argument("record_leaf: null meta");

  if (owns(*m)) {
    if (nodes_[m->node_id].leaf == m) return m->node_id;

    throw std::logic_error("record_leaf: tensor is already an op output");
  }

  int id = static_cast<int>(nodes_.size());
  nodes_.push_back(Node{
    .leaf = std::move(m),
    .name = "leaf"
  });
  bind(*nodes_[id].leaf, id);
  return id;
}

void Tape::backward(const Tensor& loss, bool retain_graph) {
  if (loss.shape().rank() != 0) {
    throw std::invalid_argument("loss must be a scalar tensor");
  }

  AutogradMeta* m = loss.meta();
  if (!m || !owns(*m)) {
    throw std::invalid_argument("loss tensor is not tracked by this tape");
  }

  NoGradScope no_grad;

  grads_.assign(nodes_.size(), Tensor{});
  grads_[loss.meta()->node_id] = Tensor::scalar(1.0f, loss.device(), loss.dtype());

  std::vector<Tensor> g_in;
  g_in.reserve(4);

  for (int i{static_cast<int>(nodes_.size()) - 1}; i >= 0; --i) {
    if (!grads_[i].defined()) {
      continue;
    }
    if (nodes_[i].leaf) {
      Tensor& dst = nodes_[i].leaf->grad;
      if (!dst.defined()) {
        dst = grads_[i].clone();
      } else {
        nn::ops::add_inplace(dst, grads_[i]);
      }
    }
    else {
      g_in.assign(nodes_[i].inputs.size(), Tensor{});
      nodes_[i].backward(nodes_[i].captures, grads_[i], g_in);

      for (int k{0}; k < static_cast<int>(nodes_[i].inputs.size()); ++k) {
        int input_id = nodes_[i].inputs[k];
        if (input_id < 0 || !g_in[k].defined()) {
          continue;
        }
        if (!grads_[input_id].defined()) {
          grads_[input_id] = g_in[k].pack();
        }
        else {
          nn::ops::add_inplace(grads_[input_id], g_in[k]);
        }
      }
    }
    grads_[i] = Tensor{};
  }

  grads_.clear();
  if (!retain_graph) clear();
}

void Tape::clear() {
  for (Node& n : nodes_) {
    if (n.destroy) n.destroy(n.captures);
  }
  nodes_.clear();
  grads_.clear();
  arena_.reset();

  live_tapes_.erase(epoch_);
  epoch_ = next_epoch(); // makes every stamped node_id stale
  live_tapes_[epoch_] = this;
}

bool Tape::owns(const AutogradMeta& m) const {
  return m.tape_epoch == epoch_ &&
         m.node_id >= 0 &&
         m.node_id < static_cast<int>(nodes_.size());
}

void Tape::bind(AutogradMeta& m, int id) const {
  m.node_id = id;
  m.tape_epoch = epoch_;
}

int Tape::size() const {
  return static_cast<int>(nodes_.size());
}

}

namespace nn {

// Defined here rather than in tensor.cpp because it is the one place core
// reaches into autograd, and doing it the other way round would mean tensor.h
// pulling in the tape.
void Tensor::backward(bool retain_graph) const {
  const AutogradMeta* m = meta();
  if (!m || m->tape_epoch == 0) {
    throw std::invalid_argument(
        "backward: this tensor was not produced by a recorded operation "
        "(no tape was active when it was computed)");
  }

  autograd::Tape* tape = autograd::Tape::by_epoch(m->tape_epoch);
  if (!tape) {
    throw std::invalid_argument(
        "backward: the tape that produced this tensor has been cleared or "
        "destroyed; keep the GradScope alive until after backward()");
  }

  tape->backward(*this, retain_graph);
}

}