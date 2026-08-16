#pragma once

#include <functional>
#include <span>

#include <nn/core/small_vec.h>
#include <nn/core/tensor.h>

namespace nn::autograd {

using BackwardFn = std::function<void(const Tensor& g_out, std::span<Tensor> g_in)>;

struct Node {
  BackwardFn backward;
  nn::SmallVec<int, 3> inputs; // node ids
  std::shared_ptr<AutogradMeta> leaf; // non-null -> accumulate into leaf->grad
  const char* name = "";       // for debugging
};

class Tape {
public:
  Tape();
  ~Tape();

  // recording
  int node_for(const Tensor& t); // resolve an input to a node id, or -1
  int record(BackwardFn fn, SmallVec<int, 3> inputs, const char* name);
  void set_producer(Tensor& out, int id);

  void backward(const Tensor& loss, bool retain_graph = false);
  void clear();
  int size() const;  
  
private:
  int record_leaf(std::shared_ptr<AutogradMeta> m);
  // true if 'm' was stamped by this tape
  bool owns(const AutogradMeta& m) const;
  // stamp a tensor's meat as produced by a node 'id' on this tape
  void bind(AutogradMeta& m, int id) const;

  std::vector<Node> nodes_;
  std::vector<Tensor> grads_;
  uint64_t epoch_ = 0;
};

class TapeScope {
public:
  explicit TapeScope(Tape& t);
  ~TapeScope();

private:
  Tape* prev_tape_ = nullptr;
};

class NoGradScope {
public:
  NoGradScope();
  ~NoGradScope();

private:
  Tape* prev_tape_ = nullptr;
};

Tape* active_tape();
bool  grad_enabled();

}