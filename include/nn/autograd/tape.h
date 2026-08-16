#pragma once

#include <functional>
#include <span>

#include <nn/core/small_vec.h>
#include <nn/core/tensor.h>
#include <nn/core/arena.h>

namespace nn::autograd {

using BackwardFn = std::function<void(const Tensor& g_out, std::span<Tensor> g_in)>;

struct Node {
  using Thunk = void(*)(void* captures, const Tensor& g, std::span<Tensor> g_in_);
  using Dtor  = void(*)(void* captures);

  Thunk backward = nullptr; // null -> leaf
  Dtor  destroy  = nullptr;
  void* captures = nullptr; // lives in the tape's arena

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
  
  template <class F>
  int record(F&& fn, SmallVec<int, 3> inputs, const char* name) {
    using Fn = std::decay_t<F>;
    void* mem = arena_.alloc(sizeof(Fn), alignof(Fn));
    new (mem) Fn(std::forward<F>(fn));

    Node n;
    n.captures = mem;
    n.backward = [](void* c, const Tensor& g, std::span<Tensor> gi) {
      (*static_cast<Fn*>(c))(g, gi);
    };
    n.destroy  = [](void* c) { static_cast<Fn*>(c)->~Fn(); };
    n.inputs   = inputs;
    n.name     = name;

    nodes_.push_back(std::move(n));
    return static_cast<int>(nodes_.size()) - 1;
  }

  void set_producer(Tensor& out, int id);

  void backward(const Tensor& loss, bool retain_graph = false);
  void clear();
  int size() const;  
 
  size_t arena_size() const { return arena_.bytes_reserved(); }
private:
  int record_leaf(std::shared_ptr<AutogradMeta> m);
  // true if 'm' was stamped by this tape
  bool owns(const AutogradMeta& m) const;
  // stamp a tensor's meat as produced by a node 'id' on this tape
  void bind(AutogradMeta& m, int id) const;

  std::vector<Node> nodes_;
  std::vector<Tensor> grads_;
  Arena arena_;
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