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
  Tensor* leaf = nullptr;      // non-null -> accumulate g_out into leaf->grad
  const char* name = "";       // for debugging
};

class Tape {
public:
  int record(BackwardFn fn, SmallVec<int, 3> inputs, const char* name);
  int record_leaf(Tensor* param);
  void backward(const Tensor& loss);
  void clear();
  int size() const;

private:
  std::vector<Node> nodes_;
  std::vector<Tensor> grads_;
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