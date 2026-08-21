#pragma once

#include <vector>
#include <memory>

#include <nn/core/tensor.h>
#include <nn/autograd/functions.h>

namespace nn {

class Module {
public:
  virtual ~Module() = default;
  virtual Tensor forward(const Tensor& x) = 0;
  virtual void collect_parameters(std::vector<Tensor*>& out) = 0;

  std::vector<Tensor*> parameters() {
    std::vector<Tensor*> p;
    collect_parameters(p);
    return p;
  }

  void zero_grad() { for (Tensor* p : parameters()) p->zero_grad(); }

  void to(Device d) {
    for (Tensor* p : parameters()) {
      const bool rg = p->requires_grad();
      *p = p->to(d);
      p->set_requires_grad(rg);
    }
  }
};

class Linear : public Module {
public:
  Linear(int in_features, int out_features, Pcg32& rng)
    : w_(Tensor::randn({in_features, out_features}, rng, std::sqrt(2.0f / in_features))),
      b_(Tensor::zeros({out_features})) {
    w_.set_requires_grad(true);
    b_.set_requires_grad(true);
  }

  Tensor forward(const Tensor& x) override {
    return autograd::add_row_bias(
      autograd::matmul(x, w_), 
    b_);
  }

  void collect_parameters(std::vector<Tensor*>& out) override {
    out.push_back(&w_);
    out.push_back(&b_);
  }

private:
  Tensor w_, b_;
};

class ReLu : public Module {
public:
  ReLu() = default;

  Tensor forward(const Tensor& x) override {
    return autograd::relu(x);
  }

  void collect_parameters(std::vector<Tensor*>& out) override {}
};

class Sequential : public Module {
public:
  Sequential() = default;

  // takes layers by value and moves them onto the heap
  template <class... Ms>
  requires (sizeof...(Ms) > 0 &&
            (std::derived_from<std::decay_t<Ms>, Module> && ...))
  explicit Sequential(Ms&&... ms) {
    layers_.reserve(sizeof...(Ms));
    (layers_.push_back(std::make_unique<std::decay_t<Ms>>(std::forward<Ms>(ms))), ...);
  }

  Tensor forward(const Tensor& x) override {
    Tensor out = x;
    for (const auto& layer : layers_) out = layer->forward(out);
    return out;
  }

  void collect_parameters(std::vector<Tensor*>& out) override {
    for (auto& layer : layers_) layer->collect_parameters(out);
  }

private:
  std::vector<std::unique_ptr<Module>> layers_;
};

}