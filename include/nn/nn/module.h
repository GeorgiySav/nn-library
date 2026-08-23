#pragma once

#include <cmath>
#include <vector>
#include <memory>
#include <stdexcept>

#include <nn/core/tensor.h>
#include <nn/autograd/functions.h>

namespace nn {

class Module {
public:
  virtual ~Module() = default;
  virtual Tensor forward(const Tensor& x) = 0;
  virtual void collect_parameters(std::vector<Tensor*>& out) = 0;

  // So a call site reads `model(x)`. Non-virtual on purpose: overriding
  // forward is the extension point, and this only spells it differently.
  Tensor operator()(const Tensor& x) { return forward(x); }

  std::vector<Tensor*> parameters() {
    std::vector<Tensor*> p;
    collect_parameters(p);
    return p;
  }

  void zero_grad() { for (Tensor* p : parameters()) p->zero_grad(); }

  virtual void set_training(bool on) { training_ = on; }
  void train() { set_training(true); }
  void eval()  { set_training(false); }
  bool training() const { return training_; }

  void to(Device d) {
    for (Tensor* p : parameters()) {
      const bool rg = p->requires_grad();
      *p = p->to(d);
      p->set_requires_grad(rg);
    }
  }

protected:
  bool training_ = true;
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
    // b_ is [out_features]; broadcasting stretches it across the batch.
    return x.mm(w_) + b_;
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
    return x.relu();
  }

  void collect_parameters(std::vector<Tensor*>&) override {}
};

class GeLu : public Module {
public:
  GeLu() = default;

  Tensor forward(const Tensor& x) override {
    return x.gelu();
  }

  void collect_parameters(std::vector<Tensor*>&) override {}
};

// Normalises over the last axis. weight starts at 1 and bias at 0, so an
// untrained LayerNorm is exactly the normalisation and nothing else.
class LayerNorm : public Module {
public:
  explicit LayerNorm(int features, float eps = 1e-5f)
    : w_(Tensor::full({features}, 1.0f)), b_(Tensor::zeros({features})), eps_(eps) {
    w_.set_requires_grad(true);
    b_.set_requires_grad(true);
  }

  Tensor forward(const Tensor& x) override {
    return autograd::layer_norm(x, w_, b_, eps_);
  }

  void collect_parameters(std::vector<Tensor*>& out) override {
    out.push_back(&w_);
    out.push_back(&b_);
  }

private:
  Tensor w_, b_;
  float eps_;
};

// Takes I32 indices of any shape and returns them with a [dim] axis appended.
class Embedding : public Module {
public:
  Embedding(int vocab, int dim, Pcg32& rng)
    : w_(Tensor::randn({vocab, dim}, rng, 0.02f)) {
    w_.set_requires_grad(true);
  }

  Tensor forward(const Tensor& idx) override {
    return autograd::embedding(w_, idx);
  }

  void collect_parameters(std::vector<Tensor*>& out) override {
    out.push_back(&w_);
  }

private:
  Tensor w_;
};

class Dropout : public Module {
public:
  explicit Dropout(float p = 0.1f) : p_(p) {
    if (!(p >= 0.0f && p <= 1.0f)) {
      throw std::invalid_argument("Dropout: p must be in [0, 1]");
    }
  }

  Tensor forward(const Tensor& x) override {
    return autograd::dropout(x, p_, training());
  }

  void collect_parameters(std::vector<Tensor*>&) override {}

  float p() const { return p_; }

private:
  float p_;
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

  void set_training(bool on) override {
    training_ = on;
    for (auto& layer : layers_) layer->set_training(on);
  }

private:
  std::vector<std::unique_ptr<Module>> layers_;
};

}
