#pragma once

#include <string>
#include <vector>

#include <nn/autograd/functions.h>
#include <nn/core/state.h>
#include <nn/core/tensor.h>
#include <nn/module/module.h>

namespace nn {

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

  void collect_named(const std::string& prefix, std::vector<NamedTensor>& out) override {
    out.push_back({prefix + "w", &w_});
    out.push_back({prefix + "b", &b_});
  }

  Tensor& weight() { return w_; }
  Tensor& bias()   { return b_; }

private:
  Tensor w_, b_;
  float eps_;
};

class RMSNorm : public Module {
public:
  explicit RMSNorm(int features, float eps=1e-6f)
    : w_(Tensor::full({features}, 1.0f)), eps_(eps) {
    w_.set_requires_grad(true);
  }

  Tensor forward(const Tensor& x) override {
    return autograd::rms_norm(x, w_, eps_);
  }

  void collect_named(const std::string& prefix, std::vector<NamedTensor>& out) override {
    out.push_back({prefix + "w", &w_});
  }

  Tensor& weight() { return w_; }

private:
  Tensor w_;
  float eps_;
};

}  // namespace nn
