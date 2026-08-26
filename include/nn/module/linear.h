#pragma once

#include <cmath>
#include <string>
#include <vector>

#include <nn/core/rng.h>
#include <nn/core/state.h>
#include <nn/core/tensor.h>
#include <nn/module/module.h>

namespace nn {

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

  void collect_named(const std::string& prefix, std::vector<NamedTensor>& out) override {
    out.push_back({prefix + "w", &w_});
    out.push_back({prefix + "b", &b_});
  }

private:
  Tensor w_, b_;
};

}  // namespace nn
