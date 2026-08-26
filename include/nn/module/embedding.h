#pragma once

#include <string>
#include <vector>

#include <nn/autograd/functions.h>
#include <nn/core/rng.h>
#include <nn/core/state.h>
#include <nn/core/tensor.h>
#include <nn/module/module.h>

namespace nn {

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

  void collect_named(const std::string& prefix, std::vector<NamedTensor>& out) override {
    out.push_back({prefix + "w", &w_});
  }

private:
  Tensor w_;
};

}  // namespace nn
