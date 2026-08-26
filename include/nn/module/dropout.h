#pragma once

#include <stdexcept>
#include <string>
#include <vector>

#include <nn/autograd/functions.h>
#include <nn/core/state.h>
#include <nn/core/tensor.h>
#include <nn/module/module.h>

namespace nn {

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

  void collect_named(const std::string&, std::vector<NamedTensor>&) override {}

  float p() const { return p_; }

private:
  float p_;
};

}  // namespace nn
