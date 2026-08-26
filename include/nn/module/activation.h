#pragma once

#include <string>
#include <vector>

#include <nn/core/state.h>
#include <nn/core/tensor.h>
#include <nn/module/module.h>

namespace nn {

class ReLu : public Module {
public:
  ReLu() = default;

  Tensor forward(const Tensor& x) override {
    return x.relu();
  }

  void collect_named(const std::string&, std::vector<NamedTensor>&) override {}
};

class GeLu : public Module {
public:
  GeLu() = default;

  Tensor forward(const Tensor& x) override {
    return x.gelu();
  }

  void collect_named(const std::string&, std::vector<NamedTensor>&) override {}
};

}  // namespace nn
