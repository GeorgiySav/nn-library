#pragma once

#include <concepts>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <nn/core/state.h>
#include <nn/core/tensor.h>
#include <nn/module/module.h>

namespace nn {

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

  void collect_named(const std::string& prefix, std::vector<NamedTensor>& out) override {
    for (size_t i = 0; i < layers_.size(); ++i) {
      layers_[i]->collect_named(prefix + std::to_string(i) + ".", out);
    }
  }

  void set_training(bool on) override {
    training_ = on;
    for (auto& layer : layers_) layer->set_training(on);
  }

private:
  std::vector<std::unique_ptr<Module>> layers_;
};

}  // namespace nn
