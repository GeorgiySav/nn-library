#pragma once

#include <span>
#include <string>
#include <vector>

#include <nn/core/tensor.h>

namespace nn {

// What a checkpoint is made of, and the common currency between a Module, an
// Optimizer and the file writer
struct NamedTensor {
  std::string name;
  Tensor* t = nullptr;
};

// Everything that is not a tensor: an Adam step count, a learning rate, the
// RNG counter
struct NamedScalar {
  std::string name;
  double value = 0.0;
};

double scalar_value(std::span<const NamedScalar> scalars, const std::string& name);

}  // namespace nn
