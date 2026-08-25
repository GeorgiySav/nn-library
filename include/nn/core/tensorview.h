#pragma once

#include <cstdint>

namespace nn {

inline constexpr int kMaxShapeRank = 8;

// layout of one tensor for the kernel
struct TensorView {
  int64_t shape[kMaxShapeRank] = {0};
  int64_t stride[kMaxShapeRank] = {0};
  int rank = 0;
};

}