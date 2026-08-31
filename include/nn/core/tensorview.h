#pragma once

#include <cstdint>

namespace nn {

inline constexpr int kMaxShapeRank = 8;

// Kernel-facing description of one tensor's memory layout. Built by
// view_of(), which drops size-1 axes and merges adjacent contiguous axes, so
// rank here can be smaller than the originating Tensor's.
struct TensorView {
  int64_t shape[kMaxShapeRank] = {0};
  int64_t stride[kMaxShapeRank] = {0};
  int rank = 0;
};

}