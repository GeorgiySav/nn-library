#pragma once

#include <stdexcept>
#include <string>

#include <nn/core/tensor.h>

namespace nn::ops {

[[maybe_unused]] static void same_device(const Tensor& a, const Tensor& b, const char* op) {
  if (a.device() != b.device()) {
    throw std::invalid_argument(std::string(op) + ": operands on different devices");
  }
}

// GEMMs and row-wise reductions absorb the stride between rows; they cannot
// absorb a gap between elements within a row. Returns the row stride to pass
// down
[[maybe_unused]] static int64_t row_stride_of(const Tensor& t, const char* op) {
  const int r = t.shape().rank();
  if (r < 2) {
    throw std::invalid_argument(std::string(op) + ": needs rank >= 2");
  }
  if (t.stride(r - 1) != 1) {
    throw std::invalid_argument(std::string(op) +
        ": innermost axis must be contiguous (call .pack() first)");
  }
  return t.stride(r - 2);
}

// In-place ops write through a pointer they did not allocate, so a strided
// destination would corrupt whatever lives between its rows.
[[maybe_unused]] static void require_contiguous(const Tensor& t, const char* op) {
  if (!t.is_contiguous()) {
    throw std::invalid_argument(std::string(op) + ": operand must be contiguous");
  }
}

[[maybe_unused]] static void same_shape(const Tensor& a, const Tensor& b, const char* op) {
  if (a.shape() != b.shape()) {
    throw std::invalid_argument(std::string(op) + ": " + a.shape().str() + " and " +
                                b.shape().str() + " must have the same shape");
  }
}

}  // namespace nn::ops
