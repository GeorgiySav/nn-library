// Row gather from a [V, D] table, and its scatter-add backward.

#include <nn/ops/ops.h>

#include <stdexcept>
#include <string>

#include <nn/kernels/kernel_api.h>

#include "ops_common.h"

namespace nn::ops {

Tensor embedding(const Tensor& weight, const Tensor& idx) {
  same_device(weight, idx, "embedding");

  if (weight.shape().rank() != 2) {
    throw std::invalid_argument("embedding: weight must be [V, D]");
  }
  if (idx.dtype() != DType::I32) {
    throw std::invalid_argument("embedding: indices must be I32");
  }
  if (idx.shape().rank() + 1 > kMaxShapeRank) {
    throw std::invalid_argument("embedding: indices have too many axes");
  }
  require_contiguous(weight, "embedding (weight)");

  const int V = weight.shape().dim(0);
  const int D = weight.shape().dim(1);
  const Tensor flat_idx = idx.pack();

  int dims[kMaxShapeRank] = {0};
  const int r = idx.shape().rank();
  for (int i = 0; i < r; ++i) dims[i] = idx.shape().dim(i);
  dims[r] = D;

  Tensor out(Shape(std::span<const int>(dims, r + 1)), weight.device(), weight.dtype());
  const auto& k = nn::kernels::kernels(weight.device());
  k.embedding(current_stream(weight.device()), weight.device_ptr(),
              flat_idx.device_ptr_i32(), out.device_ptr(), idx.numel(), D, V);
  return out;
}

Tensor embedding_backward(const Tensor& g, const Tensor& idx, int V) {
  same_device(g, idx, "embedding_backward");

  const int r = g.shape().rank();
  if (r < 1) throw std::invalid_argument("embedding_backward: gradient has no axes");
  const int D = g.shape().dim(r - 1);
  if (g.numel() != idx.numel() * D) {
    throw std::invalid_argument("embedding_backward: gradient does not match the indices");
  }

  const Tensor dense_g = g.pack();
  const Tensor flat_idx = idx.pack();

  Tensor gw = Tensor::zeros(Shape{V, D}, g.device(), g.dtype());
  const auto& k = nn::kernels::kernels(g.device());
  k.embedding_backward(current_stream(g.device()), dense_g.device_ptr(),
                       flat_idx.device_ptr_i32(), gw.device_ptr(), idx.numel(), D, V);
  return gw;
}

}  // namespace nn::ops
