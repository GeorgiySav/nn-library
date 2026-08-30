// The two directions between a strided view and dense storage.

#include <nn/ops/ops.h>

#include <stdexcept>
#include <string>

#include <kernels/kernel_api.h>

#include "ops_common.h"

namespace nn::ops {

// Gathers src through its strides into dst, which must be dense. Tensor::
// contiguous() is this and nothing else.
void pack(const Tensor& dst, const Tensor& src) {
  same_device(dst, src, "pack");

  if (dst.shape() != src.shape()) {
    throw std::invalid_argument("pack: dst and src must have the same shape");
  }
  if (dst.dtype() != src.dtype()) {
    throw std::invalid_argument("pack: dst and src must have the same dtype");
  }
  if (!dst.is_contiguous()) {
    throw std::invalid_argument("pack: dst must be contiguous");
  }

  const auto& k = nn::kernels::kernels(dst.device());
  const Stream& s = current_stream(dst.device());
  const TensorView v = view_of(src);

  switch (src.dtype()) {
    case DType::F32:
      k.pack(s, src.device_ptr(), v, dst.device_ptr(), src.numel());
      break;
    case DType::I32:
      k.pack_i32(s, src.device_ptr_i32(), v, dst.device_ptr_i32(), src.numel());
      break;
  }
}

// Writes src out through dst's strides. dst is usually a window of a larger
// tensor, so this is the one place a kernel writes to a non-dense destination.
void unpack(Tensor& dst, const Tensor& src) {
  same_device(dst, src, "unpack");

  if (dst.shape() != src.shape()) {
    throw std::invalid_argument("unpack: dst and src must have the same shape");
  }
  if (dst.dtype() != src.dtype()) {
    throw std::invalid_argument("unpack: dst and src must have the same dtype");
  }

  const Tensor packed = src.pack();

  const auto& k = nn::kernels::kernels(dst.device());
  const Stream& s = current_stream(dst.device());
  const TensorView v = view_of(dst);

  switch (dst.dtype()) {
    case DType::F32:
      k.unpack(s, packed.device_ptr(), dst.device_ptr(), v, dst.numel());
      break;
    case DType::I32:
      k.unpack_i32(s, packed.device_ptr_i32(), dst.device_ptr_i32(), v, dst.numel());
      break;
  }
}

}  // namespace nn::ops
