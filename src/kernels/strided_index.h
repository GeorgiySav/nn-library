#pragma once

#include <nn/core/strides.h>

#if defined(__CUDACC__)
#  define NN_STRIDED_INLINE __host__ __device__ inline
#else
#  define NN_STRIDED_INLINE inline
#endif

namespace nn::kernels {

// Linear position in the view's own row-major order -> element offset from the
// pointer the caller passed. The view carries no offset of its own: device_ptr()
// already points at element zero.
NN_STRIDED_INLINE int64_t offset_of(const TensorView& v, int64_t i) {
  int64_t rem = i, off = 0;
  for (int a = v.rank - 1; a >= 1; --a) {
    off += (rem % v.shape[a]) * v.stride[a];
    rem /= v.shape[a];
  }
  return (v.rank > 0) ? off + rem * v.stride[0] : off;
}

}  // namespace nn::kernels
