#pragma once

#include <nn/core/tensorview.h>

#if defined(__CUDACC__)
#  define NN_STRIDED_INLINE __host__ __device__ inline
#else
#  define NN_STRIDED_INLINE inline
#endif

namespace nn::kernels {

// converts a linear position i, as if the view were dense row-major, into
// the actual element offset from the pointer the caller passed, honoring
// the view's real strides. the view carries no offset of its own, since
// device_ptr() already points at element zero.
//
// walks axes from innermost to outermost, peeling off each axis's index
// with a mod/div pair (rem % shape[a] is the index along axis a, rem /=
// shape[a] moves to the next axis out) and accumulating index * stride[a].
// the outermost axis (a == 0) is handled after the loop without a mod,
// since whatever remains of rem at that point is already its index.
NN_STRIDED_INLINE int64_t offset_of(const TensorView& v, int64_t i) {
  int64_t rem = i, off = 0;
  for (int a = v.rank - 1; a >= 1; --a) {
    off += (rem % v.shape[a]) * v.stride[a];
    rem /= v.shape[a];
  }
  return (v.rank > 0) ? off + rem * v.stride[0] : off;
}

}  // namespace nn::kernels
