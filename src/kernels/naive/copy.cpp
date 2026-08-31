#include "naive_kernels.h"

#include "../strided_index.h"

namespace nn::kernels {

namespace {

// reads src through the strided view v and writes it out densely into dst.
template<class T>
void pack_impl(const T* src, TensorView v, T* dst, int64_t n) {
  for (int64_t i = 0; i < n; ++i) {
    dst[i] = src[offset_of(v, i)];
  }
}

}

void naive_pack(const Stream&, const float* src, TensorView v,
                        float* dst, int64_t n) {
  pack_impl(src, v, dst, n);
}

void naive_pack_i32(const Stream&, const int32_t* src, TensorView v,
                            int32_t* dst, int64_t n) {
  pack_impl(src, v, dst, n);
}

namespace {

// reverse of pack_impl, reading src densely and scattering it into dst
// through the strided view v.
template<class T>
void unpack_impl(const T* src, T* dst, TensorView v, int64_t n) {
  for (int64_t i = 0; i < n; ++i) {
    dst[offset_of(v, i)] = src[i];
  }
}

}

void naive_unpack(const Stream&, const float* src,
                             float* dst, TensorView v, int64_t n) {
  unpack_impl(src, dst, v, n);
}

void naive_unpack_i32(const Stream&, const int32_t* src,
                               int32_t* dst, TensorView v, int64_t n) {
  unpack_impl(src, dst, v, n);
}

}