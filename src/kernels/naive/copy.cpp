#include "naive_kernels.h"

#include "../strided_index.h"

namespace nn::kernels {

namespace {

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

void naive_unpack(const Stream&, const float* src,
                             float* dst, TensorView v, int64_t n) {
  for (int64_t i = 0; i < n; ++i) {
    dst[offset_of(v, i)] = src[i];
  }
}

}