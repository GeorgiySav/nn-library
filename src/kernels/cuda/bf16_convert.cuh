#pragma once

#include <cuda_bf16.h>

#include <nn/core/bf16.h>

namespace nn::kernels {

__device__ __forceinline__ float to_f32(float v) { return v; }
__device__ __forceinline__ float to_f32(__nv_bfloat16 v) { return __bfloat162float(v); }

template <typename T>
__device__ __forceinline__ T from_f32(float v);

template <>
__device__ __forceinline__ float from_f32<float>(float v) { return v; };

template <>
__device__ __forceinline__ __nv_bfloat16 from_f32<__nv_bfloat16>(float v) {
  return __float2bfloat16(v);
}


inline __nv_bfloat16* bf16_ptr(bf16* ptr) {
  return reinterpret_cast<__nv_bfloat16*>(ptr);
}
inline const __nv_bfloat16* bf16_ptr(const bf16* ptr) {
  return reinterpret_cast<const __nv_bfloat16*>(ptr);
}

}