#include "cuda_kernels.h"

#include "../../cuda_common.h"
#include "../strided_index.h"

namespace nn::kernels {

namespace {

constexpr int kBlock = 256;
constexpr int kMaxGrid = 4096;

template<class T>
__global__ void copy_strided_kernel(const T* __restrict__ src, TensorView v,
                                    T* __restrict__ dst, int64_t n) {
  for (int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
       i < n;
       i += int64_t(gridDim.x) * blockDim.x) {
    dst[i] = src[offset_of(v, i)];
  }
}

template<class T>
void launch(const Stream& s, const T* src, TensorView v, T* dst, int64_t n) {
  if (n == 0) return;
  auto stream = static_cast<cudaStream_t>(s.handle);
  const int grid = int(
    std::min<int64_t>((n + kBlock - 1) / kBlock, kMaxGrid)
  );
  copy_strided_kernel<T><<<grid, kBlock, 0, stream>>>(src, v, dst, n);
  NN_CUDA_CHECK_LAUNCH(stream);
}

}

void cuda_copy_strided(const Stream& s, const float* src, TensorView v,
                       float* dst, int64_t n) {
  launch(s, src, v, dst, n);
}

void cuda_copy_strided_i32(const Stream& s, const int32_t* src, TensorView v,
                           int32_t* dst, int64_t n) {
  launch(s, src, v, dst, n);
}

}