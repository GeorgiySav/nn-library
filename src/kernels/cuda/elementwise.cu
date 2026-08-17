#include "cuda_kernels.h"

#include "../../cuda_common.h"

namespace nn::kernels {

namespace {

constexpr int kElementwiseBlock = 256;
constexpr int kElementwiseMaxGrid = 4096;

inline int elementwise_grid(int64_t n, int block = kElementwiseBlock) {
  return int(std::min<int64_t>((n + block - 1) / block, kElementwiseMaxGrid));
}

template <typename Kernel, typename... Args>
void launch_elementwise(const Stream& s, int64_t n, Kernel kernel, Args... args) {
  if (n == 0) return;

  auto stream = static_cast<cudaStream_t>(s.handle);
  kernel<<<elementwise_grid(n), kElementwiseBlock, 0, stream>>>(args..., n);
  NN_CUDA_CHECK_LAUNCH(stream);
}

}

__global__ void fill_kernel(float v, float* X, int64_t n) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
       i < n;
       i += int64_t(gridDim.x) * blockDim.x) {
    X[i] = v;
  }
}

__global__ void scale_kernel(float alpha, float* X, int64_t n) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
       i < n;
       i += int64_t(gridDim.x) * blockDim.x) {
    X[i] *= alpha;
  }
}

__global__ void axpy_kernel(float alpha, const float* X, float* Y, int64_t n) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
       i < n;
       i += int64_t(gridDim.x) * blockDim.x) {
    Y[i] += alpha * X[i];
  }
}

__global__ void add_kernel(const float* A, const float* B, float* C, int64_t n) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
       i < n;
       i += int64_t(gridDim.x) * blockDim.x) {
    C[i] = A[i] + B[i];
  }
}

__global__ void relu_kernel(const float* X, float* Y, int64_t n) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
       i < n;
       i += int64_t(gridDim.x) * blockDim.x) {
    Y[i] = fmaxf(X[i], 0.0f);
  }
}

__global__ void relu_backward_kernel(const float* X, const float* gY, float* gX, int64_t n) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
       i < n;
       i += int64_t(gridDim.x) * blockDim.x) {
    gX[i] = X[i] > 0.0f ? gY[i] : 0.0f;
  }
}

void cuda_fill(const Stream& s, float v, float* X, int64_t n) {
  launch_elementwise(s, n, fill_kernel, v, X);
}
void cuda_scale(const Stream& s, float alpha, float* X, int64_t n) {
  launch_elementwise(s, n, scale_kernel, alpha, X);
}
void cuda_axpy(const Stream& s, float alpha, const float* X, float* Y, int64_t n) {
  launch_elementwise(s, n, axpy_kernel, alpha, X, Y);
}
void cuda_add(const Stream& s, const float* A, const float* B, float* C, int64_t n) {
  launch_elementwise(s, n, add_kernel, A, B, C);
}
void cuda_relu(const Stream& s, const float* X, float* Y, int64_t n) {
  launch_elementwise(s, n, relu_kernel, X, Y);
}
void cuda_relu_backward(const Stream& s , const float* X, const float* gY, float* gX, int64_t n) {
  launch_elementwise(s, n, relu_backward_kernel, X, gY, gX);
}

}