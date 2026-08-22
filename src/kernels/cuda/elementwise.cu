#include "cuda_kernels.h"

#include "../../cuda_common.h"
#include "../strided_index.h"

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

struct ReluOp    { __device__ float operator()(float x) const { return fmaxf(x, 0.0f); } };
struct AddOp     { __device__ float operator()(float a, float b) const { return a + b; } };
struct ReluBwdOp { __device__ float operator()(float x, float g) const { return x > 0.0f ? g : 0.0f; } };

}

template <class Op>
__global__ void map1_strided_kernel(const float* __restrict__ x, TensorView v,
                                    float* __restrict__ y, int64_t n) {
  for (int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
       i < n;
       i += int64_t(gridDim.x) * blockDim.x) {
    y[i] = Op{}(x[offset_of(v, i)]);
  }
}

template <class Op>
__global__ void map2_strided_kernel(const float* __restrict__ a, TensorView va,
                                    const float* __restrict__ b, TensorView vb,
                                    float* __restrict__ c, int64_t n) {
  for (int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
       i < n;
       i += int64_t(gridDim.x) * blockDim.x) {
    c[i] = Op{}(a[offset_of(va, i)], b[offset_of(vb, i)]);
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

__global__ void adam_step_kernel(float* __restrict__ p, const float* __restrict__ g,
                                 float* __restrict__ m, float* __restrict__ v,
                                 float lr, float b1, float b2, float eps,
                                 float bc1, float bc2, int64_t n) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
       i < n;
       i += int64_t(gridDim.x) * blockDim.x) {
    const float gi = g[i];
    const float mi = b1 * m[i] + (1.0f - b1) * gi;
    const float vi = b2 * v[i] + (1.0f - b2) * gi * gi;

    m[i] = mi;
    v[i] = vi;

    p[i] -= lr * (mi / bc1) / (sqrtf(vi / bc2) + eps);
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
void cuda_relu_strided(const Stream& s, const float* X, TensorView v, float* Y, int64_t n) {
  launch_elementwise(s, n, map1_strided_kernel<ReluOp>, X, v, Y);
}
void cuda_relu_backward_strided(const Stream& s, const float* X, TensorView vx,
                                const float* gY, TensorView vg, float* gX, int64_t n) {
  launch_elementwise(s, n, map2_strided_kernel<ReluBwdOp>, X, vx, gY, vg, gX);
}
void cuda_add_strided(const Stream& s, const float* A, TensorView va,
                      const float* B, TensorView vb, float* C, int64_t n) {
  launch_elementwise(s, n, map2_strided_kernel<AddOp>, A, va, B, vb, C);
}
void cuda_adam_step(const Stream& s, float* p, const float* g, float* m, float* v,
                    float lr, float b1, float b2, float eps, float bc1, float bc2, int64_t n) {
  launch_elementwise(s, n, adam_step_kernel, p, g, m, v, lr, b1, b2, eps, bc1, bc2);
}

}