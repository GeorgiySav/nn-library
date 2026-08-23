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

}

__global__ void unary_kernel(UnaryOp op, const float* X, TensorView vx,
                             float* Y, int64_t n) {
  for (int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
       i < n;
       i += int64_t(gridDim.x) * blockDim.x) {
    Y[i] = apply_unary(op, X[offset_of(vx, i)]);
  }
}

__global__ void unary_backward_kernel(UnaryOp op,
                                      const float* __restrict__ X, TensorView vx,
                                      const float* __restrict__ Y, TensorView vy,
                                      const float* __restrict__ G, TensorView vg,
                                      float* __restrict__ gX, int64_t n) {
  for (int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
       i < n;
       i += int64_t(gridDim.x) * blockDim.x) {
    gX[i] = apply_unary_backward(op, X[offset_of(vx, i)], Y[offset_of(vy, i)],
                                 G[offset_of(vg, i)]);
  }
}

__global__ void binary_kernel(BinaryOp op,
                              const float* A, TensorView va,
                              const float* B, TensorView vb,
                              float* C, int64_t n) {
  for (int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
       i < n;
       i += int64_t(gridDim.x) * blockDim.x) {
    C[i] = apply_binary(op, A[offset_of(va, i)], B[offset_of(vb, i)]);
  }
}

__global__ void binary_backward_kernel(BinaryOp op, int side,
                                       const float* __restrict__ A, TensorView va,
                                       const float* __restrict__ B, TensorView vb,
                                       const float* __restrict__ C, TensorView vc,
                                       const float* __restrict__ G, TensorView vg,
                                       float* __restrict__ out, int64_t n) {
  for (int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
       i < n;
       i += int64_t(gridDim.x) * blockDim.x) {
    out[i] = apply_binary_backward(op, side, A[offset_of(va, i)], B[offset_of(vb, i)],
                                   C[offset_of(vc, i)], G[offset_of(vg, i)]);
  }
}

__global__ void scalar_kernel(ScalarOp op, float k,
                              const float* X, TensorView vx,
                              float* Y, int64_t n) {
  for (int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
       i < n;
       i += int64_t(gridDim.x) * blockDim.x) {
    Y[i] = apply_scalar(op, X[offset_of(vx, i)], k);
  }
}

__global__ void scalar_backward_kernel(ScalarOp op, float k,
                                       const float* __restrict__ X, TensorView vx,
                                       const float* __restrict__ Y, TensorView vy,
                                       const float* __restrict__ G, TensorView vg,
                                       float* __restrict__ gX, int64_t n) {
  for (int64_t i = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
       i < n;
       i += int64_t(gridDim.x) * blockDim.x) {
    gX[i] = apply_scalar_backward(op, X[offset_of(vx, i)], Y[offset_of(vy, i)],
                                  G[offset_of(vg, i)], k);
  }
}

__global__ void fill_kernel(float v, float* X, int64_t n) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
       i < n;
       i += int64_t(gridDim.x) * blockDim.x) {
    X[i] = v;
  }
}

__global__ void fill_from_kernel(const float* __restrict__ src, float* X, int64_t n) {
  const float v = *src;
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
       i < n;
       i += int64_t(gridDim.x) * blockDim.x) {
    X[i] = v;
  }
}

__global__ void axpy_kernel(float alpha, const float* X, float* Y, int64_t n) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
       i < n;
       i += int64_t(gridDim.x) * blockDim.x) {
    Y[i] += alpha * X[i];
  }
}

__global__ void adam_step_kernel(float* __restrict__ p, const float* __restrict__ g,
                                 float* __restrict__ m, float* __restrict__ v,
                                 float lr, float b1, float b2, float eps, float wd,
                                 float bc1, float bc2, int64_t n) {
  for (int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
       i < n;
       i += int64_t(gridDim.x) * blockDim.x) {
    const float gi = g[i];
    const float mi = b1 * m[i] + (1.0f - b1) * gi;
    const float vi = b2 * v[i] + (1.0f - b2) * gi * gi;

    m[i] = mi;
    v[i] = vi;

    p[i] = p[i] * (1.0f - lr * wd) - lr * (mi / bc1) / (sqrtf(vi / bc2) + eps);
  }
}

void cuda_unary(const Stream& s, UnaryOp op, const float* X, TensorView vx,
                float* Y, int64_t n) {
  launch_elementwise(s, n, unary_kernel, op, X, vx, Y);
}
void cuda_unary_backward(const Stream& s, UnaryOp op,
                         const float* X, TensorView vx,
                         const float* Y, TensorView vy,
                         const float* G, TensorView vg,
                         float* gX, int64_t n) {
  launch_elementwise(s, n, unary_backward_kernel, op, X, vx, Y, vy, G, vg, gX);
}
void cuda_binary(const Stream& s, BinaryOp op,
                 const float* A, TensorView va,
                 const float* B, TensorView vb,
                 float* C, int64_t n) {
  launch_elementwise(s, n, binary_kernel, op, A, va, B, vb, C);
}
void cuda_binary_backward(const Stream& s, BinaryOp op, int side,
                          const float* A, TensorView va,
                          const float* B, TensorView vb,
                          const float* C, TensorView vc,
                          const float* G, TensorView vg,
                          float* out, int64_t n) {
  launch_elementwise(s, n, binary_backward_kernel, op, side, A, va, B, vb, C, vc, G, vg, out);
}
void cuda_scalar(const Stream& s, ScalarOp op, float k,
                 const float* X, TensorView vx, float* Y, int64_t n) {
  launch_elementwise(s, n, scalar_kernel, op, k, X, vx, Y);
}
void cuda_scalar_backward(const Stream& s, ScalarOp op, float k,
                          const float* X, TensorView vx,
                          const float* Y, TensorView vy,
                          const float* G, TensorView vg,
                          float* gX, int64_t n) {
  launch_elementwise(s, n, scalar_backward_kernel, op, k, X, vx, Y, vy, G, vg, gX);
}

void cuda_fill(const Stream& s, float v, float* X, int64_t n) {
  launch_elementwise(s, n, fill_kernel, v, X);
}
void cuda_fill_from(const Stream& s, const float* src, float* X, int64_t n) {
  launch_elementwise(s, n, fill_from_kernel, src, X);
}
void cuda_axpy(const Stream& s, float alpha, const float* X, float* Y, int64_t n) {
  launch_elementwise(s, n, axpy_kernel, alpha, X, Y);
}
void cuda_adam_step(const Stream& s, float* p, const float* g, float* m, float* v,
                    float lr, float b1, float b2, float eps, float wd,
                    float bc1, float bc2, int64_t n) {
  launch_elementwise(s, n, adam_step_kernel, p, g, m, v, lr, b1, b2, eps, wd, bc1, bc2);
}

}
