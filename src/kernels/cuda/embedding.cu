#include "cuda_kernels.h"

#include "../../cuda_common.h"

namespace nn::kernels {

namespace {

constexpr int kBlock = 256;
constexpr int kMaxGrid = 4096;

inline int grid_for(int64_t n) {
  return int(std::min<int64_t>((n + kBlock - 1) / kBlock, kMaxGrid));
}

}  // namespace

__global__ void embedding_kernel(const float* __restrict__ W,
                                 const int32_t* __restrict__ idx,
                                 float* __restrict__ Y, int64_t total, int D, int V) {
  for (int64_t t = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
       t < total;
       t += int64_t(gridDim.x) * blockDim.x) {
    const int64_t row = t / D;
    const int d = int(t - row * D);
    const int32_t v = idx[row];
    Y[t] = (v < 0 || v >= V) ? 0.0f : W[int64_t(v) * D + d];
  }
}

__global__ void embedding_backward_kernel(const float* __restrict__ G,
                                          const int32_t* __restrict__ idx,
                                          float* __restrict__ gW,
                                          int64_t total, int D, int V) {
  for (int64_t t = blockIdx.x * int64_t(blockDim.x) + threadIdx.x;
       t < total;
       t += int64_t(gridDim.x) * blockDim.x) {
    const int64_t row = t / D;
    const int d = int(t - row * D);
    const int32_t v = idx[row];
    if (v < 0 || v >= V) continue;
    atomicAdd(&gW[int64_t(v) * D + d], G[t]);
  }
}

void cuda_embedding(const Stream& s, const float* W, const int32_t* idx,
                    float* Y, int64_t n_idx, int D, int V) {
  const int64_t total = n_idx * D;
  if (total == 0) return;

  auto stream = static_cast<cudaStream_t>(s.handle);
  embedding_kernel<<<grid_for(total), kBlock, 0, stream>>>(W, idx, Y, total, D, V);
  NN_CUDA_CHECK_LAUNCH(stream);
}

void cuda_embedding_backward(const Stream& s, const float* G, const int32_t* idx,
                             float* gW, int64_t n_idx, int D, int V) {
  const int64_t total = n_idx * D;
  if (total == 0) return;

  auto stream = static_cast<cudaStream_t>(s.handle);
  embedding_backward_kernel<<<grid_for(total), kBlock, 0, stream>>>(G, idx, gW, total, D, V);
  NN_CUDA_CHECK_LAUNCH(stream);
}

}
