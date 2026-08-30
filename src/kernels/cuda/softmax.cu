#include "cuda_kernels.h"

#include <cfloat>

#include "../../cuda_common.h"
#include "cuda_reduce.cuh"

namespace nn::kernels {

__global__ void softmax_ce_kernel(const float* logits, const int32_t* labels,
                                  float* loss_out, float* probs, int M, int N, int64_t sz) {
  // one block per row
  // M blocks, each with 256 threads
  // striding across N columns
  // sz is the logits row stride (== N when dense); probs is always dense
  const float max_identity = -FLT_MAX;
  
  __shared__ float sum_bcast;
  __shared__ float max_bcast;
  __shared__ int32_t label;

  for (int64_t row = blockIdx.x; row < M; row += gridDim.x) {
    const float* z = logits + row * sz;
    float* p = probs + row * N;
    
    // Max
    float max = -FLT_MAX;
    for (int c = threadIdx.x; c < N; c += blockDim.x) {
      if (z[c] > max) max = z[c];
    }
    
    max = block_reduce(max, Max(), max_identity); 
    if (threadIdx.x == 0) {
      max_bcast = max;
      label = labels[row];
    }
    __syncthreads();
    max = max_bcast;
    
    // Sum of exp
    float sum = 0.0f;
    for (int c = threadIdx.x; c < N; c += blockDim.x) {
      float exp_z = expf(z[c] - max);
      p[c] = exp_z;
      sum += exp_z;
    }
    
    sum = block_reduce(sum, Plus(), 0.0f);
    if (threadIdx.x == 0) sum_bcast = sum;
    __syncthreads();
    sum = sum_bcast;

    // Normalise
    const float inv_s = 1.0f / sum;
    for (int c = threadIdx.x; c < N; c += blockDim.x) {
      p[c] *= inv_s;
    }

    if (threadIdx.x == 0)
      atomicAdd(loss_out,
        -((z[label] - max) - logf(sum)) / float(M)
      );
    __syncthreads();
  } 
}

__global__ void softmax_ce_backward_kernel(const float* probs, const int32_t* labels,
                                const float* g_loss, float* g_logits, int M, int N, int64_t sp) {

  const float scale = *g_loss / float(M);

  for (int64_t row = blockIdx.x; row < M; row += gridDim.x) {
    const float* p = probs + row * sp;
    float* g = g_logits + row * N;
    const int32_t label = labels[row];

    for (int64_t col = threadIdx.x; col < N; col += blockDim.x) {
      g[col] = scale * p[col];
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      g[label] -= scale;
    }
  }

}

void cuda_softmax_ce(const Stream& s, const float* logits, const int32_t* labels,
                     float* loss_out, float* probs, int M, int N, int64_t sz) {
  if (M == 0 || N == 0) return;

  auto stream = static_cast<cudaStream_t>(s.handle);
  constexpr int block{256};
  constexpr int kMaxGrid{4096};
  int grid = std::min(M, kMaxGrid);

  cudaMemsetAsync(loss_out, 0, sizeof(float), stream);

  softmax_ce_kernel<<<grid, block, 0, stream>>>(logits, labels, loss_out, probs, M, N, sz);
  NN_CUDA_CHECK_LAUNCH(stream);
}

void cuda_softmax_ce_backward(const Stream& s, const float* probs, const int32_t* labels,
                              const float* g_loss, float* g_logits, int M, int N, int64_t sp) {
  if (M == 0 || N == 0) return;

  auto stream = static_cast<cudaStream_t>(s.handle);
  constexpr int block{256};
  constexpr int kMaxGrid{4096};
  int grid = std::min(M, kMaxGrid);

  softmax_ce_backward_kernel<<<grid, block, 0, stream>>>(probs, labels, g_loss, g_logits, M, N, sp);
  NN_CUDA_CHECK_LAUNCH(stream);
}

__global__ void softmax_ce_weighted_kernel(const float* logits, const int32_t* labels,
                                  const float* weights, float* loss_out, float* probs,
                                  int M, int N, int64_t sz) {
  // Same as softmax_ce_kernel, but each row's contribution is scaled by
  // weights[row].
  const float max_identity = -FLT_MAX;

  __shared__ float sum_bcast;
  __shared__ float max_bcast;
  __shared__ int32_t label;

  for (int64_t row = blockIdx.x; row < M; row += gridDim.x) {
    const float* z = logits + row * sz;
    float* p = probs + row * N;

    // Max
    float max = -FLT_MAX;
    for (int c = threadIdx.x; c < N; c += blockDim.x) {
      if (z[c] > max) max = z[c];
    }

    max = block_reduce(max, Max(), max_identity);
    if (threadIdx.x == 0) {
      max_bcast = max;
      label = labels[row];
    }
    __syncthreads();
    max = max_bcast;

    // Sum of exp
    float sum = 0.0f;
    for (int c = threadIdx.x; c < N; c += blockDim.x) {
      float exp_z = expf(z[c] - max);
      p[c] = exp_z;
      sum += exp_z;
    }

    sum = block_reduce(sum, Plus(), 0.0f);
    if (threadIdx.x == 0) sum_bcast = sum;
    __syncthreads();
    sum = sum_bcast;

    // Normalise
    const float inv_s = 1.0f / sum;
    for (int c = threadIdx.x; c < N; c += blockDim.x) {
      p[c] *= inv_s;
    }

    if (threadIdx.x == 0)
      atomicAdd(loss_out,
        -weights[row] * ((z[label] - max) - logf(sum)) / float(M)
      );
    __syncthreads();
  }
}

__global__ void softmax_ce_weighted_backward_kernel(const float* probs, const int32_t* labels,
                                const float* weights, const float* g_loss, float* g_logits,
                                int M, int N, int64_t sp) {

  const float base = *g_loss / float(M);

  for (int64_t row = blockIdx.x; row < M; row += gridDim.x) {
    const float* p = probs + row * sp;
    float* g = g_logits + row * N;
    const int32_t label = labels[row];
    const float scale = weights[row] * base;

    for (int64_t col = threadIdx.x; col < N; col += blockDim.x) {
      g[col] = scale * p[col];
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      g[label] -= scale;
    }
  }

}

void cuda_softmax_ce_weighted(const Stream& s, const float* logits, const int32_t* labels,
                     const float* weights, float* loss_out, float* probs, int M, int N, int64_t sz) {
  if (M == 0 || N == 0) return;

  auto stream = static_cast<cudaStream_t>(s.handle);
  constexpr int block{256};
  constexpr int kMaxGrid{4096};
  int grid = std::min(M, kMaxGrid);

  cudaMemsetAsync(loss_out, 0, sizeof(float), stream);

  softmax_ce_weighted_kernel<<<grid, block, 0, stream>>>(logits, labels, weights, loss_out, probs, M, N, sz);
  NN_CUDA_CHECK_LAUNCH(stream);
}

void cuda_softmax_ce_weighted_backward(const Stream& s, const float* probs, const int32_t* labels,
                              const float* weights, const float* g_loss, float* g_logits, int M, int N, int64_t sp) {
  if (M == 0 || N == 0) return;

  auto stream = static_cast<cudaStream_t>(s.handle);
  constexpr int block{256};
  constexpr int kMaxGrid{4096};
  int grid = std::min(M, kMaxGrid);

  softmax_ce_weighted_backward_kernel<<<grid, block, 0, stream>>>(probs, labels, weights, g_loss, g_logits, M, N, sp);
  NN_CUDA_CHECK_LAUNCH(stream);
}

__global__ void softmax_rows_kernel(const float* __restrict__ X, float* __restrict__ Y,
                                    int M, int N, int64_t sx) {
  __shared__ float bcast;

  for (int64_t row = blockIdx.x; row < M; row += gridDim.x) {
    const float* x = X + row * sx;
    float* y = Y + row * N;

    float m = -FLT_MAX;
    for (int c = threadIdx.x; c < N; c += blockDim.x) {
      if (x[c] > m) m = x[c];
    }
    m = block_reduce(m, Max(), -FLT_MAX);
    if (threadIdx.x == 0) bcast = m;
    __syncthreads();
    m = bcast;

    float sum = 0.0f;
    for (int c = threadIdx.x; c < N; c += blockDim.x) {
      const float e = expf(x[c] - m);
      y[c] = e;
      sum += e;
    }
    sum = block_reduce(sum, Plus(), 0.0f);
    if (threadIdx.x == 0) bcast = sum;
    __syncthreads();

    const float inv_s = 1.0f / bcast;
    for (int c = threadIdx.x; c < N; c += blockDim.x) y[c] *= inv_s;
    __syncthreads();   // bcast is reused by the next row this block takes
  }
}

__global__ void softmax_rows_backward_kernel(const float* __restrict__ Y,
                                             const float* __restrict__ G,
                                             float* __restrict__ gX,
                                             int M, int N, int64_t sy, int64_t sg) {
  __shared__ float dot_bcast;

  for (int64_t row = blockIdx.x; row < M; row += gridDim.x) {
    const float* y = Y + row * sy;
    const float* g = G + row * sg;
    float* out = gX + row * N;

    float dot = 0.0f;
    for (int c = threadIdx.x; c < N; c += blockDim.x) dot += y[c] * g[c];
    dot = block_reduce(dot, Plus(), 0.0f);
    if (threadIdx.x == 0) dot_bcast = dot;
    __syncthreads();
    dot = dot_bcast;

    for (int c = threadIdx.x; c < N; c += blockDim.x) out[c] = y[c] * (g[c] - dot);
    __syncthreads();
  }
}

void cuda_softmax_rows(const Stream& s, const float* X, float* Y,
                       int M, int N, int64_t sx) {
  if (M == 0 || N == 0) return;

  auto stream = static_cast<cudaStream_t>(s.handle);
  constexpr int block{256};
  constexpr int kMaxGrid{4096};
  const int grid = std::min(M, kMaxGrid);

  softmax_rows_kernel<<<grid, block, 0, stream>>>(X, Y, M, N, sx);
  NN_CUDA_CHECK_LAUNCH(stream);
}

void cuda_softmax_rows_backward(const Stream& s, const float* Y, const float* G,
                                float* gX, int M, int N, int64_t sy, int64_t sg) {
  if (M == 0 || N == 0) return;

  auto stream = static_cast<cudaStream_t>(s.handle);
  constexpr int block{256};
  constexpr int kMaxGrid{4096};
  const int grid = std::min(M, kMaxGrid);

  softmax_rows_backward_kernel<<<grid, block, 0, stream>>>(Y, G, gX, M, N, sy, sg);
  NN_CUDA_CHECK_LAUNCH(stream);
}

}