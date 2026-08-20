#pragma once

struct ArgMax {
  float value;
  int index;
};

__device__ inline float shfl_down(float x, int d) {
  return __shfl_down_sync(0xffffffff, x, d);
}

__device__ inline ArgMax shfl_down(ArgMax x, int d) {
  x.value = __shfl_down_sync(0xffffffff, x.value, d);
  x.index = __shfl_down_sync(0xffffffff, x.index, d);
  return x;
}

struct Plus {
  __device__ inline float operator()(float a, float b) const {
    return a + b;
  }
};

struct Max {
  __device__ inline float operator()(float a, float b) const {
    return a > b ? a : b;
  }
};

struct MaxKeepLowestIndex {
  __device__ inline ArgMax operator()(ArgMax a, ArgMax b) const {
    if (a.value > b.value) {
      return a;
    } else if (a.value < b.value) {
      return b;
    } else {
      return (a.index < b.index) ? a : b;
    }
  }
};

template <class T, class Op>
__device__ inline T warp_reduce(T val, Op op) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    val = op(val, shfl_down(val, offset));
  }
  return val;
}

template <class T, class Op>
__device__ inline T block_reduce(T val, Op op, T identity) {
  __shared__ T shared[32]; // Assuming a maximum of 1024 threads per block
  const int lane = threadIdx.x % warpSize;
  const int warp_id = threadIdx.x / warpSize;

  val = warp_reduce(val, op);
  if (lane == 0) shared[warp_id] = val;
  __syncthreads();

  const int warps = (blockDim.x + warpSize - 1) / warpSize;
  val = (threadIdx.x < warps) ? shared[threadIdx.x] : identity;
  if (warp_id == 0) val = warp_reduce(val, op);
  return val;
}