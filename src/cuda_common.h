#pragma once

#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

// err_ is underscored so a caller can write NN_CUDA_CHECK inside a scope that
// already has its own 'err'.
#define NN_CUDA_CHECK(expr) \
  do { \
    cudaError_t err_ = (expr); \
    if (err_ != cudaSuccess) { \
      throw std::runtime_error(std::string("CUDA: ") + #expr + ": " + cudaGetErrorString(err_)); \
    } \
  } while (0)

#ifdef NDEBUG
  #define NN_CUDA_CHECK_LAUNCH(stream) NN_CUDA_CHECK(cudaGetLastError())
#else
  // debug builds sync after every launch so a kernel error is caught at its
  // own call site instead of surfacing at some unrelated later sync point
  #define NN_CUDA_CHECK_LAUNCH(stream) \
    do {\
      NN_CUDA_CHECK(cudaGetLastError()); \
      NN_CUDA_CHECK(cudaStreamSynchronize(stream)); \
    } while (0)
#endif