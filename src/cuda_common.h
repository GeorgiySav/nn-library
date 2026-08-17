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
  #define NN_CUDA_CHECK_LAUNCH(stream) \
    do {\
      NN_CUDA_CHECK(cudaGetLastError()); \
      NN_CUDA_CHECK(cudaStreamSynchronize(stream)); \
    } while (0)
#endif