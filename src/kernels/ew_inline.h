#pragma once

// NN_EW_INLINE marks a function that must compile identically as plain host
// C++ (included by the naive backend) and as CUDA __host__ __device__ code
// (included by the CUDA backend). one definition, both backends, so they
// can never disagree about what an op computes.
#if defined(__CUDACC__)
#  define NN_EW_INLINE __host__ __device__ inline
#else
#  include <cmath>
#  define NN_EW_INLINE inline
#endif
