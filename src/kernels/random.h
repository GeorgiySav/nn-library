#pragma once

#include <cstdint>

// splitmix64-style hash used as the random number generator, compiled into
// both backends

#if defined(__CUDACC__)
#  define NN_RNG_INLINE __host__ __device__ inline
#else
#  define NN_RNG_INLINE inline
#endif

namespace nn::kernels {

NN_RNG_INLINE uint32_t random_bits(uint64_t seed, uint64_t counter) {
  uint64_t z = counter * 0x9E3779B97F4A7C15ULL + seed * 0xD1B54A32D192ED03ULL;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  z =  z ^ (z >> 31);
  return uint32_t(z >> 32);   // the high half is better mixed than the low half
}

// uniform float in [0, 1), built from the top 24 bits so it rounds evenly
// into a float's mantissa.
NN_RNG_INLINE float random_uniform(uint64_t seed, uint64_t counter) {
  return float(random_bits(seed, counter) >> 8) * (1.0f / 16777216.0f);
}

}  // namespace nn::kernels
