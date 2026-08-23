#pragma once

#include <cstdint>
#include <cmath>
#include <atomic>

namespace nn {

class Pcg32 {
public:
  explicit Pcg32(uint64_t seed = 0) {
    state_ = 0;
    inc_ = (seed << 1u) | 1u;

    next();
    state_ += seed;
    next();
  }

  uint32_t next_uint32() {
    uint64_t oldstate = state_;
    next();
    uint32_t xorshifted = static_cast<uint32_t>(((oldstate >> 18u) ^ oldstate) >> 27u);
    uint32_t rot = static_cast<uint32_t>(oldstate >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
  }

  float next_uniform() { // [0, 1)
    return static_cast<float>(next_uint32()) / static_cast<float>(UINT32_MAX);
  }

  float next_normal() { // mean = 0, stddev = 1
    if (has_spare_) {
      has_spare_ = false;
      return spare_;
    }

    float u1;
    do {
      u1 = next_uniform();
    } while (u1 <= 0.0f);

    const float u2 = next_uniform();
    const float r = std::sqrt(-2.0f * std::log(u1));
    const float theta = 2.0f * 3.141592653589793 * u2;

    spare_ = r * std::sin(theta);
    has_spare_ = true;
    return r * std::cos(theta);
  }

private:
  uint64_t state_, inc_;
  bool has_spare_ = false;
  float spare_;

  void next() {
    state_ = state_ * 6364136223846793005ULL + inc_;
  }
};


// The global counter every stochastic op draws from
namespace detail {
inline std::atomic<uint64_t>& rng_seed() {
  static std::atomic<uint64_t> s{0x853c49e6748fea9bULL};
  return s;
}
inline std::atomic<uint64_t>& rng_counter() {
  static std::atomic<uint64_t> c{0};
  return c;
}
}  // namespace detail

inline void manual_seed(uint64_t seed) {
  detail::rng_seed().store(seed);
  detail::rng_counter().store(0);
}

inline uint64_t random_seed()    { return detail::rng_seed().load(); }
inline uint64_t random_counter() { return detail::rng_counter().load(); }

inline void set_random_state(uint64_t seed, uint64_t counter) {
  detail::rng_seed().store(seed);
  detail::rng_counter().store(counter);
}

inline uint64_t reserve_random(int64_t n) {
  return detail::rng_counter().fetch_add(uint64_t(n < 0 ? 0 : n));
}

}