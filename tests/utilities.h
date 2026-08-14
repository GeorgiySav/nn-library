#pragma once

#include <vector>
#include <nn/core/rng.h>

inline void fill_random(std::vector<float>& data, nn::Pcg32& rng) {
  for (auto& x : data) {
    x = rng.next_normal();
  }
}