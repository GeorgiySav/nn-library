#pragma once

#include <vector>
#include <nn/core/device.h>
#include <nn/core/rng.h>

// Tests that bypass ops:: and call kernels directly still have to name a
// stream. There is only one on CPU, so this keeps the call sites readable.
inline const nn::Stream& cpu_stream() {
  return nn::current_stream(nn::Device::CPU);
}

inline void fill_random(std::vector<float>& data, nn::Pcg32& rng) {
  for (auto& x : data) {
    x = rng.next_normal();
  }
}