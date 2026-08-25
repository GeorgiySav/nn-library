#include "bench.h"

#include <cstdio>
#include <vector>

#include <nn/core/tensor.h>
#include <nn/ops/ops.h>

using nn::Device;
using nn::Tensor;

int main() {
  const int64_t sizes[] = {int64_t(1) << 20, int64_t(1) << 24, int64_t(1) << 26};

  std::printf("%-10s %-6s %10s %10s %10s %8s\n",
              "device", "n", "Sum(ns)", "SumSq", "SumAbs", "sq/sum");

  std::vector<Device> devices{Device::CPU};
  if (nn::cuda_device_count() > 0) devices.push_back(Device::CUDA);

  for (Device d : devices) {
    for (int64_t n : sizes) {
      std::vector<float> h(static_cast<size_t>(n), 0.0f);
      for (size_t i = 0; i < h.size(); ++i) h[i] = 1e-3f * float(i % 977) - 0.4f;
      const Tensor x = Tensor::from(h, nn::Shape({int(n)}), d);

      auto run = [&](nn::ops::Accum a) {
        return nn::bench::time_ns_on(d, [&] { nn::ops::sum_all(x, a); }, 1, 0.4);
      };

      const double s  = run(nn::ops::Accum::Sum);
      const double sq = run(nn::ops::Accum::SumSq);
      const double ab = run(nn::ops::Accum::SumAbs);

      std::printf("%-10s %-6lldM %10.0f %10.0f %10.0f %8.3f\n",
                  d == Device::CPU ? "cpu" : "cuda", (long long)(n >> 20),
                  s, sq, ab, sq / s);
    }
  }
  return 0;
}
