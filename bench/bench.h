#pragma once

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>

namespace nn::bench {

// repeats fn for at least budget seconds
// returns the median time
template <class F>
double time_ns(F&& fn, double budget = 0.5) {
  using clock = std::chrono::steady_clock;
  fn();

  std::vector<double> samples;
  const auto start = clock::now();
  while (std::chrono::duration<double>(clock::now() - start).count() < budget) {
    const auto t0 = clock::now();
    fn();
    const auto t1 = clock::now();
    samples.push_back(std::chrono::duration<double, std::nano>(t1 - t0).count());
  }
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
}

// flops/ns is exactly GLOPS/s
inline void report(const char* name, double ns, double flops) {
  std::printf("%-22s %10.1f us %9.2f GFLOP/s\n", name, ns/1000.0, flops/ns);
}

inline void report(const char* name, double ns, const char* unit, double count) {
  std::printf("%-22s %10.1f us %9.2f ns/%s\n", name, ns/1000.0, ns/count, unit);
}

}