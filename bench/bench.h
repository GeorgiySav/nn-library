#pragma once

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>

#if defined(NN_WITH_CUDA)
#include <cuda_runtime.h>
#endif

#include <nn/core/device.h>

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

template <class F>
double time_ns_on(Device d, F&& fn, int reps = 1, double budget = 0.5) {
  using clock = std::chrono::steady_clock;
  const Stream& s = current_stream(d);

  fn();
  s.synchronize();

  std::vector<double> samples;
  const auto start = clock::now();
  while (std::chrono::duration<double>(clock::now() - start).count() < budget) {
    const auto t0 = clock::now();
    for (int r = 0; r < reps; ++r) fn();
    s.synchronize();
    const auto t1 = clock::now();
    samples.push_back(
      std::chrono::duration<double, std::nano>(t1 - t0).count() / reps);
  }
  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
}

// 0.0 means "unknown", which report_bandwidth already renders as a plain GB/s
// figure with no percent-of-peak column
inline double peak_bandwidth_gb_s(Device d) {
  if (d != Device::CUDA) return 0.0;
#if !defined(NN_WITH_CUDA)
  return 0.0;
#else
  int dev = 0, clock_khz = 0, bus_bits = 0;
  if (cudaGetDevice(&dev) != cudaSuccess) return 0.0;
  if (cudaDeviceGetAttribute(&clock_khz, cudaDevAttrMemoryClockRate, dev)
      != cudaSuccess) return 0.0;
  if (cudaDeviceGetAttribute(&bus_bits, cudaDevAttrGlobalMemoryBusWidth, dev)
      != cudaSuccess) return 0.0;
  return 2.0 * double(clock_khz) * 1e3 * (bus_bits / 8.0) / 1e9;
#endif
}

inline void report_bandwidth(const char* name, double ns, double bytes,
                             double peak_gb_s = 0.0) {
  const double gb_s = bytes / ns;
  if (peak_gb_s > 0.0) {
    std::printf("%-22s %10.1f us %9.2f GB/s %7.1f%% of peak\n",
                name, ns/1000.0, gb_s, 100.0 * gb_s / peak_gb_s);
  } else {
    std::printf("%-22s %10.1f us %9.2f GB/s\n", name, ns/1000.0, gb_s);
  }
}

// flops/ns is exactly GLOPS/s
inline void report(const char* name, double ns, double flops) {
  std::printf("%-22s %10.1f us %9.2f GFLOP/s\n", name, ns/1000.0, flops/ns);
}

inline void report(const char* name, double ns, const char* unit, double count) {
  std::printf("%-22s %10.1f us %9.2f ns/%s\n", name, ns/1000.0, ns/count, unit);
}

}