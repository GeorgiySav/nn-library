#include "bench.h"

#include <algorithm>
#include <cstdio>
#include <vector>

#include <nn/core/tensor.h>
#include <nn/kernels/kernel_api.h>

using nn::Device;
using nn::Stream;
using nn::Tensor;

namespace {

const int64_t kSizes[] = {
  int64_t(1) << 12,   //   4K
  int64_t(1) << 16,   //  64K
  int64_t(1) << 20,   //   1M
  int64_t(1) << 24,   //  16M
  int64_t(1) << 26,   //  64M
};

int reps_for(int64_t n) {
  const int64_t target = int64_t(1) << 22;
  return int(std::clamp<int64_t>(target / std::max<int64_t>(n, 1), 1, 200));
}

const char* size_label(int64_t n) {
  static char buf[16];
  if (n >= (1 << 20))     std::snprintf(buf, sizeof buf, "%lldM", (long long)(n >> 20));
  else if (n >= (1 << 10)) std::snprintf(buf, sizeof buf, "%lldK", (long long)(n >> 10));
  else                     std::snprintf(buf, sizeof buf, "%lld",  (long long)n);
  return buf;
}

template <class F>
void row(const char* name, Device d, int64_t n, double bytes_per_elem,
         double peak, F&& fn) {
  char label[24];
  std::snprintf(label, sizeof label, "%s %s", name, size_label(n));
  const double ns = nn::bench::time_ns_on(d, fn, reps_for(n));
  nn::bench::report_bandwidth(label, ns, bytes_per_elem * double(n), peak);
}

void bench_device(Device d) {
  const auto& k = nn::kernels::kernels(d);
  const Stream& s = nn::current_stream(d);
  const double peak = nn::bench::peak_bandwidth_gb_s(d);

  std::printf("\n=== %s (%s backend)", nn::device_name(d),
              nn::kernels::active_backend_name(d));
  if (peak > 0.0) std::printf(", theoretical peak %.0f GB/s", peak);
  std::printf(" ===\n");

  for (int64_t n : kSizes) {
    Tensor A = Tensor::zeros({int(n)}, d);
    Tensor B = Tensor::zeros({int(n)}, d);
    Tensor C = Tensor::zeros({int(n)}, d);
    float* a = A.device_ptr();
    float* b = B.device_ptr();
    float* c = C.device_ptr();

    std::printf("\n");
    if (k.fill)
      row("fill", d, n, 4.0, peak, [&] { k.fill(s, 1.0f, a, n); });
    if (k.scale)
      row("scale", d, n, 8.0, peak, [&] { k.scale(s, 1.001f, a, n); });
    if (k.relu)
      row("relu", d, n, 8.0, peak, [&] { k.relu(s, a, b, n); });
    if (k.add)
      row("add", d, n, 12.0, peak, [&] { k.add(s, a, b, c, n); });
    if (k.axpy)
      row("axpy", d, n, 12.0, peak, [&] { k.axpy(s, 2.0f, a, b, n); });
    if (k.relu_backward)
      row("relu_bwd", d, n, 12.0, peak, [&] { k.relu_backward(s, a, b, c, n); });
  }
}

void report_missing(Device d) {
  const auto& k = nn::kernels::kernels(d);
  const char* names[] = {"fill", "scale", "relu", "add", "axpy", "relu_backward"};
  const void* ptrs[] = {(const void*)k.fill,  (const void*)k.scale,
                        (const void*)k.relu,  (const void*)k.add,
                        (const void*)k.axpy,  (const void*)k.relu_backward};
  bool any = false;
  for (size_t i = 0; i < std::size(names); ++i) {
    if (!ptrs[i]) {
      if (!any) { std::printf("  not implemented on %s:", nn::device_name(d)); any = true; }
      std::printf(" %s", names[i]);
    }
  }
  if (any) std::printf("\n");
}

}

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  nn::kernels::init_kernels();

  std::vector<Device> devices{Device::CPU};
  if (nn::cuda_device_count() > 0) devices.push_back(Device::CUDA);
  else std::printf("no CUDA device found -- CPU only\n");

  for (Device d : devices) report_missing(d);
  for (Device d : devices) bench_device(d);

  std::printf("\n");
  return 0;
}
