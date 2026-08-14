#include <nn/kernels/kernel_api.h>

#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "naive/naive_kernels.h"

namespace nn::kernels {
namespace {

constexpr int kNumDevices = 2;  // CPU, CUDA

int index_of(Device d) { return static_cast<int>(d); }

KernelTable g_tables[kNumDevices];
const char* g_backend[kNumDevices] = {"none", "none"};

} // namespace

KernelTable& table(Device d) {
  int idx = index_of(d);
  if (idx < 0 || idx >= kNumDevices) {
    throw std::runtime_error("Invalid device");
  }
  return g_tables[idx];
}

const KernelTable& kernels(Device d) {
  int idx = index_of(d);
  if (idx < 0 || idx >= kNumDevices) {
    throw std::runtime_error("Invalid device");
  }
  return g_tables[idx];
}

const char* active_backend_name(Device d) {
  int idx = index_of(d);
  if (idx < 0 || idx >= kNumDevices) {
    throw std::runtime_error("Invalid device");
  }
  return g_backend[idx];
}

void register_naive_kernels() {
  KernelTable& t = table(Device::CPU);
  t.gemm = &naive_gemm;

  g_backend[index_of(Device::CPU)] = "naive";
}

void init_kernels() {
  static const bool once = [] {
    register_naive_kernels();

    const char* sel = std::getenv("NN_KERNELS");
    if (sel && std::strcmp(sel, "naive") == 0) return true;

    return true;
  }();
  (void)once;
}

}