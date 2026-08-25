#include <nn/kernels/kernel_api.h>

#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "naive/naive_kernels.h"
#if defined(NN_WITH_CUDA)
#include "cuda/cuda_kernels.h"
#endif

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
  init_kernels();

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
#define NN_KERNEL(name, Type) t.name = &naive_##name;
#include <nn/kernels/kernel_list.def>
#undef NN_KERNEL

  g_backend[index_of(Device::CPU)] = "naive";
}

#if defined(NN_WITH_CUDA)

void register_cuda_kernels() {
  KernelTable& t = table(Device::CUDA);
#define NN_KERNEL(name, Type) t.name = &cuda_##name;
#include <nn/kernels/kernel_list.def>
#undef NN_KERNEL

  g_backend[index_of(Device::CUDA)] = "CUDA";
}

#endif  // NN_WITH_CUDA

void validate_table(Device d) {
  const KernelTable& t = g_tables[index_of(d)];
#define NN_KERNEL(name, Type)                                            \
  if (!t.name) {                                                         \
    throw std::runtime_error(std::string("kernel \"" #name "\" is not "     \
                             "registered for ") + device_name(d));         \
  }
#include <nn/kernels/kernel_list.def>
#undef NN_KERNEL
}

#if defined(NN_WITH_CUDA)

void register_cublas_kernels() {
  KernelTable& t = table(Device::CUDA);

  t.gemm = &cublas_gemm;

  g_backend[index_of(Device::CUDA)] = "cuBLAS";
}

#endif  // NN_WITH_CUDA

void init_kernels() {
  static const bool once = [] {
    register_naive_kernels();

    validate_table(Device::CPU);

#if defined(NN_WITH_CUDA)
    register_cuda_kernels();

    const char* sel = std::getenv("NN_KERNELS");
    if (!(sel && std::strcmp(sel, "handwritten") == 0))
      register_cublas_kernels();

    validate_table(Device::CUDA);
#endif

    return true;
  }();
  (void)once;
}

}
