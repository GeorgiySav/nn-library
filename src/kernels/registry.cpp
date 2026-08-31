#include <kernels/kernel_api.h>

#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#include "naive/naive_kernels.h"
#if defined(NN_WITH_CUDA)
#include "cuda/cuda_kernels.h"
#endif

namespace nn::kernels {
namespace {

constexpr int kNumDevices = 2;  // CPU and CUDA

constexpr int index_of(Device d) { return static_cast<int>(d); }

constinit KernelTable g_tables[kNumDevices];
constinit const char* g_backend[kNumDevices] = {"none", "none"};

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

// NN_KERNEL is expanded once per kernel listed in kernel_list.def, so every
// slot in the table gets wired to its naive_* implementation here without
// naming them one by one.
void register_naive_kernels() {
  KernelTable& t = table(Device::CPU);
#define NN_KERNEL(name, Type) t.name = &naive_##name;
#include <kernels/kernel_list.def>
#undef NN_KERNEL

  g_backend[index_of(Device::CPU)] = "naive";
}

#if defined(NN_WITH_CUDA)

void register_cuda_kernels() {
  KernelTable& t = table(Device::CUDA);
#define NN_KERNEL(name, Type) t.name = &cuda_##name;
#include <kernels/kernel_list.def>
#undef NN_KERNEL

  g_backend[index_of(Device::CUDA)] = "CUDA";
}

#endif  // NN_WITH_CUDA

// re-expands kernel_list.def to check every slot got a registration; throws
// with the missing kernel's name instead of leaving a null function pointer
// to crash at first call.
void validate_table(Device d) {
  const KernelTable& t = g_tables[index_of(d)];
#define NN_KERNEL(name, Type)                                            \
  if (!t.name) {                                                         \
    throw std::runtime_error(std::string("kernel \"" #name "\" is not "     \
                             "registered for ") + device_name(d));         \
  }
#include <kernels/kernel_list.def>
#undef NN_KERNEL
}

#if defined(NN_WITH_CUDA)

// overrides just the gemm slot with cuBLAS's implementation, leaving every
// other CUDA kernel as the handwritten one registered above.
void register_cublas_kernels() {
  KernelTable& t = table(Device::CUDA);

  t.gemm = &cublas_gemm;

  g_backend[index_of(Device::CUDA)] = "cuBLAS";
}

#endif  // NN_WITH_CUDA

// runs registration exactly once, lazily, on first use of any device's
// kernel table.
void init_kernels() {
  static const bool once = [] {
    register_naive_kernels();

    validate_table(Device::CPU);

#if defined(NN_WITH_CUDA)
    register_cuda_kernels();

    // NN_KERNELS=handwritten opts out of cuBLAS, useful for isolating bugs
    // to one gemm implementation or the other.
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
