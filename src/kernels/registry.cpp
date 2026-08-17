#include <nn/kernels/kernel_api.h>

#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "naive/naive_kernels.h"
#include "cuda/cuda_kernels.h"

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
  // Idempotent and thread-safe via the function-local static in init_kernels.
  // Note table() must NOT do this: register_naive_kernels() calls it, and
  // re-entering init_kernels() during its own initialisation is UB.
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
  t.gemm = &naive_gemm;
  // reduce
  t.add_row_bias = &naive_add_row_bias;
  t.col_sum = &naive_col_sum;
  t.argmax_rows = &naive_argmax_rows;
  // elementwise
  t.relu = &naive_relu;
  t.relu_backward = &naive_relu_backward;
  t.add = &naive_add;
  t.scale = &naive_scale;
  t.axpy = &naive_axpy;
  t.fill = &naive_fill;
  // softmax cross-entropy
  t.softmax_ce = &naive_softmax_ce;
  t.softmax_ce_backward = &naive_softmax_ce_backward;

  g_backend[index_of(Device::CPU)] = "naive";
}

void register_cuda_kernels() {
  KernelTable& t = table(Device::CUDA); 
  t.gemm = nullptr;
  // reduce
  t.add_row_bias = nullptr;
  t.col_sum = nullptr;
  t.argmax_rows = nullptr;
  // elementwise
  t.relu = &cuda_relu;
  t.relu_backward = &cuda_relu_backward;
  t.add = &cuda_add;
  t.scale = &cuda_scale;
  t.axpy = &cuda_axpy;
  t.fill = &cuda_fill;
  // softmax cross-entropy
  t.softmax_ce = nullptr;
  t.softmax_ce_backward = nullptr;

  g_backend[index_of(Device::CUDA)] = "CUDA";

}

void init_kernels() {
  static const bool once = [] {
    register_naive_kernels();
    register_cuda_kernels();

    const char* sel = std::getenv("NN_KERNELS");
    if (sel && std::strcmp(sel, "naive") == 0) return true;

    return true;
  }();
  (void)once;
}

}