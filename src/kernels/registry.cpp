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
  t.sum_to = &naive_sum_to;
  t.sum_all = &naive_sum_all;
  t.sum_all_strided = &naive_sum_all_strided;
  // elementwise
  t.relu = &naive_relu;
  t.relu_backward = &naive_relu_backward;
  t.add = &naive_add;
  t.scale = &naive_scale;
  t.axpy = &naive_axpy;
  t.fill = &naive_fill;
  t.fill_from = &naive_fill_from;
  // elementwise, strided reads
  t.relu_strided = &naive_relu_strided;
  t.relu_backward_strided = &naive_relu_backward_strided;
  t.add_strided = &naive_add_strided;
  // softmax cross-entropy
  t.softmax_ce = &naive_softmax_ce;
  t.softmax_ce_backward = &naive_softmax_ce_backward;
  // optimisers
  t.adam_step = &naive_adam_step;
  // copy
  t.copy_strided = &naive_copy_strided;
  t.copy_strided_i32 = &naive_copy_strided_i32;
  t.copy_into_strided = &naive_copy_into_strided;

  g_backend[index_of(Device::CPU)] = "naive";
}

void register_cuda_kernels() {
  KernelTable& t = table(Device::CUDA); 
  t.gemm = &cuda_gemm;
  // reduce
  t.add_row_bias = &cuda_add_row_bias;
  t.col_sum = &cuda_col_sum;
  t.argmax_rows = &cuda_argmax_rows;
  t.sum_to = &cuda_sum_to;
  t.sum_all = &cuda_sum_all;
  t.sum_all_strided = &cuda_sum_all_strided;
  // elementwise
  t.relu = &cuda_relu;
  t.relu_backward = &cuda_relu_backward;
  t.add = &cuda_add;
  t.scale = &cuda_scale;
  t.axpy = &cuda_axpy;
  t.fill = &cuda_fill;
  t.fill_from = &cuda_fill_from;
  // elementwise, strided reads
  t.relu_strided = &cuda_relu_strided;
  t.relu_backward_strided = &cuda_relu_backward_strided;
  t.add_strided = &cuda_add_strided;
  // softmax cross-entropy
  t.softmax_ce = &cuda_softmax_ce;
  t.softmax_ce_backward = &cuda_softmax_ce_backward;
  // optimisers
  t.adam_step = &cuda_adam_step;
  // copy
  t.copy_strided = &cuda_copy_strided;
  t.copy_strided_i32 = &cuda_copy_strided_i32;
  t.copy_into_strided = &cuda_copy_into_strided;

  g_backend[index_of(Device::CUDA)] = "CUDA";
}

void register_cublas_kernels() {
  KernelTable& t = table(Device::CUDA);

  t.gemm = &cublas_gemm;

  g_backend[index_of(Device::CUDA)] = "cuBLAS";
}

void init_kernels() {
  static const bool once = [] {
    register_naive_kernels();
    register_cuda_kernels();

    const char* sel = std::getenv("NN_KERNELS");
    if (!(sel && std::strcmp(sel, "handwritten") == 0))
      register_cublas_kernels();

    return true;
  }();
  (void)once;
}

}