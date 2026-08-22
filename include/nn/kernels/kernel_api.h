#pragma once

#include <cstdint>

#include <nn/core/strides.h>
#include <nn/core/device.h>

namespace nn::kernels {

// Which strides each kind of kernel absorbs, and what it requires instead:
//
//   kernel kind               absorbs                    requires
//   ------------------------  -------------------------  ------------------------
//   elementwise               all strides, on inputs     output is dense
//   GEMM                      the row stride, as ld*     innermost stride == 1
//   row reductions            the row stride             innermost stride == 1
//   whole-tensor reductions   all strides                nothing
//   gather, embedding         nothing                    fully contiguous
//   scatter (copy_into)       all strides, on the OUTPUT input is dense
//
// Outputs are otherwise always dense: every non-inplace ops:: function
// allocates its own output, so it is contiguous by construction and only the
// reading side needs strides. The in-place ops require contiguity on their
// destination. copy_into is the one deliberate exception -- writing a result
// into a window of a larger dense tensor is what the backward of slice is, and
// it cannot be expressed as a dense output.

using GemmFn              = void(*)(const Stream& s, const float* A, const float* B,
                                    float*C, int M, int N, int K,
                                    int64_t lda, int64_t ldb, int64_t ldc,
                                    bool transA, bool transB);
using AddRowBiasFn        = void(*)(const Stream& s, const float* X, const float* b, float* Y,
                                    int M, int N, int64_t sx);
using ColSumFn            = void(*)(const Stream& s, const float* X, float* out,
                                    int M, int N, int64_t sx);
using ReluFn              = void(*)(const Stream& s, const float* X, float* Y, int64_t n);
using ReluBackwardFn      = void(*)(const Stream& s, const float* X, const float* gY, float* gX, int64_t n);
using AddFn               = void(*)(const Stream& s, const float* A, const float* B, float* C, int64_t n);
using ScaleFn             = void(*)(const Stream& s, float alpha, float* X, int64_t n);
using AxpyFn              = void(*)(const Stream& s, float alpha, const float* X, float* Y, int64_t n);
using FillFn              = void(*)(const Stream& s, float v, float* X, int64_t n);
// fill, but the value is read from device memory instead of a host argument
using FillFromFn          = void(*)(const Stream& s, const float* src, float* X, int64_t n);
using SoftmaxCeFn         = void(*)(const Stream& s, const float* logits, const int32_t* labels,
                                    float* loss_out, float* probs, int M, int N, int64_t sz);
using SoftmaxCeBackwardFn = void(*)(const Stream& s, const float* probs, const int32_t* labels,
                                    const float* g_loss, float* g_logits, int M, int N, int64_t sp);
using AdamStepFn          = void(*)(const Stream& s, float* p, const float* g, float* m, float* v,
                                    float lr, float b1, float b2, float eps, float bc1, float bc2, int64_t n);
using ArgmaxRowsFn        = void(*)(const Stream& s, const float* X, int32_t* out,
                                    int M, int N, int64_t sx);

// Strided siblings
using ReluStridedFn         = void(*)(const Stream& s, const float* X, TensorView v,
                                      float* Y, int64_t n);
using ReluBackwardStridedFn = void(*)(const Stream& s, const float* X, TensorView vx,
                                      const float* gY, TensorView vg,
                                      float* gX, int64_t n);
using AddStridedFn          = void(*)(const Stream& s, const float* A, TensorView va,
                                      const float* B, TensorView vb,
                                      float* C, int64_t n);

using CopyStridedFn        = void(*)(const Stream& s, const float* src, TensorView v,
                                    float* dst, int64_t n);
using CopyStridedI32Fn     = void(*)(const Stream& s, const int32_t* src, TensorView v,
                                    int32_t* dst, int64_t n);
using CopyIntoStridedFn    = void(*)(const Stream& s, const float* src,
                                     float* dst, TensorView vdst, int64_t n);

constexpr int kSumAllWorkspace = 1024;

using SumAllFn             = void(*)(const Stream& s, const float* X, float* out,
                                     float* workspace, int64_t n);
using SumAllStridedFn      = void(*)(const Stream& s, const float* X, TensorView v,
                                     float* out, float* workspace, int64_t n);

struct KernelTable {
  GemmFn gemm = nullptr;
  AddRowBiasFn add_row_bias = nullptr;
  ColSumFn col_sum = nullptr;
  ReluFn relu = nullptr;
  ReluBackwardFn relu_backward = nullptr;
  AddFn add = nullptr;
  ScaleFn scale = nullptr;
  AxpyFn axpy = nullptr;
  FillFn fill = nullptr;
  FillFromFn fill_from = nullptr;
  SoftmaxCeFn softmax_ce = nullptr;
  SoftmaxCeBackwardFn softmax_ce_backward = nullptr;
  AdamStepFn adam_step = nullptr;
  ArgmaxRowsFn argmax_rows = nullptr;

  ReluStridedFn relu_strided = nullptr;
  ReluBackwardStridedFn relu_backward_strided = nullptr;
  AddStridedFn add_strided = nullptr;

  CopyStridedFn copy_strided = nullptr;
  CopyStridedI32Fn copy_strided_i32 = nullptr;
  CopyIntoStridedFn copy_into_strided = nullptr;

  SumAllFn sum_all = nullptr;
  SumAllStridedFn sum_all_strided = nullptr;
};

KernelTable& table(Device d);
const KernelTable& kernels(Device d);
void register_naive_kernels();
void init_kernels();
const char* active_backend_name(Device d);

}