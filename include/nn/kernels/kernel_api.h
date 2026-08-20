#pragma once

#include <cstdint>

#include <nn/core/device.h>

namespace nn::kernels {

using GemmFn              = void(*)(const Stream& s, const float* A, const float* B,
                                    float*C, int M, int N, int K, bool transA, bool transB);
using AddRowBiasFn        = void(*)(const Stream& s, const float* X, const float* b, float* Y, int M, int N);
using ColSumFn            = void(*)(const Stream& s, const float* X, float* out, int M, int N);
using ReluFn              = void(*)(const Stream& s, const float* X, float* Y, int64_t n);
using ReluBackwardFn      = void(*)(const Stream& s, const float* X, const float* gY, float* gX, int64_t n);
using AddFn               = void(*)(const Stream& s, const float* A, const float* B, float* C, int64_t n);
using ScaleFn             = void(*)(const Stream& s, float alpha, float* X, int64_t n);
using AxpyFn              = void(*)(const Stream& s, float alpha, const float* X, float* Y, int64_t n);
using FillFn              = void(*)(const Stream& s, float v, float* X, int64_t n);
using SoftmaxCeFn         = void(*)(const Stream& s, const float* logits, const int32_t* labels,
                                    float* loss_out, float* probs, int M, int N);
using SoftmaxCeBackwardFn = void(*)(const Stream& s, const float* probs, const int32_t* labels,
                                    const float* g_loss, float* g_logits, int M, int N);
using AdamStepFn          = void(*)(const Stream& s, float* p, const float* g, float* m, float* v,
                                    float lr, float b1, float b2, float eps, float bc1, float bc2, int64_t n);
using ArgmaxRowsFn        = void(*)(const Stream& s, const float* X, int32_t* out, int M, int N);

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
  SoftmaxCeFn softmax_ce = nullptr;
  SoftmaxCeBackwardFn softmax_ce_backward = nullptr;
  AdamStepFn adam_step = nullptr;
  ArgmaxRowsFn argmax_rows = nullptr;
};

KernelTable& table(Device d);
const KernelTable& kernels(Device d);
void register_naive_kernels();
void init_kernels();
const char* active_backend_name(Device d);

}