#pragma once

#include <cstdint>

#include <nn/core/device.h>

namespace nn::kernels {
void cuda_gemm(const Stream& s, const float* A, const float* B, float* C,
               int M, int N, int K, bool transA, bool transB);

void cuda_fill(const Stream& s, float v, float* X, int64_t n);
void cuda_scale(const Stream& s, float alpha, float* X, int64_t n);
void cuda_axpy(const Stream& s, float alpha, const float* X, float* Y, int64_t n);
void cuda_add(const Stream& s, const float* A, const float* B, float* C, int64_t n);
void cuda_relu(const Stream& s, const float* X, float* Y, int64_t n);
void cuda_relu_backward(const Stream& s, const float* X, const float* gY, float* gX, int64_t n);

void cuda_add_row_bias(const Stream& s, const float* X, const float* b, float* Y, int M, int N);
void cuda_argmax_rows(const Stream& s, const float* X, int32_t* out, int M, int N);
void cuda_col_sum(const Stream& s, const float* X, float* out, int M, int N);

void cuda_softmax_ce(const Stream& s, const float* logits, const int32_t* labels,
                      float* loss_out, float* probs, int M, int N);
void cuda_softmax_ce_backward(const Stream& s, const float* probs, const int32_t* labels,
                              const float* g_loss, float* g_logits, int M, int N);

void cuda_adam_step(const Stream& s, float* p, const float* g, float* m, float* v,
                    float lr, float b1, float b2, float eps, float bc1, float bc2, int64_t n);
}