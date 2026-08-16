#pragma once

#include <cstdint>

namespace nn::kernels {

void naive_gemm(const float* A, const float* B, float* C, int M, int N, int K, bool transA, bool transB);
void naive_add_row_bias(const float* X, const float* b, float* Y, int M, int N);
void naive_col_sum(const float* X, float* out, int M, int N);
void naive_relu(const float* X, float* Y, int64_t n);
void naive_relu_backward(const float* X, const float* gY, float* gX, int64_t n);
void naive_add(const float* A, const float* B, float* C, int64_t n);
void naive_scale(float alpha, float* X, int64_t n);
void naive_axpy(float alpha, const float* X, float* Y, int64_t n);
void naive_fill(float v, float* X, int64_t n);
void naive_softmax_ce(const float* logits, const int32_t* labels, float* loss_out, float* probs, int M, int N);
void naive_softmax_ce_backward(const float* probs, const int32_t* labels, const float* g_loss, float* g_logits, int M, int N);

}