#pragma once

#include <cstdint>

#include <nn/core/device.h>
#include <nn/core/strides.h>

namespace nn::kernels {
// lda/ldb/ldc are leading dimensions: the distance between consecutive rows of
// each operand as stored. They equal K/N/N only when the operand is dense.
void cuda_gemm(const Stream& s, const float* A, const float* B, float* C,
               int M, int N, int K, int64_t lda, int64_t ldb, int64_t ldc,
               bool transA, bool transB);
void cublas_gemm(const Stream& s, const float* A, const float* B, float* C,
               int M, int N, int K, int64_t lda, int64_t ldb, int64_t ldc,
               bool transA, bool transB);

void cuda_fill(const Stream& s, float v, float* X, int64_t n);
void cuda_fill_from(const Stream& s, const float* src, float* X, int64_t n);
void cuda_scale(const Stream& s, float alpha, float* X, int64_t n);
void cuda_axpy(const Stream& s, float alpha, const float* X, float* Y, int64_t n);
void cuda_add(const Stream& s, const float* A, const float* B, float* C, int64_t n);
void cuda_relu(const Stream& s, const float* X, float* Y, int64_t n);
void cuda_relu_backward(const Stream& s, const float* X, const float* gY, float* gX, int64_t n);

void cuda_relu_strided(const Stream& s, const float* X, TensorView v, float* Y, int64_t n);
void cuda_relu_backward_strided(const Stream& s, const float* X, TensorView vx,
                                const float* gY, TensorView vg, float* gX, int64_t n);
void cuda_add_strided(const Stream& s, const float* A, TensorView va,
                      const float* B, TensorView vb, float* C, int64_t n);

// sx / sz / sp is the row stride of the *reading* operand named in the signature;
// every output stays dense with row stride N.
void cuda_add_row_bias(const Stream& s, const float* X, const float* b, float* Y, int M, int N, int64_t sx);
void cuda_argmax_rows(const Stream& s, const float* X, int32_t* out, int M, int N, int64_t sx);
void cuda_col_sum(const Stream& s, const float* X, float* out, int M, int N, int64_t sx);

void cuda_softmax_ce(const Stream& s, const float* logits, const int32_t* labels,
                      float* loss_out, float* probs, int M, int N, int64_t sz);
void cuda_softmax_ce_backward(const Stream& s, const float* probs, const int32_t* labels,
                              const float* g_loss, float* g_logits, int M, int N, int64_t sp);

void cuda_adam_step(const Stream& s, float* p, const float* g, float* m, float* v,
                    float lr, float b1, float b2, float eps, float bc1, float bc2, int64_t n);

void cuda_copy_strided(const Stream& s, const float* src, TensorView v,
                       float* dst, int64_t n);
void cuda_copy_strided_i32(const Stream& s, const int32_t* src, TensorView v,
                           int32_t* dst, int64_t n);
void cuda_copy_into_strided(const Stream& s, const float* src,
                            float* dst, TensorView vdst, int64_t n);

void cuda_sum_all(const Stream& s, const float* X, float* out,
                  float* workspace, int64_t n);
void cuda_sum_all_strided(const Stream& s, const float* X, TensorView v,
                          float* out, float* workspace, int64_t n);
}