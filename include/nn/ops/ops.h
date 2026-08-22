#pragma once

#include <nn/core/tensor.h>

namespace nn::ops {

Tensor matmul(const Tensor& a, const Tensor& b, bool transA = false, bool transB = false);
void   matmul_into(Tensor& out, const Tensor& a, const Tensor& b,
                   bool transA = false, bool transB = false);
Tensor add_row_bias(const Tensor& x, const Tensor& bias);
Tensor col_sum(const Tensor& x);
Tensor relu(const Tensor& x);
Tensor relu_backward(const Tensor& x, const Tensor& g_out);
Tensor add(const Tensor& a, const Tensor& b);

void   add_inplace(Tensor& a, const Tensor& b);
void   scale_inplace(Tensor& a, float alpha);
void   axpy_inplace(Tensor& y, float alpha, const Tensor& x);
void   fill_inplace(Tensor& a, float v);
// Broadcast a scalar that already lives on the device.
void   fill_from(Tensor& a, const Tensor& value);
void   softmax_ce(const Tensor& logits, const Tensor& labels, Tensor& loss_out, Tensor& probs);

Tensor softmax_ce_backward(const Tensor& probs, const Tensor& labels, const Tensor& g_loss);

Tensor argmax_rows(const Tensor& x);

void adam(const Tensor& p, const Tensor& g, Tensor& m, Tensor& v,
          float lr, float beta1, float beta2, float eps, int step);

void copy_strided(const Tensor& dst, const Tensor& src);
void copy_into(Tensor& dst, const Tensor& src);

Tensor sum_all(const Tensor& x);
}