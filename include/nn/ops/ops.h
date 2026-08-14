#pragma once

#include <nn/core/tensor.h>

namespace nn::ops {

Tensor matmul(const Tensor& a, const Tensor& b, bool transA = false, bool transB = false);
Tensor add_row_bias(const Tensor& x, const Tensor& bias);
Tensor col_sum(const Tensor& x);
Tensor relu(const Tensor& x);
Tensor relu_backward(const Tensor& x, const Tensor& g_out);
Tensor add(const Tensor& a, const Tensor& b);

void   add_inplace(Tensor& a, const Tensor& b);
void   scale_inplace(Tensor& a, float alpha);
void   axpy_inplace(Tensor& y, float alpha, const Tensor& x);
void   fill_inplace(Tensor& a, float v);
void   softmax_ce(const Tensor& logits, const Tensor& labels, Tensor& loss_out, Tensor& probs);

Tensor softmax_ce_backward(const Tensor& probs, const Tensor& labels, float g_loss);

}