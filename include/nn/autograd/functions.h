#pragma once

#include <nn/core/tensor.h>

namespace nn::autograd {

Tensor matmul(const Tensor& x, const Tensor& w); // [M, K] @ [K, N] -> [M, N]
Tensor add_row_bias(const Tensor& x, const Tensor& b);
Tensor relu(const Tensor& x);
Tensor cross_entropy(const Tensor& logits, const Tensor& labels); // rank 0

}