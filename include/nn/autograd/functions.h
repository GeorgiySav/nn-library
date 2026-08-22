#pragma once

#include <nn/core/tensor.h>

namespace nn::autograd {

Tensor matmul(const Tensor& x, const Tensor& w); // [M, K] @ [K, N] -> [M, N]
Tensor add_row_bias(const Tensor& x, const Tensor& b);
Tensor relu(const Tensor& x);
Tensor cross_entropy(const Tensor& logits, const Tensor& labels); // rank 0

Tensor permute(const Tensor& x, std::span<const int> order);
Tensor reshape(const Tensor& x, std::span<const int> shape);
Tensor   slice(const Tensor& x, int axis, int64_t start, int64_t len);

Tensor sum_all(const Tensor& x);

}