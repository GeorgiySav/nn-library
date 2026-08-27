#pragma once

#include <nn/core/tensor.h>
#include <nn/ops/ops.h>

namespace nn::autograd {

Tensor unary(ops::UnaryOp op, const Tensor& x);
Tensor binary(ops::BinaryOp op, const Tensor& a, const Tensor& b);
Tensor scalar(ops::ScalarOp op, const Tensor& x, float k);

#define NN_UNARY(Name, method) Tensor method(const Tensor& x);
#include <nn/ops/unary_ops.def>
#undef NN_UNARY

#define NN_BINARY(Name, method) Tensor method(const Tensor& a, const Tensor& b);
#include <nn/ops/binary_ops.def>
#undef NN_BINARY

#define NN_SCALAR(Name, method) Tensor method(const Tensor& x, float k);
#include <nn/ops/scalar_ops.def>
#undef NN_SCALAR

Tensor matmul(const Tensor& x, const Tensor& w); // [M, K] @ [K, N] -> [M, N]
Tensor cross_entropy(const Tensor& logits, const Tensor& labels); // rank 0

Tensor contiguous(const Tensor& x);   // differentiable pack; identity backward
Tensor  permute(const Tensor& x, std::span<const int> order);
Tensor transpose(const Tensor& x, int a, int b);
Tensor  reshape(const Tensor& x, const Shape& shape);
Tensor    slice(const Tensor& x, int axis, int64_t start, int64_t len);
Tensor   expand(const Tensor& x, const Shape& to);

Tensor sum_all(const Tensor& x);
Tensor sum(const Tensor& x, int dim, bool keepdim = false);
Tensor mean(const Tensor& x);
Tensor mean(const Tensor& x, int dim, bool keepdim = false);

Tensor var(const Tensor& x, int dim, bool keepdim = false, bool unbiased = true);
Tensor stddev(const Tensor& x, int dim, bool keepdim = false, bool unbiased = true);

Tensor softmax(const Tensor& x);                  // over the last axis

Tensor dropout(const Tensor& x, float p, bool training = true);
Tensor embedding(const Tensor& weight, const Tensor& idx);

Tensor layer_norm(const Tensor& x, const Tensor& weight, const Tensor& bias,
                  float eps = 1e-5f);
Tensor rms_norm(const Tensor& x, const Tensor& weight, float eps = 1e-6f);

Tensor cat(std::span<const Tensor> parts, int dim);
Tensor cat(std::initializer_list<Tensor> parts, int dim);

Tensor masked_fill(const Tensor& x, const Tensor& mask, float value);

}

namespace nn {

// The differentiable ops, reachable without naming the tape machinery. The
// non-differentiable kernels keep their own names under nn::ops.
using autograd::cat;
using autograd::contiguous;
using autograd::expand;
using autograd::permute;
using autograd::reshape;
using autograd::slice;
using autograd::transpose;
using autograd::cross_entropy;
using autograd::dropout;
using autograd::layer_norm;
using autograd::rms_norm;
using autograd::masked_fill;
using autograd::softmax;
using autograd::embedding;
using autograd::matmul;

}
