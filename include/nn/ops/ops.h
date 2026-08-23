#pragma once

#include <nn/core/tensor.h>
#include <nn/kernels/elementwise_ops.h>

namespace nn::ops {

using kernels::UnaryOp;
using kernels::BinaryOp;
using kernels::ScalarOp;

Tensor matmul(const Tensor& a, const Tensor& b, bool transA = false, bool transB = false);
void   matmul_into(Tensor& out, const Tensor& a, const Tensor& b,
                   bool transA = false, bool transB = false);

// The elementwise family. One entry point per arity; which arithmetic runs is
// the op code, from unary_ops.def / binary_ops.def / scalar_ops.def.
Tensor unary(UnaryOp op, const Tensor& x);
Tensor unary_backward(UnaryOp op, const Tensor& x, const Tensor& y, const Tensor& g);
Tensor binary(BinaryOp op, const Tensor& a, const Tensor& b);
Tensor binary_backward(BinaryOp op, int side, const Tensor& a, const Tensor& b,
                       const Tensor& c, const Tensor& g);
Tensor scalar(ScalarOp op, const Tensor& x, float k);
Tensor scalar_backward(ScalarOp op, const Tensor& x, const Tensor& y,
                       const Tensor& g, float k);

// Named shortcuts for the ops with callers older than the generic family.
Tensor relu(const Tensor& x);
Tensor relu_backward(const Tensor& x, const Tensor& g_out);
Tensor add(const Tensor& a, const Tensor& b);
Tensor mul(const Tensor& a, const Tensor& b);

// Reductions. sum_to is the primitive: it is the backward of every broadcast,
// and sum over an axis is the same kernel with that axis set to 1.
Tensor sum_to(const Tensor& g, const Shape& target);
Tensor sum_all(const Tensor& x);
Tensor sum_dim(const Tensor& x, int dim, bool keepdim);
Tensor mean_all(const Tensor& x);
Tensor mean_dim(const Tensor& x, int dim, bool keepdim);

// Softmax over the last axis
Tensor softmax_rows(const Tensor& x);
Tensor softmax_rows_backward(const Tensor& y, const Tensor& g);

// weight is [V, D] and idx is I32 of any shape; the result is idx.shape + [D].
Tensor embedding(const Tensor& weight, const Tensor& idx);
Tensor embedding_backward(const Tensor& g, const Tensor& idx, int V);

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

// Resolve a possibly-negative axis index against a rank, the numpy way.
int normalise_dim(int dim, int rank, const char* op);

}
