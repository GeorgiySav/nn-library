#pragma once

#include <nn/core/rng.h>
#include <nn/core/tensor.h>
#include <nn/ops/op_enums.h>

namespace nn::ops {

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
Tensor sum_all(const Tensor& x, Accum a = Accum::Sum);
Tensor sum_dim(const Tensor& x, int dim, bool keepdim);
Tensor mean_all(const Tensor& x);
Tensor mean_dim(const Tensor& x, int dim, bool keepdim);

Tensor dropout(const Tensor& x, float p, uint64_t seed, uint64_t offset);

// Softmax over the last axis
Tensor softmax_rows(const Tensor& x);
Tensor softmax_rows_backward(const Tensor& y, const Tensor& g);

// weight is [V, D] and idx is I32 of any shape; the result is idx.shape + [D].
Tensor embedding(const Tensor& weight, const Tensor& idx);
Tensor embedding_backward(const Tensor& g, const Tensor& idx, int V);

void   add_inplace(Tensor& a, const Tensor& b);
void   scale_inplace(Tensor& a, float alpha);
void   scalar_inplace(Tensor& a, ScalarOp op, float k);
void   axpy_inplace(Tensor& y, float alpha, const Tensor& x);
void   fill_inplace(Tensor& a, float v);
// Broadcast a scalar that already lives on the device.
void   fill_from(Tensor& a, const Tensor& value);
void   softmax_ce(const Tensor& logits, const Tensor& labels, Tensor& loss_out, Tensor& probs);

Tensor softmax_ce_backward(const Tensor& probs, const Tensor& labels, const Tensor& g_loss);

Tensor argmax_rows(const Tensor& x);
void   topk_rows(const Tensor& x, int k, Tensor& values, Tensor& indices);

// Draws one index per row of `weights` ([M, N]), weighted by row. Rows need
// not sum to 1 -- any nonnegative row works -- so softmax output and
// topk_rows' raw values both apply directly, with no renormalising step at
// the call site. The one op here with no kernel behind it: Pcg32 is a
// host-only generator, and by the time a row of weights exists it is already
// tiny next to the forward pass that produced it, so drawing on the host
// costs nothing worth a device kernel.
Tensor multinomial(const Tensor& weights, Pcg32& rng);

// out[i] = src[i, idx[i]] for each row i. src is [M, N], F32 or I32 -- the
// dtype passes through unchanged -- and idx is I32 [M] with values in
// [0, N). What turns multinomial's or argmax_rows' pick, an index into
// topk_rows' [M, K] values, back into topk_rows' matching [M, K] indices,
// i.e. the vocab id actually being sampled. Runs on the host, same as
// multinomial and for the same reason.
Tensor gather_rows(const Tensor& src, const Tensor& idx);

void adam(const Tensor& p, const Tensor& g, Tensor& m, Tensor& v,
          float lr, float beta1, float beta2, float eps, float weight_decay, int step);

// The two directions between a view and dense storage.
//   pack:   src may be any layout, dst must be dense -- this is contiguous().
//   unpack: src is read densely, dst may be any layout -- this is a scatter
//           into a window of a larger tensor, which slice's backward and cat
//           both need.
void pack(const Tensor& dst, const Tensor& src);
void unpack(Tensor& dst, const Tensor& src);


}
