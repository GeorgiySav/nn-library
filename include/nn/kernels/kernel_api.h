#pragma once

#include <cstdint>

#include <nn/core/strides.h>
#include <nn/core/device.h>
#include <nn/kernels/elementwise_ops.h>

namespace nn::kernels {

// Which strides each kind of kernel absorbs, and what it requires instead:
//
//   kernel kind               absorbs                    requires
//   ------------------------  -------------------------  ------------------------
//   elementwise               all strides, on inputs     output is dense
//   GEMM                      the row stride, as ld*     innermost stride == 1
//   row reductions            the row stride             innermost stride == 1
//   whole-tensor reductions   all strides                nothing
//   gather, embedding         nothing                    fully contiguous
//   scatter (unpack)          all strides, on the OUTPUT input is dense
//
// Outputs are otherwise always dense: every non-inplace ops:: function
// allocates its own output, so it is contiguous by construction and only the
// reading side needs strides. The in-place ops require contiguity on their
// destination. unpack is the one deliberate exception -- writing a result
// into a window of a larger dense tensor is what the backward of slice is, and
// it cannot be expressed as a dense output.
//

using GemmFn              = void(const Stream& s, const float* A, const float* B,
                                    float*C, int M, int N, int K,
                                    int64_t lda, int64_t ldb, int64_t ldc,
                                    bool transA, bool transB);
using AxpyFn              = void(const Stream& s, float alpha, const float* X, float* Y, int64_t n);
using FillFn              = void(const Stream& s, float v, float* X, int64_t n);
// fill, but the value is read from device memory instead of a host argument
using FillFromFn          = void(const Stream& s, const float* src, float* X, int64_t n);
using SoftmaxCeFn         = void(const Stream& s, const float* logits, const int32_t* labels,
                                    float* loss_out, float* probs, int M, int N, int64_t sz);
using SoftmaxCeBackwardFn = void(const Stream& s, const float* probs, const int32_t* labels,
                                    const float* g_loss, float* g_logits, int M, int N, int64_t sp);
using AdamStepFn          = void(const Stream& s, float* p, const float* g, float* m, float* v,
                                    float lr, float b1, float b2, float eps, float bc1, float bc2, int64_t n);
using ArgmaxRowsFn        = void(const Stream& s, const float* X, int32_t* out,
                                    int M, int N, int64_t sx);

// The elementwise family. One slot per arity rather than per op: the op code
// selects the arithmetic inside the kernel, from elementwise_ops.h.
using UnaryFn             = void(const Stream& s, UnaryOp op,
                                    const float* X, TensorView vx,
                                    float* Y, int64_t n);
using UnaryBackwardFn     = void(const Stream& s, UnaryOp op,
                                    const float* X, TensorView vx,
                                    const float* Y, TensorView vy,
                                    const float* G, TensorView vg,
                                    float* gX, int64_t n);
using BinaryFn            = void(const Stream& s, BinaryOp op,
                                    const float* A, TensorView va,
                                    const float* B, TensorView vb,
                                    float* C, int64_t n);
// side 0 -> d/dA, side 1 -> d/dB, both at the broadcast shape.
using BinaryBackwardFn    = void(const Stream& s, BinaryOp op, int side,
                                    const float* A, TensorView va,
                                    const float* B, TensorView vb,
                                    const float* C, TensorView vc,
                                    const float* G, TensorView vg,
                                    float* out, int64_t n);
using ScalarFn            = void(const Stream& s, ScalarOp op, float k,
                                    const float* X, TensorView vx,
                                    float* Y, int64_t n);
using ScalarBackwardFn    = void(const Stream& s, ScalarOp op, float k,
                                    const float* X, TensorView vx,
                                    const float* Y, TensorView vy,
                                    const float* G, TensorView vg,
                                    float* gX, int64_t n);

// Softmax over the last axis
using SoftmaxRowsFn         = void(const Stream& s, const float* X, float* Y,
                                      int M, int N, int64_t sx);
using SoftmaxRowsBackwardFn = void(const Stream& s, const float* Y, const float* G,
                                      float* gX, int M, int N,
                                      int64_t sy, int64_t sg);

// Row gather
using EmbeddingFn         = void(const Stream& s, const float* W, const int32_t* idx,
                                    float* Y, int64_t n_idx, int D, int V);
using EmbeddingBackwardFn = void(const Stream& s, const float* G, const int32_t* idx,
                                    float* gW, int64_t n_idx, int D, int V);

// pack gathers a strided view into dense storage -- what contiguous() does.
// unpack is the other direction: a dense buffer written out through the
// destination's strides, which is what the backward of slice and the forward of
// cat need. Neither is a "strided variant" of anything; the strides are the
// whole point, so there is no sibling to pair them with.
using PackFn               = void(const Stream& s, const float* src, TensorView v,
                                     float* dst, int64_t n);
using PackI32Fn            = void(const Stream& s, const int32_t* src, TensorView v,
                                     int32_t* dst, int64_t n);
using UnpackFn             = void(const Stream& s, const float* src,
                                     float* dst, TensorView vdst, int64_t n);

using SumToFn              = void(const Stream& s, const float* g,
                                     TensorView keep, TensorView red,
                                     float* out, int64_t n_out, int64_t n_red);

constexpr int kSumAllWorkspace = 1024;

// One kernel for every layout: see strided_index.h for the measurement that
// retired the dense sibling.
using SumAllFn             = void(const Stream& s, const float* X, TensorView v,
                                     float* out, float* workspace, int64_t n);

// Slots are pointers to the function types above, generated so that the table
// and the two backends can never drift out of step.
struct KernelTable {
#define NN_KERNEL(name, Type) Type* name = nullptr;
#include <nn/kernels/kernel_list.def>
#undef NN_KERNEL
};

KernelTable& table(Device d);
const KernelTable& kernels(Device d);
void register_naive_kernels();
// Throws if any slot is still null -- a forgotten registration is otherwise a
// null call at first use, arbitrarily far from the omission.
void validate_table(Device d);
void init_kernels();
const char* active_backend_name(Device d);

}
