// GEMM: batching, transposes and the row-folding that lets a rank-N tensor reach a 2-D kernel.

#include <nn/ops/ops.h>

#include <stdexcept>
#include <string>

#include <kernels/kernel_api.h>

#include "ops_common.h"

namespace nn::ops {

// Private to GEMM: batch bookkeeping and the row/stride folding.
namespace {

Shape batch_shape_of(const Shape& s) {
  int dims[kMaxShapeRank] = {0};
  const int n = (s.rank() > 2) ? s.rank() - 2 : 0;
  for (int i = 0; i < n; ++i) dims[i] = s.dim(i);
  return Shape(std::span<const int>(dims, n));
}

Shape with_core(const Shape& batch, int r, int c) {
  int dims[kMaxShapeRank] = {0};
  for (int i = 0; i < batch.rank(); ++i) dims[i] = batch.dim(i);
  dims[batch.rank()] = r;
  dims[batch.rank() + 1] = c;
  return Shape(std::span<const int>(dims, batch.rank() + 2));
}

bool foldable_rows(const Tensor& t, int64_t& rows, int64_t& row_stride) {
  const int r = t.shape().rank();
  const int64_t last = t.shape().dim(r - 1);
  rows = (last > 0) ? t.numel() / last : 0;
  if (t.is_contiguous()) { row_stride = last; return true; }

  const TensorView v = view_of(t);
  if (v.rank == 2 && v.shape[1] == last && v.stride[1] == 1) {
    row_stride = v.stride[0];
    return true;
  }
  return false;
}

bool flat_batch_stride(const Tensor& t, int64_t nbatch, int64_t& out) {
  TensorView v{};
  for (int i = 0; i < t.shape().rank() - 2; ++i) {
    const int64_t d = t.shape().dim(i), s = t.stride(i);
    if (d == 1) continue;
    if (v.rank > 0 && v.stride[v.rank - 1] == s * d) {
      v.shape[v.rank - 1] *= d;
      v.stride[v.rank - 1] = s;
      continue;
    }
    v.shape[v.rank] = d;
    v.stride[v.rank] = s;
    ++v.rank;
  }
  if (v.rank == 0) { out = 0; return true; }          // every batch axis is 1
  if (v.rank == 1 && v.shape[0] == nbatch) { out = v.stride[0]; return true; }
  return false;
}

// What a GEMM operand contributes, plus a packed copy when one was unavoidable.
struct GemmOperand {
  Tensor t;
  int64_t ld = 0;
  int64_t batch_stride = 0;
};

GemmOperand gemm_operand(const Tensor& expanded, const Shape& own_batch,
                         const Shape& batch, int64_t nbatch, const char* op) {
  const int r = expanded.shape().rank();
  int64_t sbatch = 0;
  const bool rows_ok = (expanded.stride(r - 1) == 1);
  if (rows_ok && flat_batch_stride(expanded, nbatch, sbatch)) {
    return {expanded, expanded.stride(r - 2), sbatch};
  }

  if (own_batch != batch) {
    const Tensor dense = expanded.pack();
    const int dr = dense.shape().rank();
    return {dense, dense.shape().dim(dr - 1),
            int64_t(dense.shape().dim(dr - 2)) * dense.shape().dim(dr - 1)};
  }

  if (!rows_ok) {
    throw std::invalid_argument(std::string(op) +
        ": innermost axis must be contiguous (call .pack() first)");
  }
  throw std::invalid_argument(std::string(op) + ": the batch axes of " +
      expanded.shape().str() + " are not evenly spaced, so the matrices cannot "
      "be reached by one stride (call .pack() first)");
}

}  // namespace

// The last two axes are the matrix; everything left of them is batch
Tensor matmul(const Tensor& a, const Tensor& b, bool transA, bool transB) {
  same_device(a, b, "matmul");

  const int ar = a.shape().rank(), br = b.shape().rank();
  if (ar < 2 || br < 2) {
    throw std::invalid_argument("matmul: both operands need rank >= 2, got " +
                                a.shape().str() + " and " + b.shape().str());
  }

  const int M  = transA ? a.shape().dim(ar - 1) : a.shape().dim(ar - 2);
  const int Ka = transA ? a.shape().dim(ar - 2) : a.shape().dim(ar - 1);
  const int Kb = transB ? b.shape().dim(br - 1) : b.shape().dim(br - 2);
  const int N  = transB ? b.shape().dim(br - 2) : b.shape().dim(br - 1);
  if (Ka != Kb) {
    throw std::invalid_argument("matmul: inner dimensions must match, got " +
                                a.shape().str() + " and " + b.shape().str());
  }

  const Shape a_batch = batch_shape_of(a.shape());
  const Shape b_batch = batch_shape_of(b.shape());
  const auto& k = nn::kernels::kernels(a.device());
  const Stream& s = current_stream(a.device());

  if (a_batch.rank() > 0 && b_batch.rank() == 0 && !transA) {
    int64_t rows = 0, row_stride = 0;
    if (!foldable_rows(a, rows, row_stride)) {
      throw std::invalid_argument("matmul (A): " + a.shape().str() +
          " cannot be folded into a matrix, its rows are not evenly spaced "
          "(call .pack() first)");
    }
    Shape out_shape = a.shape();
    out_shape.set_dim(ar - 1, N);

    Tensor out(out_shape, a.device(), a.dtype());
    k.gemm(s, a.device_ptr(), b.device_ptr(), out.device_ptr(),
           int(rows), N, Ka, row_stride, row_stride_of(b, "matmul (B)"), /*ldc=*/N,
           /*transA=*/false, transB, /*batch=*/1, 0, 0, 0);
    return out;
  }

  const Shape batch = broadcast_shapes(a_batch, b_batch);
  const int64_t nbatch = batch.numel();
  if (nbatch > INT32_MAX) {
    throw std::invalid_argument("matmul: batch " + batch.str() + " is too large");
  }

  const Tensor ea = a.expand_view(with_core(batch, a.shape().dim(ar - 2), a.shape().dim(ar - 1)));
  const Tensor eb = b.expand_view(with_core(batch, b.shape().dim(br - 2), b.shape().dim(br - 1)));

  const GemmOperand oa = gemm_operand(ea, a_batch, batch, nbatch, "matmul (A)");
  const GemmOperand ob = gemm_operand(eb, b_batch, batch, nbatch, "matmul (B)");

  Tensor out(with_core(batch, M, N), a.device(), a.dtype());
  k.gemm(s, oa.t.device_ptr(), ob.t.device_ptr(), out.device_ptr(),
         M, N, Ka, oa.ld, ob.ld, /*ldc=*/N, transA, transB,
         int(nbatch), oa.batch_stride, ob.batch_stride, int64_t(M) * N);
  return out;
}

void matmul_into(Tensor& out, const Tensor& a, const Tensor& b, bool transA, bool transB) {
  same_device(a, b, "matmul_into");
  same_device(a, out, "matmul_into");

  if (a.shape().rank() != 2 || b.shape().rank() != 2 || out.shape().rank() != 2) {
    throw std::invalid_argument("matmul_into: all tensors must be 2D");
  }
  int M = transA ? a.shape().dim(1) : a.shape().dim(0);
  int K_a = transA ? a.shape().dim(0) : a.shape().dim(1);
  int K_b = transB ? b.shape().dim(1) : b.shape().dim(0);
  int N = transB ? b.shape().dim(0) : b.shape().dim(1);
  if (K_a != K_b) {
    throw std::invalid_argument("matmul_into: inner dimensions must match");
  }
  if (out.shape() != Shape{M, N}) {
    throw std::invalid_argument("matmul_into: out has the wrong shape");
  }

  const int64_t lda = row_stride_of(a, "matmul_into (A)");
  const int64_t ldb = row_stride_of(b, "matmul_into (B)");
  const int64_t ldc = row_stride_of(out, "matmul_into (C)");

  const auto& k = nn::kernels::kernels(a.device());
  k.gemm(current_stream(a.device()), a.device_ptr(), b.device_ptr(), out.device_ptr(),
         M, N, K_a, lda, ldb, ldc, transA, transB, /*batch=*/1, 0, 0, 0);
}

}  // namespace nn::ops
