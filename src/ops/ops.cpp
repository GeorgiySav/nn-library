#include <nn/ops/ops.h>

#include <cmath>
#include <stdexcept>

#include <nn/kernels/kernel_api.h>

namespace nn::ops {

namespace {
void same_device(const Tensor& a, const Tensor& b, const char* op) {
  if (a.device() != b.device()) {
    throw std::invalid_argument(std::string(op) + ": operands on different devices");
  }
}

// GEMMs and row-wise reductions absorb the stride between rows; they cannot
// absorb a gap between elements within a row. Returns the row stride to pass
// down
int64_t row_stride_of(const Tensor& t, const char* op) {
  const int r = t.shape().rank();
  if (r < 2) {
    throw std::invalid_argument(std::string(op) + ": needs rank >= 2");
  }
  if (t.stride(r - 1) != 1) {
    throw std::invalid_argument(std::string(op) +
        ": innermost axis must be contiguous (call .contiguous() first)");
  }
  return t.stride(r - 2);
}

// In-place ops write through a pointer they did not allocate, so a strided
// destination would corrupt whatever lives between its rows.
void require_contiguous(const Tensor& t, const char* op) {
  if (!t.is_contiguous()) {
    throw std::invalid_argument(std::string(op) + ": operand must be contiguous");
  }
}

void same_shape(const Tensor& a, const Tensor& b, const char* op) {
  if (a.shape() != b.shape()) {
    throw std::invalid_argument(std::string(op) + ": " + a.shape().str() + " and " +
                                b.shape().str() + " must have the same shape");
  }
}

struct Rows {
  Tensor t;         // keeps a materialised copy alive, when one was needed
  int M = 0;
  int N = 0;
  int64_t stride = 0;
};

Rows rows_of(const Tensor& x) {
  const int r = x.shape().rank();
  const int N = (r == 0) ? 1 : x.shape().dim(r - 1);
  const int M = (N > 0) ? int(x.numel() / N) : 0;

  if (x.is_contiguous()) return {x, M, N, N};

  const TensorView v = view_of(x);
  if (v.rank == 2 && v.shape[1] == N && v.stride[1] == 1) {
    return {x, M, N, v.stride[0]};
  }
  return {x.contiguous(), M, N, N};
}
}

int normalise_dim(int dim, int rank, const char* op) {
  const int d = (dim < 0) ? dim + rank : dim;
  if (d < 0 || d >= rank) {
    throw std::invalid_argument(std::string(op) + ": axis " + std::to_string(dim) +
                                " is out of range for rank " + std::to_string(rank));
  }
  return d;
}

Tensor matmul(const Tensor& a, const Tensor& b, bool transA, bool transB) {
  same_device(a, b, "matmul");

  if (a.shape().rank() != 2 || b.shape().rank() != 2) {
    throw std::invalid_argument("Both tensors must be 2D");
  }
  int M = transA ? a.shape().dim(1) : a.shape().dim(0);
  int K_a = transA ? a.shape().dim(0) : a.shape().dim(1);
  int K_b = transB ? b.shape().dim(1) : b.shape().dim(0);
  int N = transB ? b.shape().dim(0) : b.shape().dim(1);
  if (K_a != K_b) {
    throw std::invalid_argument("Inner dimensions must match for matrix multiplication");
  }

  const int64_t lda = row_stride_of(a, "matmul (A)");
  const int64_t ldb = row_stride_of(b, "matmul (B)");

  Tensor out(Shape{M, N}, a.device(), a.dtype());
  const auto& k = nn::kernels::kernels(a.device());
  k.gemm(current_stream(a.device()), a.device_ptr(), b.device_ptr(), out.device_ptr(),
         M, N, K_a, lda, ldb, /*ldc=*/N, transA, transB);
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
         M, N, K_a, lda, ldb, ldc, transA, transB);
}

Tensor unary(UnaryOp op, const Tensor& x) {
  Tensor out(x.shape(), x.device(), x.dtype());
  const auto& k = nn::kernels::kernels(x.device());
  k.unary(current_stream(x.device()), op, x.device_ptr(), view_of(x),
          out.device_ptr(), out.numel());
  return out;
}

Tensor unary_backward(UnaryOp op, const Tensor& x, const Tensor& y, const Tensor& g) {
  same_device(x, g, "unary_backward");
  same_shape(x, y, kernels::unary_op_name(op));
  same_shape(x, g, kernels::unary_op_name(op));

  Tensor gx(x.shape(), x.device(), x.dtype());
  const auto& k = nn::kernels::kernels(x.device());
  k.unary_backward(current_stream(x.device()), op,
                   x.device_ptr(), view_of(x),
                   y.device_ptr(), view_of(y),
                   g.device_ptr(), view_of(g),
                   gx.device_ptr(), gx.numel());
  return gx;
}

Tensor binary(BinaryOp op, const Tensor& a, const Tensor& b) {
  same_device(a, b, kernels::binary_op_name(op));

  const Shape out_shape = broadcast_shapes(a.shape(), b.shape());
  const Tensor ea = a.expand(out_shape);
  const Tensor eb = b.expand(out_shape);

  Tensor out(out_shape, a.device(), a.dtype());
  const auto& k = nn::kernels::kernels(a.device());
  k.binary(current_stream(a.device()), op, ea.device_ptr(), view_of(ea),
           eb.device_ptr(), view_of(eb), out.device_ptr(), out.numel());
  return out;
}


Tensor binary_backward(BinaryOp op, int side, const Tensor& a, const Tensor& b,
                       const Tensor& c, const Tensor& g) {
  const char* name = kernels::binary_op_name(op);
  same_shape(c, g, name);

  const Shape& target = (side == 0) ? a.shape() : b.shape();
  if (op == BinaryOp::Add || (op == BinaryOp::Sub && side == 0)) {
    return sum_to(g, target);
  }
  if (op == BinaryOp::Sub) {
    return scalar(ScalarOp::MulScalar, sum_to(g, target), -1.0f);
  }

  const Shape bshape = c.shape();
  const Tensor ea = a.expand(bshape);
  const Tensor eb = b.expand(bshape);

  Tensor full(bshape, c.device(), c.dtype());
  const auto& k = nn::kernels::kernels(c.device());
  k.binary_backward(current_stream(c.device()), op, side,
                    ea.device_ptr(), view_of(ea),
                    eb.device_ptr(), view_of(eb),
                    c.device_ptr(), view_of(c),
                    g.device_ptr(), view_of(g),
                    full.device_ptr(), full.numel());
  return sum_to(full, (side == 0) ? a.shape() : b.shape());
}

Tensor scalar(ScalarOp op, const Tensor& x, float k_value) {
  Tensor out(x.shape(), x.device(), x.dtype());
  const auto& k = nn::kernels::kernels(x.device());
  k.scalar(current_stream(x.device()), op, k_value, x.device_ptr(), view_of(x),
           out.device_ptr(), out.numel());
  return out;
}

Tensor scalar_backward(ScalarOp op, const Tensor& x, const Tensor& y,
                       const Tensor& g, float k_value) {
  same_shape(x, y, kernels::scalar_op_name(op));
  same_shape(x, g, kernels::scalar_op_name(op));

  Tensor gx(x.shape(), x.device(), x.dtype());
  const auto& k = nn::kernels::kernels(x.device());
  k.scalar_backward(current_stream(x.device()), op, k_value,
                    x.device_ptr(), view_of(x),
                    y.device_ptr(), view_of(y),
                    g.device_ptr(), view_of(g),
                    gx.device_ptr(), gx.numel());
  return gx;
}

Tensor relu(const Tensor& x) { return unary(UnaryOp::Relu, x); }

Tensor relu_backward(const Tensor& x, const Tensor& g_out) {
  same_device(x, g_out, "relu_backward");
  same_shape(x, g_out, "relu_backward");
  return unary_backward(UnaryOp::Relu, x, x, g_out);
}

Tensor add(const Tensor& a, const Tensor& b) { return binary(BinaryOp::Add, a, b); }
Tensor mul(const Tensor& a, const Tensor& b) { return binary(BinaryOp::Mul, a, b); }

// Sum g down to `target`, which must broadcast up to g's shape. This is the
// backward of every broadcast: an axis that was stretched to feed many
// outputs collects the gradient from all of them.
Tensor sum_to(const Tensor& g, const Shape& target) {
  if (g.shape() == target) return g;
  if (target.rank() > g.shape().rank()) {
    throw std::invalid_argument("sum_to: " + target.str() + " has more axes than " +
                                g.shape().str());
  }

  const int r = g.shape().rank();
  const int lead = r - target.rank();

  TensorView keep{}, red{};
  keep.rank = red.rank = r;
  int64_t n_out = 1, n_red = 1;
  for (int i = 0; i < r; ++i) {
    const int ti = i - lead;
    const int td = (ti >= 0) ? target.dim(ti) : 1;
    const int gd = g.shape().dim(i);
    if (td != gd && td != 1) {
      throw std::invalid_argument("sum_to: " + g.shape().str() + " does not reduce to " +
                                  target.str());
    }
    const bool reduced = (td == 1 && gd > 1);
    keep.shape[i]  = td;
    keep.stride[i] = g.stride(i);
    red.shape[i]   = reduced ? gd : 1;
    red.stride[i]  = reduced ? g.stride(i) : 0;
    n_out *= td;
    n_red *= red.shape[i];
  }

  Tensor out(target, g.device(), g.dtype());
  const auto& kk = nn::kernels::kernels(g.device());
  kk.sum_to(current_stream(g.device()), g.device_ptr(), keep, red,
            out.device_ptr(), n_out, n_red);
  return out;
}

Tensor sum_dim(const Tensor& x, int dim, bool keepdim) {
  const int r = x.shape().rank();
  const int d = normalise_dim(dim, r, "sum");

  Shape kept = x.shape();
  kept.set_dim(d, 1);
  Tensor out = sum_to(x, kept);
  if (keepdim) return out;

  int dims[kMaxShapeRank] = {0};
  int n = 0;
  for (int i = 0; i < r; ++i) {
    if (i != d) dims[n++] = x.shape().dim(i);
  }
  return out.reshape(Shape(std::span<const int>(dims, n)));
}

Tensor mean_dim(const Tensor& x, int dim, bool keepdim) {
  const int d = normalise_dim(dim, x.shape().rank(), "mean");
  const int n = x.shape().dim(d);
  return scalar(ScalarOp::MulScalar, sum_dim(x, d, keepdim), 1.0f / float(n));
}

Tensor sum_all(const Tensor& x) {
  Tensor out(Shape{}, x.device(), x.dtype());
  Tensor workspace(Shape{nn::kernels::kSumAllWorkspace}, x.device(), x.dtype());

  const auto& k = nn::kernels::kernels(x.device());
  const Stream& s = current_stream(x.device());
  if (x.is_contiguous()) {
    k.sum_all(s, x.device_ptr(), out.device_ptr(), workspace.device_ptr(), x.numel());
  } else {
    k.sum_all_strided(s, x.device_ptr(), view_of(x), out.device_ptr(),
                      workspace.device_ptr(), x.numel());
  }
  return out;
}

Tensor mean_all(const Tensor& x) {
  const int64_t n = x.numel();
  if (n == 0) throw std::invalid_argument("mean: empty tensor");
  return scalar(ScalarOp::MulScalar, sum_all(x), 1.0f / float(n));
}

Tensor softmax_rows(const Tensor& x) {
  const Rows r = rows_of(x);
  Tensor out(x.shape(), x.device(), x.dtype());
  const auto& k = nn::kernels::kernels(x.device());
  k.softmax_rows(current_stream(x.device()), r.t.device_ptr(), out.device_ptr(),
                 r.M, r.N, r.stride);
  return out;
}

Tensor softmax_rows_backward(const Tensor& y, const Tensor& g) {
  same_device(y, g, "softmax_backward");
  same_shape(y, g, "softmax_backward");

  const Rows ry = rows_of(y);
  const Rows rg = rows_of(g);
  Tensor gx(y.shape(), y.device(), y.dtype());
  const auto& k = nn::kernels::kernels(y.device());
  k.softmax_rows_backward(current_stream(y.device()), ry.t.device_ptr(),
                          rg.t.device_ptr(), gx.device_ptr(),
                          ry.M, ry.N, ry.stride, rg.stride);
  return gx;
}

Tensor embedding(const Tensor& weight, const Tensor& idx) {
  same_device(weight, idx, "embedding");

  if (weight.shape().rank() != 2) {
    throw std::invalid_argument("embedding: weight must be [V, D]");
  }
  if (idx.dtype() != DType::I32) {
    throw std::invalid_argument("embedding: indices must be I32");
  }
  if (idx.shape().rank() + 1 > kMaxShapeRank) {
    throw std::invalid_argument("embedding: indices have too many axes");
  }
  require_contiguous(weight, "embedding (weight)");

  const int V = weight.shape().dim(0);
  const int D = weight.shape().dim(1);
  const Tensor flat_idx = idx.contiguous();

  int dims[kMaxShapeRank] = {0};
  const int r = idx.shape().rank();
  for (int i = 0; i < r; ++i) dims[i] = idx.shape().dim(i);
  dims[r] = D;

  Tensor out(Shape(std::span<const int>(dims, r + 1)), weight.device(), weight.dtype());
  const auto& k = nn::kernels::kernels(weight.device());
  k.embedding(current_stream(weight.device()), weight.device_ptr(),
              flat_idx.device_ptr_i32(), out.device_ptr(), idx.numel(), D, V);
  return out;
}

Tensor embedding_backward(const Tensor& g, const Tensor& idx, int V) {
  same_device(g, idx, "embedding_backward");

  const int r = g.shape().rank();
  if (r < 1) throw std::invalid_argument("embedding_backward: gradient has no axes");
  const int D = g.shape().dim(r - 1);
  if (g.numel() != idx.numel() * D) {
    throw std::invalid_argument("embedding_backward: gradient does not match the indices");
  }

  const Tensor dense_g = g.contiguous();
  const Tensor flat_idx = idx.contiguous();

  Tensor gw = Tensor::zeros(Shape{V, D}, g.device(), g.dtype());
  const auto& k = nn::kernels::kernels(g.device());
  k.embedding_backward(current_stream(g.device()), dense_g.device_ptr(),
                       flat_idx.device_ptr_i32(), gw.device_ptr(), idx.numel(), D, V);
  return gw;
}

void add_inplace(Tensor& a, const Tensor& b) {
  same_device(a, b, "add_inplace");
  same_shape(a, b, "add_inplace");
  require_contiguous(a, "add_inplace (destination)");

  const auto& k = nn::kernels::kernels(a.device());
  k.binary(current_stream(a.device()), BinaryOp::Add, a.device_ptr(), view_of(a),
           b.device_ptr(), view_of(b), a.device_ptr(), a.numel());
}

void scale_inplace(Tensor& a, float alpha) {
  require_contiguous(a, "scale_inplace");

  const auto& k = nn::kernels::kernels(a.device());
  k.scalar(current_stream(a.device()), ScalarOp::MulScalar, alpha,
           a.device_ptr(), view_of(a), a.device_ptr(), a.numel());
}

void axpy_inplace(Tensor& y, float alpha, const Tensor& x) {
  same_device(y, x, "axpy_inplace");
  same_shape(y, x, "axpy_inplace");
  require_contiguous(y, "axpy_inplace (destination)");
  require_contiguous(x, "axpy_inplace");

  const auto& k = nn::kernels::kernels(y.device());
  k.axpy(current_stream(y.device()), alpha, x.device_ptr(), y.device_ptr(), y.numel());
}

void fill_inplace(Tensor& a, float v) {
  require_contiguous(a, "fill_inplace");

  const auto& k = nn::kernels::kernels(a.device());
  k.fill(current_stream(a.device()), v, a.device_ptr(), a.numel());
}

void fill_from(Tensor& a, const Tensor& value) {
  same_device(a, value, "fill_from");
  require_contiguous(a, "fill_from");

  if (value.numel() != 1) {
    throw std::invalid_argument("fill_from: value must be a single element");
  }
  if (a.dtype() != value.dtype()) {
    throw std::invalid_argument("fill_from: dtypes must match");
  }

  const auto& k = nn::kernels::kernels(a.device());
  k.fill_from(current_stream(a.device()), value.device_ptr(), a.device_ptr(), a.numel());
}

void softmax_ce(const Tensor& logits, const Tensor& labels, Tensor& loss_out, Tensor& probs) {
  same_device(logits, labels, "softmax_ce");
  same_device(labels, loss_out, "softmax_ce");
  same_device(probs, loss_out, "softmax_ce");

  if (logits.shape().rank() != 2 || labels.shape().rank() != 1) {
    throw std::invalid_argument("logits must be 2D and labels must be 1D");
  }
  if (logits.shape().dim(0) != labels.shape().dim(0)) {
    throw std::invalid_argument("Number of samples in logits and labels must match");
  }
  if (loss_out.shape().rank() != 0) {
    throw std::invalid_argument("loss_out must be a scalar tensor");
  }
  if (probs.shape() != logits.shape()) {
    throw std::invalid_argument("probs must have the same shape as logits");
  }

  require_contiguous(probs, "softmax_ce (probs)");

  const auto& k = nn::kernels::kernels(logits.device());
  k.softmax_ce(current_stream(logits.device()), logits.device_ptr(), labels.device_ptr_i32(),
               loss_out.device_ptr(), probs.device_ptr(),
               logits.shape().dim(0), logits.shape().dim(1),
               row_stride_of(logits, "softmax_ce"));
}

Tensor softmax_ce_backward(const Tensor& probs, const Tensor& labels, const Tensor& g_loss) {
  same_device(probs, labels, "softmax_ce_backward");
  same_device(g_loss, labels, "softmax_ce_backward");

  if (probs.shape().rank() != 2 || labels.shape().rank() != 1) {
    throw std::invalid_argument("probs must be 2D and labels must be 1D");
  }
  if (probs.shape().dim(0) != labels.shape().dim(0)) {
    throw std::invalid_argument("Number of samples in probs and lebls must match");
  }
  if (g_loss.shape().rank() != 0) {
    throw std::invalid_argument("g_loss must be a scalar tensor");
  }

  Tensor g_logits(probs.shape(), probs.device(), probs.dtype());
  const auto& k = nn::kernels::kernels(g_logits.device());
  k.softmax_ce_backward(current_stream(probs.device()), probs.device_ptr(), labels.device_ptr_i32(),
                        g_loss.device_ptr(), g_logits.device_ptr(),
                        probs.shape().dim(0), probs.shape().dim(1),
                        row_stride_of(probs, "softmax_ce_backward"));
  return g_logits;
}

Tensor argmax_rows(const Tensor& x) {
  if (x.shape().rank() != 2) throw std::invalid_argument("argmax rows: x must be 2D");
  Tensor out(Shape{x.shape().dim(0)}, x.device(), DType::I32);
  const auto& k = nn::kernels::kernels(x.device());
  k.argmax_rows(current_stream(x.device()), x.device_ptr(), out.device_ptr_i32(),
                x.shape().dim(0), x.shape().dim(1), row_stride_of(x, "argmax_rows"));
  return out;
}

void adam(const Tensor& p, const Tensor& g, Tensor& m, Tensor& v,
          float lr, float beta1, float beta2, float eps, int step) {
  same_device(p, g, "adam");
  same_device(p, m, "adam");
  same_device(p, v, "adam");

  if (p.shape() != g.shape() || p.shape() != m.shape() || p.shape() != v.shape()) {
    throw std::invalid_argument("All tensors must have the same shape for adam");
  }

  const float bc1 = 1.0f - std::pow(beta1, step);
  const float bc2 = 1.0f - std::pow(beta2, step);

  const auto& k = nn::kernels::kernels(p.device());
  k.adam_step(current_stream(p.device()), p.device_ptr(), g.device_ptr(), m.device_ptr(), v.device_ptr(),
              lr, beta1, beta2, eps, bc1, bc2, p.numel());
}

void copy_strided(const Tensor& dst, const Tensor& src) {
  same_device(dst, src, "copy_strided");

  if (dst.shape() != src.shape()) {
    throw std::invalid_argument("copy_strided: dst and src must have the same shape");
  }
  if (dst.dtype() != src.dtype()) {
    throw std::invalid_argument("copy_strided: dst and src must have the same dtype");
  }
  if (!dst.is_contiguous()) {
    throw std::invalid_argument("copy_strided: dst must be contiguous");
  }

  const auto& k = nn::kernels::kernels(dst.device());
  const Stream& s = current_stream(dst.device());
  const TensorView v = view_of(src);

  switch (src.dtype()) {
    case DType::F32:
      k.copy_strided(s, src.device_ptr(), v, dst.device_ptr(), src.numel());
      break;
    case DType::I32:
      k.copy_strided_i32(s, src.device_ptr_i32(), v, dst.device_ptr_i32(), src.numel());
      break;
  }
}

void copy_into(Tensor& dst, const Tensor& src) {
  same_device(dst, src, "copy_into");

  if (dst.shape() != src.shape()) {
    throw std::invalid_argument("copy_into: dst and src must have the same shape");
  }
  if (dst.dtype() != src.dtype()) {
    throw std::invalid_argument("copy_into: dst and src must have the same dtype");
  }
  if (dst.dtype() != DType::F32) {
    throw std::invalid_argument("copy_into: only F32 is supported");
  }

  const Tensor packed = src.contiguous();

  const auto& k = nn::kernels::kernels(dst.device());
  k.copy_into_strided(current_stream(dst.device()), packed.device_ptr(),
                      dst.device_ptr(), view_of(dst), dst.numel());
}

}
