#include <nn/ops/ops.h>

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

Tensor add_row_bias(const Tensor& x, const Tensor& bias) {
  same_device(x, bias, "add_row_bias");

  if (x.shape().rank() != 2 || bias.shape().rank() != 1) {
    throw std::invalid_argument("x must be 2D and bias must be 1D");
  }
  if (x.shape().dim(1) != bias.shape().dim(0)) {
    throw std::invalid_argument("Bias dimension must match x's second dimension");
  }

  Tensor out(x.shape(), x.device(), x.dtype());
  const auto& k = nn::kernels::kernels(x.device());
  k.add_row_bias(current_stream(x.device()), x.device_ptr(), bias.device_ptr(), out.device_ptr(),
                 x.shape().dim(0), x.shape().dim(1), row_stride_of(x, "add_row_bias"));
  return out;
}

Tensor col_sum(const Tensor& x) {
  if (x.shape().rank() != 2) {
    throw std::invalid_argument("x must be 2D");
  }

  Tensor out(Shape{x.shape().dim(1)}, x.device(), x.dtype());
  const auto& k = nn::kernels::kernels(x.device());
  k.col_sum(current_stream(x.device()), x.device_ptr(), out.device_ptr(),
            x.shape().dim(0), x.shape().dim(1), row_stride_of(x, "col_sum"));
  return out;
}

Tensor relu(const Tensor& x) {
  Tensor out(x.shape(), x.device(), x.dtype());
  const auto& k = nn::kernels::kernels(x.device());
  const Stream& s = current_stream(x.device());
  if (x.is_contiguous()) {
    k.relu(s, x.device_ptr(), out.device_ptr(), x.numel());
  } else {
    k.relu_strided(s, x.device_ptr(), view_of(x), out.device_ptr(), x.numel());
  }
  return out;
}

Tensor relu_backward(const Tensor& x, const Tensor& g_out) {
  same_device(x, g_out, "relu_backward");

  if (x.shape() != g_out.shape()) {
    throw std::invalid_argument("x and g_out must have the same shape");
  }

  Tensor g_x(x.shape(), x.device(), x.dtype());
  const auto& k = nn::kernels::kernels(x.device());
  const Stream& s = current_stream(x.device());
  
  if (x.is_contiguous() && g_out.is_contiguous()) {
    k.relu_backward(s, x.device_ptr(), g_out.device_ptr(), g_x.device_ptr(), x.numel());
  } else {
    k.relu_backward_strided(s, x.device_ptr(), view_of(x),
                            g_out.device_ptr(), view_of(g_out),
                            g_x.device_ptr(), x.numel());
  }
  return g_x;
}

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

Tensor add(const Tensor& a, const Tensor& b) {
  same_device(a, b, "add");

  const Shape out_shape = broadcast_shapes(a.shape(), b.shape());
  const Tensor ea = a.expand(out_shape);
  const Tensor eb = b.expand(out_shape);

  Tensor out(out_shape, a.device(), a.dtype());
  const auto& k = nn::kernels::kernels(a.device());
  const Stream& s = current_stream(a.device());
  // Equal shapes expand to themselves, so the dense path is unchanged.
  if (ea.is_contiguous() && eb.is_contiguous()) {
    k.add(s, ea.device_ptr(), eb.device_ptr(), out.device_ptr(), out.numel());
  } else {
    k.add_strided(s, ea.device_ptr(), view_of(ea), eb.device_ptr(), view_of(eb),
                  out.device_ptr(), out.numel());
  }
  return out;
}

void add_inplace(Tensor& a, const Tensor& b) {
  same_device(a, b, "add_inplace");

  if (a.shape() != b.shape()) {
    throw std::invalid_argument("Tensors must have the same shape for addition");
  }
  require_contiguous(a, "add_inplace (destination)");
  require_contiguous(b, "add_inplace");

  const auto& k = nn::kernels::kernels(a.device());
  k.add(current_stream(a.device()), a.device_ptr(), b.device_ptr(), a.device_ptr(), a.numel());
}

void scale_inplace(Tensor& a, float alpha) {
  require_contiguous(a, "scale_inplace");

  const auto& k = nn::kernels::kernels(a.device());
  k.scale(current_stream(a.device()), alpha, a.device_ptr(), a.numel());
}

void axpy_inplace(Tensor& y, float alpha, const Tensor& x) {
  same_device(y, x, "axpy_inplace");

  if (y.shape() != x.shape()) {
    throw std::invalid_argument("Tensors must have the same shape for axpy");
  }
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

}
