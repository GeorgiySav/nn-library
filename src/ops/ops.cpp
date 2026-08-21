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

  Tensor out(Shape{M, N}, a.device(), a.dtype());
  const auto& k = nn::kernels::kernels(a.device());
  k.gemm(current_stream(a.device()), a.device_ptr(), b.device_ptr(), out.device_ptr(), M, N, K_a, transA, transB);
  return out;
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
  k.add_row_bias(current_stream(x.device()), x.device_ptr(), bias.device_ptr(), out.device_ptr(), x.shape().dim(0), x.shape().dim(1));
  return out;
}

Tensor col_sum(const Tensor& x) {
  if (x.shape().rank() != 2) {
    throw std::invalid_argument("x must be 2D");
  }

  Tensor out(Shape{x.shape().dim(1)}, x.device(), x.dtype());
  const auto& k = nn::kernels::kernels(x.device());
  k.col_sum(current_stream(x.device()), x.device_ptr(), out.device_ptr(), x.shape().dim(0), x.shape().dim(1));
  return out;
}

Tensor relu(const Tensor& x) {
  Tensor out(x.shape(), x.device(), x.dtype());
  const auto& k = nn::kernels::kernels(x.device());
  k.relu(current_stream(x.device()), x.device_ptr(), out.device_ptr(), x.numel());
  return out;
}

Tensor relu_backward(const Tensor& x, const Tensor& g_out) {
  same_device(x, g_out, "relu_backward");
  
  if (x.shape() != g_out.shape()) {
    throw std::invalid_argument("x and g_out must have the same shape");
  }

  Tensor g_x(x.shape(), x.device(), x.dtype());
  const auto& k = nn::kernels::kernels(x.device());
  k.relu_backward(current_stream(x.device()), x.device_ptr(), g_out.device_ptr(), g_x.device_ptr(), x.numel());
  return g_x;
}

Tensor add(const Tensor& a, const Tensor& b) {
  same_device(a, b, "add");

  if (a.shape() != b.shape()) {
    throw std::invalid_argument("Tensors must have the same shape for addition");
  }

  Tensor out(a.shape(), a.device(), a.dtype());
  const auto& k = nn::kernels::kernels(a.device());
  k.add(current_stream(a.device()), a.device_ptr(), b.device_ptr(), out.device_ptr(), a.numel());
  return out;
}

void add_inplace(Tensor& a, const Tensor& b) {
  same_device(a, b, "add_inplace");

  if (a.shape() != b.shape()) {
    throw std::invalid_argument("Tensors must have the same shape for addition");
  }

  const auto& k = nn::kernels::kernels(a.device());
  k.add(current_stream(a.device()), a.device_ptr(), b.device_ptr(), a.device_ptr(), a.numel());
}

void scale_inplace(Tensor& a, float alpha) {
  const auto& k = nn::kernels::kernels(a.device());
  k.scale(current_stream(a.device()), alpha, a.device_ptr(), a.numel());
}

void axpy_inplace(Tensor& y, float alpha, const Tensor& x) {
  same_device(y, x, "axpy_inplace");

  if (y.shape() != x.shape()) {
    throw std::invalid_argument("Tensors must have the same shape for axpy");
  }

  const auto& k = nn::kernels::kernels(y.device());
  k.axpy(current_stream(y.device()), alpha, x.device_ptr(), y.device_ptr(), y.numel());
}

void fill_inplace(Tensor& a, float v) {
  const auto& k = nn::kernels::kernels(a.device());
  k.fill(current_stream(a.device()), v, a.device_ptr(), a.numel());
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

  const auto& k = nn::kernels::kernels(logits.device());
  k.softmax_ce(current_stream(logits.device()), logits.device_ptr(), labels.device_ptr_i32(), loss_out.device_ptr(), probs.device_ptr(), logits.shape().dim(0), logits.shape().dim(1));
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
  k.softmax_ce_backward(current_stream(probs.device()), probs.device_ptr(), labels.device_ptr_i32(), g_loss.device_ptr(),
                        g_logits.device_ptr(), probs.shape().dim(0), probs.shape().dim(1));
  return g_logits;
}

Tensor argmax_rows(const Tensor& x) {
  if (x.shape().rank() != 2) throw std::invalid_argument("argmax rows: x must be 2D");
  Tensor out(Shape{x.shape().dim(0)}, x.device(), DType::I32);
  const auto& k = nn::kernels::kernels(x.device());
  k.argmax_rows(current_stream(x.device()), x.device_ptr(), out.device_ptr_i32(), x.shape().dim(0), x.shape().dim(1));
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

}