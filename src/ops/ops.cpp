#include <nn/ops/ops.h>

#include <stdexcept>

#include <nn/kernels/kernel_api.h>

namespace nn::ops {

Tensor matmul(const Tensor& a, const Tensor& b, bool transA, bool transB) {
  if (a.device() != b.device()) {
    throw std::invalid_argument("Tensors must be on the same device");
  }
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
  k.gemm(a.data(), b.data(), out.data(), M, N, K_a, transA, transB);
  return out;
}

Tensor add_row_bias(const Tensor& x, const Tensor& bias) {
  if (x.device() != bias.device()) {
    throw std::invalid_argument("Tensors must be on the same device");
  }
  if (x.shape().rank() != 2 || bias.shape().rank() != 1) {
    throw std::invalid_argument("x must be 2D and bias must be 1D");
  }
  if (x.shape().dim(1) != bias.shape().dim(0)) {
    throw std::invalid_argument("Bias dimension must match x's second dimension");
  }

  Tensor out(x.shape(), x.device(), x.dtype());
  const auto& k = nn::kernels::kernels(x.device());
  k.add_row_bias(x.data(), bias.data(), out.data(), x.shape().dim(0), x.shape().dim(1));
  return out;
}

Tensor col_sum(const Tensor& x) {
  if (x.shape().rank() != 2) {
    throw std::invalid_argument("x must be 2D");
  }

  Tensor out(Shape{x.shape().dim(1)}, x.device(), x.dtype());
  const auto& k = nn::kernels::kernels(x.device());
  k.col_sum(x.data(), out.data(), x.shape().dim(0), x.shape().dim(1));
  return out;
}

Tensor relu(const Tensor& x) {
  Tensor out(x.shape(), x.device(), x.dtype());
  const auto& k = nn::kernels::kernels(x.device());
  k.relu(x.data(), out.data(), x.numel());
  return out;
}

Tensor relu_backward(const Tensor& x, const Tensor& g_out) {
  if (x.shape() != g_out.shape()) {
    throw std::invalid_argument("x and g_out must have the same shape");
  }

  Tensor g_x(x.shape(), x.device(), x.dtype());
  const auto& k = nn::kernels::kernels(x.device());
  k.relu_backward(x.data(), g_out.data(), g_x.data(), x.numel());
  return g_x;
}

Tensor add(const Tensor& a, const Tensor& b) {
  if (a.shape() != b.shape()) {
    throw std::invalid_argument("Tensors must have the same shape for addition");
  }

  Tensor out(a.shape(), a.device(), a.dtype());
  const auto& k = nn::kernels::kernels(a.device());
  k.add(a.data(), b.data(), out.data(), a.numel());
  return out;
}

void add_inplace(Tensor& a, const Tensor& b) {
  if (a.shape() != b.shape()) {
    throw std::invalid_argument("Tensors must have the same shape for addition");
  }

  const auto& k = nn::kernels::kernels(a.device());
  k.add(a.data(), b.data(), a.data(), a.numel());
}

void scale_inplace(Tensor& a, float alpha) {
  const auto& k = nn::kernels::kernels(a.device());
  k.scale(alpha, a.data(), a.numel());
}

void axpy_inplace(Tensor& y, float alpha, const Tensor& x) {
  if (y.shape() != x.shape()) {
    throw std::invalid_argument("Tensors must have the same shape for axpy");
  }

  const auto& k = nn::kernels::kernels(y.device());
  k.axpy(alpha, x.data(), y.data(), y.numel());
}

void fill_inplace(Tensor& a, float v) {
  const auto& k = nn::kernels::kernels(a.device());
  k.fill(v, a.data(), a.numel());
}

void softmax_ce(const Tensor& logits, const Tensor& labels, Tensor& loss_out, Tensor& probs) {
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
  k.softmax_ce(logits.data(), labels.data_i32(), loss_out.data(), probs.data(), logits.shape().dim(0), logits.shape().dim(1));
}

}