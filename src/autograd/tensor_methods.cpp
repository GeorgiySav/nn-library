// Tensor's own differentiable surface, and the operators. Every definition
// here is a one-line forward to nn::autograd -- the method spellings live
// separately from the functions they call so that adding an op means touching
// the .def file and one implementation file, not both halves of a 500-line
// translation unit.

#include <nn/autograd/functions.h>

#include <initializer_list>
#include <span>

#include <nn/core/tensor.h>

namespace nn {

#define NN_UNARY(Name, method) \
  Tensor Tensor::method() const { return autograd::unary(kernels::UnaryOp::Name, *this); }
#include <nn/ops/unary_ops.def>
#undef NN_UNARY

#define NN_BINARY(Name, method)                                  \
  Tensor Tensor::method(const Tensor& other) const {             \
    return autograd::binary(kernels::BinaryOp::Name, *this, other); }
#include <nn/ops/binary_ops.def>
#undef NN_BINARY

#define NN_SCALAR(Name, method) \
  Tensor Tensor::method(float k) const { return autograd::scalar(kernels::ScalarOp::Name, *this, k); }
#include <nn/ops/scalar_ops.def>
#undef NN_SCALAR

Tensor Tensor::contiguous() const { return autograd::contiguous(*this); }
Tensor Tensor::expand(const Shape& to) const { return autograd::expand(*this, to); }
Tensor Tensor::permute(std::span<const int> order) const {
  return autograd::permute(*this, order);
}
Tensor Tensor::permute(std::initializer_list<int> order) const {
  return autograd::permute(*this, std::span<const int>(order.begin(), order.size()));
}
Tensor Tensor::transpose(int a, int b) const { return autograd::transpose(*this, a, b); }
Tensor Tensor::reshape(Shape s) const { return autograd::reshape(*this, s); }
Tensor Tensor::slice(int axis, int64_t start, int64_t len) const {
  return autograd::slice(*this, axis, start, len);
}

Tensor Tensor::pow(float e) const { return pow_scalar(e); }
Tensor Tensor::mm(const Tensor& other, bool transB, bool transA) const {
  return autograd::matmul(*this, other, transA, transB);
}
Tensor Tensor::t() const { return autograd::transpose(*this, -2, -1); }
Tensor Tensor::softmax() const { return autograd::softmax(*this); }

Tensor Tensor::sum() const { return autograd::sum_all(*this); }
Tensor Tensor::sum(int dim, bool keepdim) const { return autograd::sum(*this, dim, keepdim); }
Tensor Tensor::mean() const { return autograd::mean(*this); }
Tensor Tensor::mean(int dim, bool keepdim) const { return autograd::mean(*this, dim, keepdim); }
Tensor Tensor::var(int dim, bool keepdim, bool unbiased) const {
  return autograd::var(*this, dim, keepdim, unbiased);
}

Tensor operator+(const Tensor& a, const Tensor& b) { return autograd::add(a, b); }
Tensor operator-(const Tensor& a, const Tensor& b) { return autograd::sub(a, b); }
Tensor operator*(const Tensor& a, const Tensor& b) { return autograd::mul(a, b); }
Tensor operator/(const Tensor& a, const Tensor& b) { return autograd::div(a, b); }

Tensor operator+(const Tensor& a, float k) { return autograd::add_scalar(a, k); }
Tensor operator+(float k, const Tensor& a) { return autograd::add_scalar(a, k); }
Tensor operator-(const Tensor& a, float k) { return autograd::add_scalar(a, -k); }
Tensor operator-(float k, const Tensor& a) { return autograd::rsub_scalar(a, k); }
Tensor operator*(const Tensor& a, float k) { return autograd::mul_scalar(a, k); }
Tensor operator*(float k, const Tensor& a) { return autograd::mul_scalar(a, k); }
Tensor operator/(const Tensor& a, float k) { return autograd::mul_scalar(a, 1.0f / k); }
Tensor operator/(float k, const Tensor& a) { return autograd::rdiv_scalar(a, k); }

Tensor operator-(const Tensor& a) { return autograd::neg(a); }

}  // namespace nn
