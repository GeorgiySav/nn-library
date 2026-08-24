#pragma once

#include <memory>
#include <iosfwd>
#include <span>
#include <string>

#include <nn/core/rng.h>
#include <nn/core/dtype.h>
#include <nn/core/shape.h>
#include <nn/core/storage.h>
#include <nn/core/strides.h>

namespace nn {

struct AutogradMeta;

class Tensor {
public:
  Tensor() = default;
  Tensor(Shape s, Device d = Device::CPU, DType t = DType::F32);

  static Tensor zeros(Shape s, Device d = Device::CPU, DType t = DType::F32);
  static Tensor full(Shape s, float value, Device d = Device::CPU, DType t = DType::F32);
  static Tensor randn(Shape s, Pcg32& rng, float stddev, Device d = Device::CPU, DType t = DType::F32);
  static Tensor scalar(float v, Device d = Device::CPU, DType t = DType::F32);
  static Tensor from(std::initializer_list<float> values, Device d = Device::CPU, DType t = DType::F32);
  static Tensor from(std::initializer_list<std::initializer_list<float>> rows, Device d = Device::CPU, DType t = DType::F32);
  static Tensor from_i32(std::initializer_list<int32_t> values, Device d = Device::CPU, DType t = DType::I32);

  static Tensor from(std::span<const float> values, Shape s, Device d = Device::CPU, DType t = DType::F32);
  static Tensor from_i32(std::span<const int32_t> values, Shape s, Device d = Device::CPU, DType t = DType::I32);

  // [0, 1, ..., n-1], rank 1.
  static Tensor arange(int64_t n, Device d = Device::CPU, DType t = DType::I32);

  Tensor view_like(const Shape& s, const Strides&, int64_t offset) const;

  const Shape& shape() const { return shape_; }
  int64_t numel() const { return shape_.numel(); }
  Device device() const { return storage_->device(); }
  DType dtype() const { return dtype_; }
  bool defined() const { return storage_ != nullptr; }

  const Strides& strides() const { return strides_; }
  int64_t stride(int i) const { return strides_.at(i); }
  int64_t offset() const { return offset_; }

  // A dense copy of this tensor, or *this when it is already dense
  Tensor pack() const;
  bool is_contiguous() const;

  // raw views
  Tensor expand_view(const Shape& to) const;
  Tensor permute_view(std::span<const int> order) const;
  Tensor permute_view(std::initializer_list<int> order) const;
  Tensor transpose_view(int a, int b) const;
  Tensor reshape_view(Shape s) const;
  Tensor slice_view(int axis, int64_t start, int64_t len) const;

  // Address in the owning device's address space. NOT dereferenceable on the
  // host unless device() == Device::CPU. For passing to kernels only.
  float* device_ptr() const;
  int32_t* device_ptr_i32() const;

  // Dereferenceable host pointer. Requires device() == Device::CPU. For a
  // device tensor, call .to(Device::CPU) and keep the result alive.
  float* host_data() const;
  int32_t* host_data_i32() const;

  // Untyped device-space address. Only copy_bytes should consume this.
  void* raw();
  const void* raw() const;

  float item() const;
  // Convenience for printf/debuggers; see to_string below.
  std::string str() const;

  Tensor to(Device d) const;
  Tensor clone() const;

  bool requires_grad() const;
  void set_requires_grad(bool requires_grad);
  AutogradMeta* meta() const { return meta_.get(); }
  const std::shared_ptr<AutogradMeta>& ensure_meta_shared() const;
  AutogradMeta& ensure_meta() const { return *ensure_meta_shared(); };
  Tensor& grad();
  void zero_grad();

  // Seeds this scalar with a gradient of 1 and walks the tape that produced it
  // back to the leaves. The tape does not have to still be the active one, but
  // it does have to still exist and not have been cleared.
  void backward(bool retain_graph = false) const;

  // --- differentiable ops -------------------------------------------------
  //
  // Every method below routes through nn::autograd, so it records itself on
  // the active tape and is a plain kernel call when there is none. The
  // elementwise ones are generated from the same three lists the kernels are:
  // adding a line to unary_ops.def adds the kernel, the autograd node, the
  // free function and this method at once.
#define NN_UNARY(Name, method, fwd, bwd) Tensor method() const;
#include <nn/kernels/unary_ops.def>
#undef NN_UNARY

#define NN_BINARY(Name, method, fwd, da, db) Tensor method(const Tensor& other) const;
#include <nn/kernels/binary_ops.def>
#undef NN_BINARY

#define NN_SCALAR(Name, method, fwd, bwd) Tensor method(float k) const;
#include <nn/kernels/scalar_ops.def>
#undef NN_SCALAR

  Tensor contiguous() const;                 // pack(), plus identity backward
  Tensor expand(const Shape& to) const;
  Tensor permute(std::span<const int> order) const;
  Tensor permute(std::initializer_list<int> order) const;
  Tensor transpose(int a, int b) const;
  Tensor reshape(Shape s) const;
  Tensor slice(int axis, int64_t start, int64_t len) const;

  Tensor pow(float e) const;                 // the scalar form, spelled better
  Tensor mm(const Tensor& other) const;      // 2-D matrix product
  Tensor t() const;                          // swap the last two axes
  Tensor softmax() const;                    // over the last axis

  Tensor sum() const;
  Tensor sum(int dim, bool keepdim = false) const;
  Tensor mean() const;
  Tensor mean(int dim, bool keepdim = false) const;
  Tensor var(int dim, bool keepdim = false, bool unbiased = true) const;

private:
  std::shared_ptr<Storage> storage_;
  Shape shape_;
  Strides strides_;
  int64_t offset_ = 0;
  DType dtype_ = DType::F32;
  mutable std::shared_ptr<AutogradMeta> meta_;
};

TensorView view_of(const Tensor& t);

Tensor tril_mask(int n, Device d = Device::CPU);

// Operators, so an expression reads left to right instead of inside out.
// All of them are differentiable; the float forms avoid allocating and
// uploading a one-element tensor just to broadcast it.
Tensor operator+(const Tensor& a, const Tensor& b);
Tensor operator-(const Tensor& a, const Tensor& b);
Tensor operator*(const Tensor& a, const Tensor& b);
Tensor operator/(const Tensor& a, const Tensor& b);

Tensor operator+(const Tensor& a, float k);
Tensor operator+(float k, const Tensor& a);
Tensor operator-(const Tensor& a, float k);
Tensor operator-(float k, const Tensor& a);
Tensor operator*(const Tensor& a, float k);
Tensor operator*(float k, const Tensor& a);
Tensor operator/(const Tensor& a, float k);
Tensor operator/(float k, const Tensor& a);

Tensor operator-(const Tensor& a);

std::string to_string(const Tensor& t, int edge = 3);
std::ostream& operator<<(std::ostream& os, const Tensor& t);

struct AutogradMeta {
  bool requires_grad = false;
  Tensor grad;
  int node_id = -1;
  uint64_t tape_epoch = 0; // 0 == never recorded on any tape
};

}