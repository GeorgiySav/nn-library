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

  Tensor view_like(const Shape& s, const Strides&, int64_t offset) const;

  const Shape& shape() const { return shape_; }
  int64_t numel() const { return shape_.numel(); }
  Device device() const { return storage_->device(); }
  DType dtype() const { return dtype_; }
  bool defined() const { return storage_ != nullptr; }

  const Strides& strides() const { return strides_; }
  int64_t stride(int i) const { return strides_.at(i); }
  int64_t offset() const { return offset_; }

  Tensor contiguous() const;
  bool is_contiguous() const;

  Tensor permute(std::span<const int> order) const;
  Tensor transpose(int a, int b) const;
  Tensor reshape(Shape s) const;
  Tensor slice(int axis, int64_t start, int64_t len) const;

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

private:
  std::shared_ptr<Storage> storage_;
  Shape shape_;
  Strides strides_;
  int64_t offset_ = 0;
  DType dtype_ = DType::F32;
  mutable std::shared_ptr<AutogradMeta> meta_;
};

TensorView view_of(const Tensor& t);

// Human-readable dump: header line (shape, dtype, device, and strides when they
// are not the contiguous ones) followed by a numpy-style body with the middle
// of any axis longer than 2*edge elided. Works on any device and any layout.
std::string to_string(const Tensor& t, int edge = 3);
std::ostream& operator<<(std::ostream& os, const Tensor& t);

struct AutogradMeta {
  bool requires_grad = false;
  Tensor grad;
  int node_id = -1;
  uint64_t tape_epoch = 0; // 0 == never recorded on any tape
};

}