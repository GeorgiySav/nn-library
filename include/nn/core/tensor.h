#pragma once

#include <memory>

#include "rng.h"
#include "dtype.h"
#include "shape.h"
#include "storage.h"

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

  const Shape& shape() const { return shape_; }
  int64_t numel() const { return shape_.numel(); }
  Device device() const { return storage_->device(); }
  DType dtype() const { return dtype_; }
  bool defined() const { return storage_ != nullptr; }

  float* data() const;
  int32_t* data_i32();
  const int32_t* data_i32() const;
  void* raw();
  const void* raw() const;
  float item() const;

  Tensor to(Device d) const;
  Tensor clone() const;

  bool requires_grad() const;
  void set_requires_grad(bool requires_grad);
  AutogradMeta* meta() const { return meta_.get(); }
  AutogradMeta& ensure_meta();
  Tensor& grad();
  void zero_grad();

private:
  std::shared_ptr<Storage> storage_;
  Shape shape_;
  DType dtype_ = DType::F32;
  std::shared_ptr<AutogradMeta> meta_;
};

struct AutogradMeta {
  bool requires_grad = false;
  Tensor grad;
  int node_id = -1;
};

}