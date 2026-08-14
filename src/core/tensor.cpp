#include <nn/core/tensor.h>

#include <cassert>

namespace nn {

Tensor::Tensor(Shape s, Device d, DType t) : shape_(s), dtype_(t) {
  size_t bytes = shape_.numel() * dtype_size(dtype_);
  storage_ = std::make_shared<Storage>(bytes, d);
}

Tensor Tensor::zeros(Shape s, Device d, DType t) {
  Tensor z(s, d, t);
  std::memset(z.raw(), 0, z.numel() * dtype_size(t));
  return z;
}

Tensor Tensor::full(Shape s, float value, Device d, DType t) {
  Tensor f(s, d, t);
  float* data = f.data();
  for (size_t i{0u}; i < f.numel(); ++i) {
    data[i] = value;
  }
  return f;
}

Tensor Tensor::randn(Shape s, Pcg32& rng, float stddev, Device d, DType t) {
  Tensor r(s, d, t);
  float* data = r.data();
  for (size_t i{0u}; i < r.numel(); ++i) {
    data[i] = rng.next_normal() * stddev;
  }
  return r;
}

Tensor Tensor::scalar(float v, Device d, DType t) {
  Tensor s({}, d, t);
  float* data = s.data();
  data[0] = v;
  return s;
}

float* Tensor::data() const {
  return static_cast<float*>(storage_->data());
}

int32_t* Tensor::data_i32() {
  return static_cast<int32_t*>(storage_->data());
}

const int32_t* Tensor::data_i32() const {
  return static_cast<const int32_t*>(storage_->data());
}

void* Tensor::raw() {
  return storage_->data();
}

const void* Tensor::raw() const {
  return storage_->data();
}

float Tensor::item() const {
  assert(numel() == 1);
  return data()[0];
}

Tensor Tensor::to(Device d) const {
  if (device() == d) {
    return *this;
  }
  Tensor t(shape_, d, dtype_);
  std::memcpy(t.raw(), raw(), numel() * dtype_size(dtype_));
  return t;
}

Tensor Tensor::clone() const {
  Tensor t(shape_, device(), dtype_);
  std::memcpy(t.raw(), raw(), numel() * dtype_size(dtype_));
  return t;
}

bool Tensor::requires_grad() const {
  return meta_ && meta_->requires_grad;
}

void Tensor::set_requires_grad(bool requires_grad) {
  ensure_meta().requires_grad = requires_grad;
}

AutogradMeta& Tensor::ensure_meta() {
  if (!meta_) {
    meta_ = std::make_shared<AutogradMeta>();
    meta_->grad = Tensor::zeros(shape_, device(), dtype_);
  }
  return *meta_;
}

Tensor& Tensor::grad() {
  return ensure_meta().grad;
}

void Tensor::zero_grad() {
  if (meta_ && meta_->grad.defined()) {
    std::memset(meta_->grad.raw(), 0, meta_->grad.numel() * dtype_size(meta_->grad.dtype()));
  }
}

}