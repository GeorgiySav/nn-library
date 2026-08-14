#include <nn/core/tensor.h>

#include <cassert>
#include <stdexcept>

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

Tensor Tensor::from(std::initializer_list<float> values, Device d, DType t) {
  Tensor r(Shape({static_cast<int>(values.size())}), d, t);
  float* out = r.data();
  size_t i = 0;
  for (float v : values) {
    out[i++] = v;
  }
  return r;
}

Tensor Tensor::from(std::initializer_list<std::initializer_list<float>> rows, Device d, DType t) {
  Tensor r(Shape({static_cast<int>(rows.size()),
                   rows.size() ? static_cast<int>(rows.begin()->size()) : 0}),
           d, t);
  float* out = r.data();
  size_t i = 0;
  for (const auto& row : rows) {
    assert(row.size() == rows.begin()->size() && "ragged nested initializer list");
    for (float v : row) {
      out[i++] = v;
    }
  }
  return r;
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
  if (d == Device::CUDA) {
    throw std::runtime_error("CUDA device not supported yet");
  }
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
  ensure_meta();
  std::memset(meta_->grad.raw(), 0, meta_->grad.numel() * dtype_size(dtype_));
}

}