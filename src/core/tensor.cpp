#include <nn/core/tensor.h>

#include <cassert>
#include <stdexcept>

#include <nn/core/allocator.h>

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
  float* data = f.host_data();
  for (size_t i{0u}; i < f.numel(); ++i) {
    data[i] = value;
  }
  return f;
}

Tensor Tensor::randn(Shape s, Pcg32& rng, float stddev, Device d, DType t) {
  Tensor r(s, d, t);
  float* data = r.host_data();
  for (size_t i{0u}; i < r.numel(); ++i) {
    data[i] = rng.next_normal() * stddev;
  }
  return r;
}

Tensor Tensor::scalar(float v, Device d, DType t) {
  Tensor s({}, d, t);
  float* data = s.host_data();
  data[0] = v;
  return s;
}

Tensor Tensor::from(std::initializer_list<float> values, Device d, DType t) {
  Tensor r(Shape({static_cast<int>(values.size())}), d, t);
  float* out = r.host_data();
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
  float* out = r.host_data();
  size_t i = 0;
  for (const auto& row : rows) {
    assert(row.size() == rows.begin()->size() && "ragged nested initializer list");
    for (float v : row) {
      out[i++] = v;
    }
  }
  return r;
}

Tensor Tensor::from_i32(std::initializer_list<int32_t> values, Device d, DType t) {
  Tensor r(Shape({static_cast<int>(values.size())}), d, t);
  int32_t* out = r.host_data_i32();
  size_t i = 0;
  for (int32_t v : values) {
    out[i++] = v;
  }
  return r;
}

float* Tensor::device_ptr() const {
  assert(dtype_ == DType::F32);
  return static_cast<float*>(storage_->data());
}

int32_t* Tensor::device_ptr_i32() const {
  assert(dtype_ == DType::I32);
  return static_cast<int32_t*>(storage_->data());
}

float* Tensor::host_data() const {
  assert(device() == Device::CPU &&
         "host_data() on a device tensor; use .to(Device::CPU)");
  return device_ptr();
}

int32_t* Tensor::host_data_i32() const {
  assert(device() == Device::CPU &&
         "host_data_i32() on a device tensor; use .to(Device::CPU)");
  return device_ptr_i32();
}

void* Tensor::raw() {
  return storage_->data();
}

const void* Tensor::raw() const {
  return storage_->data();
}

float Tensor::item() const {
  assert(numel() == 1);
  if (device() != Device::CPU) return to(Device::CPU).item();
  return host_data()[0];
}

Tensor Tensor::to(Device d) const {
  if (!storage_) return Tensor{};
  if (device() == d) return *this;

  Tensor out(shape_, d, dtype_);
  copy_bytes(out.raw(), d, raw(), device(),
             size_t(numel()) * dtype_size(dtype_));
  
  return out;
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
  AutogradMeta& m = ensure_meta();
  m.requires_grad = requires_grad;
  // parameters get one persistent buffer, reused for life
  if (requires_grad && !m.grad.defined()) {
    m.grad = Tensor::zeros(shape_, device(), dtype_);
  }
}

const std::shared_ptr<AutogradMeta>& Tensor::ensure_meta_shared() const {
  if (!meta_) {
    meta_ = std::make_shared<AutogradMeta>();
  }
  return meta_;
}

Tensor& Tensor::grad() {
  AutogradMeta& m = ensure_meta();
  if (!m.grad.defined()) m.grad = Tensor::zeros(shape_, device(), dtype_);
  return m.grad;
}

void Tensor::zero_grad() {
  if (meta_ && meta_->grad.defined())
    std::memset(meta_->grad.raw(), 0, meta_->grad.numel() * dtype_size(dtype_));
}

}