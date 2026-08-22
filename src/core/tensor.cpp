#include <nn/core/tensor.h>

#include <cassert>
#include <stdexcept>
#include <vector>

#include <nn/core/allocator.h>
#include <nn/ops/ops.h>

namespace nn {

namespace {

template <class Init>
Tensor host_init(Shape s, Device d, DType t, Init&& init) {
  Tensor h(s, Device::CPU, t);
  init(h);
  return h.to(d);
}

}

Tensor::Tensor(Shape s, Device d, DType t) : shape_(s), dtype_(t) {
  strides_ = Strides::contiguous_for(shape_);
  size_t bytes = shape_.numel() * dtype_size(dtype_);
  storage_ = std::make_shared<Storage>(bytes, d);
}

Tensor Tensor::zeros(Shape s, Device d, DType t) {
  Tensor z(s, d, t);
  memset_bytes(z.raw(), d, 0, z.numel() * dtype_size(t));
  return z;
}

Tensor Tensor::full(Shape s, float value, Device d, DType t) {
  return host_init(s, d, t, [value](Tensor& h) {
    float* data = h.host_data();
    for (size_t i{0u}; i < h.numel(); ++i) data[i] = value;
  });
}

Tensor Tensor::randn(Shape s, Pcg32& rng, float stddev, Device d, DType t) {
  return host_init(s, d, t, [&](Tensor& h) {
    float* data = h.host_data();
    for (size_t i{0u}; i < h.numel(); ++i) data[i] = rng.next_normal() * stddev;
  });
}

Tensor Tensor::scalar(float v, Device d, DType t) {
  return host_init({}, d, t, [&](Tensor& h) { h.host_data()[0] = v; });
}

Tensor Tensor::from(std::span<const float> values, Shape s, Device d, DType t) {
  assert(static_cast<int>(values.size()) == s.numel());
  Tensor r(s, d, t);
  copy_bytes(r.raw(), d, values.data(), Device::CPU,
             values.size() * sizeof(float));
  return r;
}

Tensor Tensor::from_i32(std::span<const int32_t> values, Shape s, Device d, DType t) {
  assert(static_cast<int>(values.size()) == s.numel());
  Tensor r(s, d, t);
  copy_bytes(r.raw(), d, values.data(), Device::CPU,
             values.size() * sizeof(int32_t));
  return r;
}

Tensor Tensor::from(std::initializer_list<float> values, Device d, DType t) {
  return from(std::span<const float>(values.begin(), values.size()),
              Shape({static_cast<int>(values.size())}), d, t);
}

Tensor Tensor::from(std::initializer_list<std::initializer_list<float>> rows, Device d, DType t) {
  const int cols = rows.size() ? static_cast<int>(rows.begin()->size()) : 0;

  std::vector<float> flat;
  flat.reserve(rows.size() * size_t(cols));
  for (const auto& row : rows) {
    assert(row.size() == rows.begin()->size() && "ragged nested initializer list");
    flat.insert(flat.end(), row.begin(), row.end());
  }

  return from(flat, Shape({static_cast<int>(rows.size()), cols}), d, t);
}

Tensor Tensor::from_i32(std::initializer_list<int32_t> values, Device d, DType t) {
  return from_i32(std::span<const int32_t>(values.begin(), values.size()),
                  Shape({static_cast<int>(values.size())}), d, t);
}

Tensor Tensor::view_like(const Shape& s, const Strides& strides, int64_t offset) const {
  Tensor v;
  v.storage_ = storage_;
  v.shape_ = s;
  v.strides_ = strides;
  v.offset_ = offset;
  v.dtype_ = dtype_;
  return v;
}

Tensor Tensor::contiguous() const {
  if (is_contiguous()) return *this;
  Tensor out(shape_, device(), dtype_);
  ops::copy_strided(out, *this);
  return out;
}

bool Tensor::is_contiguous() const {
  int64_t expected_stride = 1;
  for (int i = shape_.rank() - 1; i >= 0; --i) {
    if (shape_.dim(i) == 1) continue; // stride doesn't matter for size-1 dims
    if (strides_.at(i) != expected_stride) return false;
    expected_stride *= shape_.dim(i);
  }
  return true;
}

Tensor Tensor::permute(std::span<const int> order) const {
  assert(int(order.size()) == shape_.rank());

  Shape new_shape = shape_;
  Strides new_strides(shape_.rank());
  bool seen[kMaxShapeRank] = {false};
  for (int i = 0; i < shape_.rank(); ++i) {
    const int src = order[i];
    assert(src >= 0 && src < shape_.rank() && !seen[src] && "invalid permutation");
    seen[src] = true;
    new_shape.set_dim(i, shape_.dim(src));
    new_strides.at(i) = strides_.at(src);
  }

  Tensor v = view_like(new_shape, new_strides, offset_);
  return v;
}

Tensor Tensor::transpose(int a, int b) const {
  int order[kMaxShapeRank];
  for (int i = 0; i < shape_.rank(); ++i) order[i] = i;
  std::swap(order[a], order[b]);
  return permute(std::span<const int>(order, shape_.rank()));
}

Tensor Tensor::reshape(Shape s) const {
  assert(s.numel() == numel() && "reshape must preserve number of elements");
  if (!is_contiguous()) return contiguous().reshape(s);
  return view_like(s, Strides::contiguous_for(s), offset_);
}

Tensor Tensor::slice(int axis, int64_t start, int64_t len) const {
  assert(axis >= 0 && axis < shape_.rank());
  assert(start >= 0 && len >= 0 && start + len <= shape_.dim(axis));

  Shape new_shape = shape_;
  new_shape.set_dim(axis, int(len));
  int64_t new_offset = offset_ + start * strides_.at(axis);

  return view_like(new_shape, strides_, new_offset);
}

float* Tensor::device_ptr() const {
  assert(dtype_ == DType::F32);
  return static_cast<float*>(storage_->data()) + offset_;
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
  assert(is_contiguous() && "raw() on a non-contiguous tensor");
  return static_cast<char*>(storage_->data()) + offset_ * dtype_size(dtype_);
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
  if (!is_contiguous()) return contiguous().to(d);

  Tensor out(shape_, d, dtype_);
  copy_bytes(out.raw(), d, raw(), device(),
             size_t(numel()) * dtype_size(dtype_));
  
  return out;
}

Tensor Tensor::clone() const {
  if (!is_contiguous()) return contiguous();

  Tensor t(shape_, device(), dtype_);
  copy_bytes(t.raw(), device(), raw(), device(),
             size_t(numel()) * dtype_size(dtype_));
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
    memset_bytes(meta_->grad.raw(), meta_->grad.device(), 0,
                 meta_->grad.numel() * dtype_size(dtype_));
}

TensorView view_of(const Tensor& t) {
  TensorView v;
  v.rank = t.shape().rank();
  for (int i = 0; i < v.rank; ++i) {
    v.shape[i] = t.shape().dim(i);
    v.stride[i] = t.stride(i);
  }
  return v;
}

}