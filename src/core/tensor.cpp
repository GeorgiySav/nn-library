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
    for (int64_t i{0}; i < h.numel(); ++i) data[i] = value;
  });
}

Tensor Tensor::randn(Shape s, Pcg32& rng, float stddev, Device d, DType t) {
  return host_init(s, d, t, [&](Tensor& h) {
    float* data = h.host_data();
    for (int64_t i{0}; i < h.numel(); ++i) data[i] = rng.next_normal() * stddev;
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

Tensor Tensor::arange(int64_t n, Device d, DType t) {
  if (n < 0) {
    throw std::invalid_argument("arange: n must not be negative, got " + std::to_string(n));
  }
  if (n > INT32_MAX) {
    throw std::invalid_argument("arange: " + std::to_string(n) + " is too large for one axis");
  }

  Tensor h(Shape({int(n)}), Device::CPU, t);
  switch (t) {
    case DType::I32: {
      int32_t* data = h.host_data_i32();
      for (int64_t i = 0; i < n; ++i) data[i] = int32_t(i);
      break;
    }
    case DType::F32: {
      float* data = h.host_data();
      for (int64_t i = 0; i < n; ++i) data[i] = float(i);
      break;
    }
  }
  return h.to(d);
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

Tensor Tensor::pack() const {
  if (is_contiguous()) return *this;
  Tensor out(shape_, device(), dtype_);
  ops::pack(out, *this);
  return out;
}

bool Tensor::is_contiguous() const {
  int64_t expected_stride = 1;
  for (int i = shape_.rank() - 1; i >= 0; --i) {
    if (shape_.dim(i) == 1) continue; // a size-1 axis can carry any stride
    if (strides_.at(i) != expected_stride) return false;
    expected_stride *= shape_.dim(i);
  }
  return true;
}

Tensor Tensor::expand_view(const Shape& to) const {
  if (to.rank() < shape_.rank()) {
    throw std::invalid_argument("expand: " + shape_.str() + " -> " + to.str() +
                                " would drop axes");
  }

  const int lead = to.rank() - shape_.rank();   // axes `to` prepends
  Strides st(to.rank());
  for (int i = 0; i < to.rank(); ++i) {
    const int a = i - lead;
    if (a < 0) { st.at(i) = 0; continue; }             // brand new axis
    if (shape_.dim(a) == to.dim(i)) { st.at(i) = strides_.at(a); continue; }
    if (shape_.dim(a) != 1) {
      throw std::invalid_argument("expand: " + shape_.str() + " -> " + to.str() +
                                  ": axis is neither equal nor 1");
    }
    st.at(i) = 0;                                      // stretched
  }
  return view_like(to, st, offset_);
}

Tensor Tensor::permute_view(std::span<const int> order) const {
  const int r = shape_.rank();
  if (int(order.size()) != r) {
    throw std::invalid_argument("permute: " + std::to_string(order.size()) +
                                " axes given for rank " + std::to_string(r));
  }

  Shape new_shape = shape_;
  Strides new_strides(r);
  bool seen[kMaxShapeRank] = {false};
  for (int i = 0; i < r; ++i) {
    const int src = shape_.resolve_dim(order[i], "permute");
    if (seen[src]) {
      throw std::invalid_argument("permute: axis " + std::to_string(src) +
                                  " appears twice in the order");
    }
    seen[src] = true;
    new_shape.set_dim(i, shape_.dim(src));
    new_strides.at(i) = strides_.at(src);
  }

  return view_like(new_shape, new_strides, offset_);
}

Tensor Tensor::permute_view(std::initializer_list<int> order) const {
  return permute_view(std::span<const int>(order.begin(), order.size()));
}

Tensor Tensor::transpose_view(int a, int b) const {
  const int r = shape_.rank();
  const int ia = shape_.resolve_dim(a, "transpose");
  const int ib = shape_.resolve_dim(b, "transpose");

  int order[kMaxShapeRank];
  for (int i = 0; i < r; ++i) order[i] = i;
  std::swap(order[ia], order[ib]);
  return permute_view(std::span<const int>(order, r));
}

Tensor Tensor::reshape_view(Shape s) const {
  assert(s.numel() == numel() && "reshape must preserve number of elements");
  if (!is_contiguous()) return pack().reshape_view(s);
  return view_like(s, Strides::contiguous_for(s), offset_);
}

Tensor Tensor::slice_view(int axis, int64_t start, int64_t len) const {
  const int a = shape_.resolve_dim(axis, "slice");
  if (start < 0 || len < 0 || start + len > shape_.dim(a)) {
    throw std::invalid_argument("slice: [" + std::to_string(start) + ", " +
                                std::to_string(start + len) + ") is out of range for axis " +
                                std::to_string(a) + " of " + shape_.str());
  }

  Shape new_shape = shape_;
  new_shape.set_dim(a, int(len));
  int64_t new_offset = offset_ + start * strides_.at(a);

  return view_like(new_shape, strides_, new_offset);
}

float* Tensor::device_ptr() const {
  assert(dtype_ == DType::F32);
  return static_cast<float*>(storage_->data()) + offset_;
}

int32_t* Tensor::device_ptr_i32() const {
  assert(dtype_ == DType::I32);
  return static_cast<int32_t*>(storage_->data()) + offset_;
}

bf16* Tensor::device_ptr_bf16() const {
  assert(dtype_ == DType::BF16);
  return static_cast<bf16*>(storage_->data()) + offset_;
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
  assert(is_contiguous() && "raw() on a non-contiguous tensor");
  return static_cast<const char*>(storage_->data()) + offset_ * dtype_size(dtype_);
}

float Tensor::item() const {
  assert(numel() == 1);
  if (device() != Device::CPU) return to(Device::CPU).item();
  return host_data()[0];
}

Tensor Tensor::to(Device d) const {
  if (!storage_) return Tensor{};
  if (device() == d) return *this;
  if (!is_contiguous()) return pack().to(d);

  Tensor out(shape_, d, dtype_);
  copy_bytes(out.raw(), d, raw(), device(),
             size_t(numel()) * dtype_size(dtype_));
  
  return out;
}

Tensor Tensor::clone() const {
  if (!is_contiguous()) return pack();

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

void Tensor::zero_grad(bool set_to_none) {
  if (!meta_ || !meta_->grad.defined()) return;
  if (set_to_none) {
    meta_->grad = Tensor{};
  } else {
    memset_bytes(meta_->grad.raw(), meta_->grad.device(), 0,
                 meta_->grad.numel() * dtype_size(dtype_));
  }
}

// Builds the kernel-facing view: drops size-1 axes and merges an axis into
// its neighbor whenever they are contiguous with each other, so kernels see
// the smallest rank that still describes the same memory layout.
TensorView view_of(const Tensor& t) {
  TensorView v;
  for (int i = 0; i < t.shape().rank(); ++i) {
    const int64_t d = t.shape().dim(i);
    const int64_t s = t.stride(i);
    if (d == 1) continue;
    // The previous axis steps exactly one full run of this one.
    if (v.rank > 0 && v.stride[v.rank - 1] == s * d) {
      v.shape[v.rank - 1] *= d;
      v.stride[v.rank - 1] = s;
      continue;
    }
    v.shape[v.rank] = d;
    v.stride[v.rank] = s;
    ++v.rank;
  }
  return v;
}

Tensor tril_mask(int n, Device d) {
  if (n < 0) {
    throw std::invalid_argument("tril_mask: n must not be negative, got " + std::to_string(n));
  }

  Tensor h(Shape({n, n}), Device::CPU, DType::F32);
  float* data = h.host_data();
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) data[size_t(i) * n + j] = (j <= i) ? 1.0f : 0.0f;
  }
  return h.to(d);
}

}
