#pragma once

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include <nn/core/tensor.h>

namespace nn::data {

struct Field {
  Shape shape;
  DType dtype = DType::F32;
};

inline Shape batched_shape(const Shape& s, int n) {
  if (s.rank() + 1 > kMaxShapeRank) {
    throw std::invalid_argument("batched_shape: rank exceeds kMaxShapeRank");
  }
  int dims[kMaxShapeRank];
  dims[0] = n;
  for (int i = 0; i < s.rank(); ++i) dims[i + 1] = s.dim(i);
  return Shape(std::span<const int>(dims, s.rank() + 1));
}

inline Shape sample_shape_of(const Tensor& t) {
  if (t.rank() < 1) {
    throw std::invalid_argument("sample_shape_of: tensor rank must be >= 1");
  }
  int dims[kMaxShapeRank];
  for (int i = 1; i < t.rank(); ++i) dims[i - 1] = t.extent(i);
  return Shape(std::span<const int>(dims, t.rank() - 1));
}

// a writable view of one row of a batch tensor, for Dataset::get to fill in
template <class T>
std::span<T> row_span(const Tensor& batch, int row) {
  static_assert(std::is_same_v<T, float> || std::is_same_v<T, int32_t>,
                "row_span: T must be float or int32_t");
  assert(row >= 0 && row < batch.extent(0));

  const int64_t per = batch.numel() / batch.extent(0);
  if constexpr (std::is_same_v<T, float>) {
    assert(batch.dtype() == DType::F32);
    return std::span<T>(batch.host_data() + row * per, size_t(per));
  } else {
    assert(batch.dtype() == DType::I32);
    return std::span<T>(batch.host_data_i32() + row * per, size_t(per));
  }
}

template <int N = 2>
class Dataset {
public:
  static constexpr int kFields = N;
  using Batch = std::array<Tensor, kFields>;

  virtual ~Dataset() = default;

  virtual int size() const = 0;
  virtual std::array<Field, kFields> fields() const = 0;

  virtual void get(int index, Batch& out, int row) const = 0;

  virtual void gather(std::span<const int> rows, Batch& out) const {
    for (int i = 0; i < int(rows.size()); ++i) get(rows[i], out, i);
  }
};

class TensorDataset : public Dataset<2> {
public:
  TensorDataset(Tensor a, Tensor b) : a_(std::move(a)), b_(std::move(b)) {
    if (a_.rank() < 1 || b_.rank() < 1) {
      throw std::invalid_argument("TensorDataset: tensors must have rank >= 1");
    }
    if (a_.extent(0) != b_.extent(0)) {
      throw std::invalid_argument("TensorDataset: tensors must have the same first dimension");
    }

    if (!a_.is_contiguous() || !b_.is_contiguous()) {
      throw std::invalid_argument("TensorDataset: tensors must be contiguous");
    }
  }

  int size() const override { return a_.extent(0); }
  std::array<Field, kFields> fields() const override {
    return {Field{sample_shape_of(a_), a_.dtype()},
            Field{sample_shape_of(b_), b_.dtype()}};
  }

  void get(int index, Batch& out, int row) const override {
    copy_row(out[0], row, a_, index);
    copy_row(out[1], row, b_, index);
  }

private:
  void copy_row(      Tensor& dst, int dst_row,
                const Tensor& src, int src_row) const {
    const size_t n = size_t(
      src.numel() / src.extent(0)
    ) * dtype_size(src.dtype());

    std::memcpy(
      static_cast<std::byte*>(dst.raw()) + size_t(dst_row) * n,
      static_cast<const std::byte*>(src.raw()) + size_t(src_row) * n,
      n
    );
  }

  Tensor a_, b_;
};

} // namespace nn::data