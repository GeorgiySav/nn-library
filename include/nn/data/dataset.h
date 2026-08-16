#pragma once

#include <cstdint>
#include <span>
#include <stdexcept>

#include <nn/core/tensor.h>

namespace nn::data {

template <class T> struct dtype_of;
template <> struct dtype_of<float> { static constexpr DType value = DType::F32; };
template <> struct dtype_of<int32_t> { static constexpr DType value = DType::I32; };
template <class T> inline constexpr DType dtype_of_v = dtype_of<T>::value;

template <class Target = int32_t>
class Dataset {
public:
  using target_type = Target;

  virtual ~Dataset() = default;

  virtual int size() const = 0;                // number of samples
  virtual int features() const = 0;            // floats per input sample
  virtual int target_width() const { return 1; } // target values per sample

  virtual void get(int index, std::span<float> x_out,
                   std::span<Target> y_out) const = 0;
  
  virtual void gather(std::span<const int> rows,
                      Tensor& x_out, Tensor& y_out) const {
    const size_t d = size_t(features());
    const size_t k = size_t(target_width());

    float* x = x_out.data();
    Target* y = target_ptr(y_out);

    for (size_t i{0}; i < rows.size(); ++i) {
      get(rows[i], std::span<float>(x + i * d, d),
                   std::span<Target>(y + i * k, k));
    }
  }

protected:
  static Target* target_ptr(Tensor& t) {
    if constexpr (std::is_same_v<Target, float>) return t.data();
    else                                         return t.data_i32();
  }
};

using ClassificationDataset = Dataset<int32_t>;
using RegressionDataset     = Dataset<float>;

template <class Target = int32_t>
class TensorDataset : public Dataset<Target> {
public:
  TensorDataset(Tensor x, Tensor y) : x_(std::move(x)), y_(std::move(y)) {
    if (x_.shape().rank() != 2) {
      throw std::invalid_argument("TensorDataset: x must be [N, D]");
    }
    if (y_.shape().rank() != 1 && y_.shape().rank() != 2) {
      throw std::invalid_argument("TensorDataset: y must be [N] or [N, K]");
    }
    if (x_.shape().dim(0) != y.shape().dim(0)) {
      throw std::invalid_argument("TensorDataset: sample count mismatch");
    }
    if (y_.dtype() != dtype_of_v<Target>) {
      throw std::invalid_argument("TensorDataset: target dtype mismatch");
    }
  }

  int size() const override { return x_.shape().dim(0); }
  int features () const override { return x_.shape().dim(1); }
  int target_width() const override {
    return y_.shape().rank() == 1 ? 1 : y_.shape().dim(1);
  }

  void get(int index, std::span<float> x_out,
           std::span<Target> y_out) const override {
    const size_t d = size_t(features());
    const size_t k = size_t(target_width());

    assert(x_out.size() == d && y_out.size() == k);

    std::memcpy(x_out.data(), x_.data() + size_t(index) * d, d * sizeof(float));
    std::memcpy(y_out.data(), src_targets() + size_t(index) * k, k * sizeof(Target));
  }

private:
  const Target* src_targets() const {
    if constexpr (std::is_same_v<Target, float>) return y_.data();
    else                                         return y_.data_i32();
  }

  Tensor x_, y_;
};
  
} // namespace nn::data