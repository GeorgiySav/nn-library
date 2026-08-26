#pragma once

#include <numeric>
#include <stdexcept>
#include <vector>

#include <nn/core/tensor.h>
#include <nn/data/dataset.h>

namespace nn::data {

struct Batch {
  Tensor x;
  Tensor y;
};

template <class Target = int32_t>
class DataLoader {
public:
  DataLoader(std::shared_ptr<const Dataset<Target>> ds,
             int batch_size,
             Pcg32& rng,
             bool shuffle = true,
             bool drop_last = true,
             Device device = Device::CPU)
             : ds_(std::move(ds)), rng_(rng), batch_(batch_size),
               shuffle_(shuffle), drop_last_(drop_last), device_(device) {
    if (!ds_) throw std::invalid_argument("DataLoader: null dataset");
    if (batch_size <= 0 || batch_size > ds_->size()) {
      throw std::invalid_argument("DataLoader: bad batch sizes");
    }
    order_.resize(size_t(ds_->size()));
    std::iota(order_.begin(), order_.end(), 0);

    xb_ = Tensor(Shape{batch_size, ds_->features()});
    yb_ = make_target(batch_size);
    reset();
  }

  void reset() {
    cursor_ = 0;
    if (!shuffle_) return;
    for (size_t i{order_.size()}; i > 1; --i) {
      std::swap(order_[i-1], order_[bounded(uint32_t(i))]);
    }
  }

  bool has_next() const {
    return drop_last_ ? cursor_ + batch_ <= ds_->size() : cursor_ < ds_->size();
  }

  int batches_per_epoch() const {
    return drop_last_ ? ds_->size() / batch_
                      : (ds_->size()+ batch_ - 1) / batch_;
  }

  Batch next() {
    if (!has_next()) throw std::logic_error("DataLoader: epoch exhausted");

    const int n = std::min(batch_, ds_->size() - cursor_);
    const int d = ds_->features();

    Batch b{n == batch_ ? xb_ : Tensor(Shape{n, d}),
            n == batch_ ? yb_ : make_target(n)};

    ds_->gather(std::span<const int>(order_.data() + cursor_, size_t(n)), b.x, b.y);

    cursor_ += n;

    b.x = b.x.to(device_);
    b.y = b.y.to(device_);

    return b;
  }
  
private:
  Tensor make_target(int n) const {
    const int k = ds_->target_width();
    return k == 1 ? Tensor(Shape{n},    Device::CPU, dtype_of_v<Target>)
                  : Tensor(Shape{n, k}, Device::CPU, dtype_of_v<Target>);
  }

  uint32_t bounded(uint32_t n) {
    const uint32_t threshold = (0u - n) % n;
    uint32_t v;
    do { v = rng_.next_uint32(); } while (v < threshold);
    return v % n;
  }

  std::shared_ptr<const Dataset<Target>> ds_;
  Pcg32& rng_;
  int batch_;
  bool shuffle_, drop_last_;
  Device device_;
  int cursor_ = 0;
  std::vector<int> order_;
  Tensor xb_, yb_;
};
  
} // namespace nn::data
