#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <numeric>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#include <nn/core/rng.h>
#include <nn/core/tensor.h>
#include <nn/data/dataset.h>

namespace nn::data {

template <int N = 2>
class DataLoader {
public:
  using Batch = std::array<Tensor, N>;

  DataLoader(std::shared_ptr<Dataset<N>> ds,
             int batch_size,
             Pcg32& rng,
             bool shuffle = true,
             bool drop_last = false,
             Device device = Device::CPU)
    : ds_(std::move(ds)),
      rng_(rng),
      batch_(batch_size),
      shuffle_(shuffle),
      drop_last_(drop_last),
      device_(device) {
    if (!ds_) throw std::invalid_argument("DataLoader: dataset is null");
    if (batch_size <= 0 || batch_size > ds_->size()) {
      throw std::invalid_argument("DataLoader: invalid batch size");
    }

    fields_ = ds_->fields();

    order_.resize(size_t(ds_->size()));
    std::iota(order_.begin(), order_.end(), 0);
    reset();
  }

  // Fisher-Yates shuffle of the row order, done in place
  void reset() {
    cursor_ = 0;
    if (!shuffle_) return;
    for (int i{int(order_.size())}; i > 1; --i) {
      std::swap(order_[i - 1], order_[bounded(uint32_t(i))]);
    }
  }

  bool has_next() const {
    return drop_last_ ?
            cursor_ + batch_ <= ds_->size() :
            cursor_ < ds_->size();
  }

  int batches_per_epoch() const {
    return drop_last_ ?
            ds_->size() / batch_ :
            (ds_->size() + batch_ - 1) / batch_;
  }

  Batch next() {
    if (!has_next()) throw std::logic_error("DataLoader: epoch exhausted");

    const int n = std::min(batch_, ds_->size() - cursor_);

    // one tensor per field, always allocated on the host since gather()
    // fills them via row_span, then moved to the target device below
    Batch batch;
    for (int f = 0; f < N; ++f) {
      batch[f] = Tensor(batched_shape(fields_[f].shape, n),
                        Device::CPU, fields_[f].dtype);
    }

    ds_->gather(std::span<const int>(order_.data() + cursor_, size_t(n)), batch);
    cursor_ += n;

    for (Tensor& t : batch) t = t.to(device_);
    return batch;
  }

private:
  // uniform random value in [0, n), rejecting the low draws that would
  // make some outputs more likely than others under plain modulo bias
  // (threshold is 2^32 mod n, computed via unsigned wraparound)
  uint32_t bounded(uint32_t n) {
    const uint32_t threshold = (0u - n) % n;
    uint32_t v;
    do { v = rng_.next_uint32(); } while (v < threshold);
    return v % n;
  }
  
  std::shared_ptr<Dataset<N>> ds_;
  Pcg32& rng_;
  int batch_;
  bool shuffle_, drop_last_;
  Device device_;
  int cursor_ = 0;
  std::vector<int> order_;
  std::array<Field, N> fields_;
};
  
} // namespace nn::data
