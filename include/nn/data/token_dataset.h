#pragma once

#include <algorithm>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <nn/data/dataset.h>

namespace nn::data {

// A window of `window` token ids as x, the same window shifted by one as y.
class TokenDataset : public Dataset<2> {
public:
  TokenDataset(std::span<const int32_t> ids, int window, int stride)
    : tokens_(ids.data()),
      count_(int64_t(ids.size())),
      window_(window),
      stride_(stride) {
    if (window <= 0) throw std::invalid_argument("TokenDataset: window must be > 0");
    if (stride <= 0) throw std::invalid_argument("TokenDataset: stride must be > 0");
    // get() reads window_ + 1 ids per row (x, and y shifted by one), so
    // count_ == window_ is one short, not enough.
    if (count_ <= window_) {
      throw std::invalid_argument("TokenDataset: not enough tokens for window");
    }
  }

  int size() const override {
    return int((count_ - window_ - 1) / stride_) + 1;
  }

  std::array<Field, kFields> fields() const override {
    return {Field{Shape{window_}, DType::I32},
            Field{Shape{window_}, DType::I32}};
  }

  void get(int index, Batch& out, int row) const override {
    const int32_t* src = tokens_ + int64_t(index) * stride_;
    auto x = row_span<int32_t>(out[0], row);
    auto y = row_span<int32_t>(out[1], row);
    std::copy_n(src,     window_, x.data());
    std::copy_n(src + 1, window_, y.data());
  }

private:
  const int32_t* tokens_;
  int64_t count_;
  int window_, stride_;
};

// Reads a whole token file into memory. Simple, and the corpus must fit in
// RAM; for a corpus that does not, see MappedTokens.
std::vector<int32_t> load_tokens(const std::string& path);

class MappedTokens {
public:
  explicit MappedTokens(const std::string& path);
  ~MappedTokens();

  MappedTokens(MappedTokens&&) noexcept;
  MappedTokens& operator=(MappedTokens&&) noexcept;
  MappedTokens(const MappedTokens&) = delete;
  MappedTokens& operator=(const MappedTokens&) = delete;

  std::span<const int32_t> tokens() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace nn::data
