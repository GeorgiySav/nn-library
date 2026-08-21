#pragma once

#include <initializer_list>
#include <cassert>
#include <string>

namespace nn {

inline constexpr int kMaxShapeRank = 8;
class Shape {
public:
  Shape() = default;
  Shape(std::initializer_list<int> dims) {
    assert(dims.size() <= kMaxShapeRank);
    rank_ = dims.size();
    int i = 0;
    for (const auto& d : dims) {
      dims_[i++] = d;
    }
  }

  int rank() const { return rank_; }
  int dim(int i) const {
    assert(i < rank_);
    return dims_[i];
  }

  int64_t numel() const {
    int64_t n = 1;
    for (int i = 0; i < rank_; ++i) {
      n *= dims_[i];
    }
    return n;
  }

  bool operator==(const Shape& other) const {
    if (rank_ != other.rank_) {
      return false;
    }
    for (int i = 0; i < rank_; ++i) {
      if (dims_[i] != other.dims_[i]) {
        return false;
      }
    }
    return true;
  }

  bool operator!=(const Shape& other) const {
    return !(*this == other);
  }

  std::string str() const {
    std::string s = "[";
    for (int i = 0; i < rank_; ++i) {
      if (i > 0) {
        s += ", ";
      }
      s += std::to_string(dims_[i]);
    }
    s += "]";
    return s;
  }

private:
  int dims_[kMaxShapeRank] = {0};
  int rank_ = 0;
};

}