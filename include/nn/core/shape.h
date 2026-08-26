#pragma once

#include <initializer_list>
#include <cassert>
#include <span>
#include <stdexcept>
#include <string>

#include <nn/core/tensorview.h>
namespace nn {

class Shape {
public:
  constexpr Shape() = default;
  constexpr Shape(std::initializer_list<int> dims) {
    assert(dims.size() <= kMaxShapeRank);
    rank_ = dims.size();
    int i = 0;
    for (const auto& d : dims) {
      dims_[i++] = d;
    }
  }
  constexpr Shape(std::span<const int> dims) {
    assert(dims.size() <= kMaxShapeRank);
    rank_ = dims.size();
    for (int i = 0; i < rank_; ++i) {
      dims_[i] = dims[i];
    }
  }

  constexpr int rank() const { return rank_; }

  constexpr int dim(int i) const {
    assert(i >= 0 && i < rank_);
    return dims_[i];
  }

  constexpr void set_dim(int i, int d) {
    assert(i >= 0 && i < rank_);
    dims_[i] = d;
  }

  constexpr int64_t numel() const {
    int64_t n = 1;
    for (int i = 0; i < rank_; ++i) {
      n *= dims_[i];
    }
    return n;
  }

  constexpr bool operator==(const Shape& other) const = default;

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
  int rank_ = 0;
  int dims_[kMaxShapeRank] = {0};
};

// numpy broadcasting: align from the right, each axis must match or be 1.
// Missing leading axes on the shorter shape count as 1.
constexpr Shape broadcast_shapes(const Shape& a, const Shape& b) {
  const int r = (a.rank() > b.rank()) ? a.rank() : b.rank();
  int dims[kMaxShapeRank] = {0};
  for (int i = 0; i < r; ++i) {
    const int ia = a.rank() - r + i, ib = b.rank() - r + i;
    const int da = (ia >= 0) ? a.dim(ia) : 1;
    const int db = (ib >= 0) ? b.dim(ib) : 1;
    if (da == db)      dims[i] = da;
    else if (da == 1)  dims[i] = db;
    else if (db == 1)  dims[i] = da;
    else throw std::invalid_argument(
        "broadcast_shapes: " + a.str() + " and " + b.str() + " are not compatible");
  }
  return Shape(std::span<const int>(dims, r));
}

}
