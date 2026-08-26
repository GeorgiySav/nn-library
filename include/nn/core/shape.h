#pragma once

#include <initializer_list>
#include <cassert>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>

#include <nn/core/tensorview.h>
namespace nn {

// Resolve a possibly-negative axis index against a rank, the numpy way: -1 is
// the last axis. `op` names the caller so the message says which operation
// rejected the axis.
constexpr int normalise_dim(int dim, int rank, std::string_view op) {
  const int d = (dim < 0) ? dim + rank : dim;
  if (d < 0 || d >= rank) {
    throw std::invalid_argument(std::string(op) + ": axis " + std::to_string(dim) +
                                " is out of range for rank " + std::to_string(rank));
  }
  return d;
}

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

  // Trusted index in, extent out. Asserts rather than throws -- see the note
  // on normalise_dim above.
  constexpr int dim(int i) const {
    assert(i >= 0 && i < rank_);
    return dims_[i];
  }

  // User-supplied axis in, validated index out. Saves every caller from
  // passing a rank it just read off this same shape.
  constexpr int resolve_dim(int d, std::string_view op) const {
    return normalise_dim(d, rank_, op);
  }

  // The public counterpart of dim(): accepts a negative axis the way every op
  // in the library does, and throws rather than asserting on an out-of-range
  // one. Reach for this from outside the library; dim() is for code that has
  // already established its index is in range.
  constexpr int extent(int axis) const {
    return dims_[normalise_dim(axis, rank_, "extent")];
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
