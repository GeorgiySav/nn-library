#pragma once

#include <cassert>

#include <nn/core/shape.h>
#include <nn/core/tensorview.h>

namespace nn {

class Strides {
public:
  Strides() = default;
  explicit Strides(int rank) : rank_(rank) {}

  static Strides contiguous_for(const Shape& s) {
    Strides st(s.rank());
    int64_t acc = 1;
    for (int i = s.rank() - 1; i >= 0; --i) {
      st.v_[i] = acc;
      acc *= s.dim(i);
    }
    return st;
  }

  int rank() const { return rank_; }
  int64_t at(int i) const { 
    assert(i >= 0 && i < rank_);
    return v_[i]; 
  }
  int64_t& at(int i) { 
    assert(i >= 0 && i < rank_);
    return v_[i]; 
  }

private:
  int64_t v_[kMaxShapeRank] = {0};
  int rank_ = 0;
};

}