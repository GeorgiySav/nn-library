#pragma once

#include <initializer_list>
#include <cassert>
#include <span>

namespace nn {

template<class T, int N> class SmallVec {
public:
  SmallVec() = default;
  SmallVec(std::initializer_list<T> init) {
    assert(init.size() <= N);
    for (const auto& v : init) {
      data_[size_++] = v;
    }
  }
  SmallVec(size_t n) {
    assert(n <= N);
    size_ = n;
  }

  std::span<T> span() { return {data_, size_t(size_)}; }

  void push_back(const T& v) {
    assert(size_ < N);
    data_[size_++] = v;
  }

  void pop_back() {
    assert(size_ > 0);
    --size_;
  }

  T& operator[](int i) { return data_[i]; }
  const T& operator[](int i) const { return data_[i]; }

  T* begin() { return data_; }
  T* end() { return data_ + size_; }

  int size() const { return size_; }
  bool empty() const { return size_ == 0; }

private:
  T data_[N];
  int size_ = 0;
};

}