#pragma once

#include <cstddef>

#include "device.h"

namespace nn {

// Owns one raw device allocation for its entire lifetime; freed on
// destruction through the allocator for `device_`. Not copyable, and not
// movable to another device, since Tensor shares it by std::shared_ptr.
class Storage {
public:
  Storage(size_t bytes, Device d);
  ~Storage();
  Storage(const Storage&) = delete;
  Storage& operator=(const Storage&) = delete;

  void*       data()         { return ptr_; }
  const void* data()   const { return ptr_; }
  size_t      bytes()  const { return bytes_; }
  Device      device() const { return device_; }

private:
  void* ptr_ = nullptr;
  size_t bytes_ = 0;
  Device device_;
};

}