#pragma once

#include "device.h"

namespace nn {

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