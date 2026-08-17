#pragma once

#include <string>
#include <vector>

#include <nn/core/device.h>

#include "test_harness.h"

namespace nn::test {

inline const std::vector<Device>& devices() {
  static const std::vector<Device> d = [] {
    std::vector<Device> v{Device::CPU};
    #ifdef NN_WITH_CUDA
    if (cuda_device_count() > 0) v.push_back(Device::CUDA);
    #endif
    return v;
  }();
  return d;
}

// Suffixes the reported test name with the device, so an assertion failing
// inside a per-device loop says which device produced it.
class DeviceLabel {
public:
  explicit DeviceLabel(Device d) : prev_(current()) {
    label() = std::string(prev_) + " [" + device_name(d) + "]";
    current() = label().c_str();
  }
  ~DeviceLabel() { current() = prev_; }

  DeviceLabel(const DeviceLabel&) = delete;
  DeviceLabel& operator=(const DeviceLabel&) = delete;

private:
  static std::string& label() { static std::string s; return s; }
  const char* prev_;
};

}  // namespace nn::test

#define NN_TEST_FOR_EACH_DEVICE(dev)                      \
  for (nn::Device dev : nn::test::devices())              \
    if (nn::test::DeviceLabel nn_device_label_{dev}; true)
