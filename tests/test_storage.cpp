#include "test_harness.h"

#include <nn/core/allocator.h>
#include <nn/core/storage.h>

NN_TEST(test_storage) {
  nn::Storage s(1024, nn::Device::CPU);
  NN_CHECK(s.bytes() == 1024);
  NN_CHECK(s.device() == nn::Device::CPU);
  NN_CHECK(s.data() != nullptr);
  NN_CHECK(reinterpret_cast<uintptr_t>(s.data()) % 64 == 0);

  // writing float values across the buffer and reading them back returns that same values
  float* data = reinterpret_cast<float*>(s.data());
  for (int i = 0; i < 256; ++i) {
    data[i] = static_cast<float>(i) * 0.5f;
  }
  for (int i = 0; i < 256; ++i) {
    NN_CHECK(data[i] == static_cast<float>(i) * 0.5f);
  }
 
  nn::Storage z(0, nn::Device::CPU);
  NN_CHECK(z.bytes() == 0);
  NN_CHECK(z.data() == nullptr);
}