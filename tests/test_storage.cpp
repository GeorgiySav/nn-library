#include "test_harness.h"
#include "devices.h"

#include <nn/core/allocator.h>
#include <nn/core/storage.h>

NN_TEST(test_storage) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    nn::Storage s(1024, dev);
    NN_CHECK(s.bytes() == 1024);
    NN_CHECK(s.device() == dev);
    NN_CHECK(s.data() != nullptr);
    NN_CHECK(reinterpret_cast<uintptr_t>(s.data()) % 64 == 0);

    // writing float values across the buffer and reading them back returns
    // those same values -- routed through copy_bytes so this holds on any
    // device, since s.data() is not host-dereferenceable off the CPU
    std::vector<float> written(256);
    for (int i = 0; i < 256; ++i) {
      written[i] = static_cast<float>(i) * 0.5f;
    }
    nn::copy_bytes(s.data(), dev, written.data(), nn::Device::CPU,
                   written.size() * sizeof(float));

    std::vector<float> read_back(256, -1.0f);
    nn::copy_bytes(read_back.data(), nn::Device::CPU, s.data(), dev,
                   read_back.size() * sizeof(float));

    for (int i = 0; i < 256; ++i) {
      NN_CHECK(read_back[i] == static_cast<float>(i) * 0.5f);
    }

    nn::Storage z(0, dev);
    NN_CHECK(z.bytes() == 0);
    NN_CHECK(z.data() == nullptr);
  }
}
