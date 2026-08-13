#include <nn/core/allocator.h>

#include <cstdlib>
#include <stdexcept>

namespace nn {

struct CpuAllocator : public Allocator {
  void* alloc(size_t bytes) override {
    if (bytes == 0) {
      return nullptr;
    }
    bytes = (bytes + 63) & ~size_t(63); // align to 64 bytes
    return new(std::align_val_t(64)) char[bytes];
  }

  void free(void* ptr) override {
    if (ptr) {
      ::operator delete(ptr, std::align_val_t(64));
    }
  }
};

Allocator& allocator_for(Device d) {
  switch (d) {
    case Device::CPU:
      static CpuAllocator cpu_allocator;
      return cpu_allocator;
    case Device::CUDA:
      throw std::runtime_error("CUDA allocator not implemented");
    default:
      throw std::runtime_error("Unknown device");
  }
}

}