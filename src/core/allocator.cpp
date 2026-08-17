#include <nn/core/allocator.h>

#include <cstdlib>
#include <cstring>
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

void copy_bytes(void* dst, Device dst_dev,
                const void* src, Device src_dev, size_t bytes) {
  if (bytes == 0) return;

  if (dst_dev == Device::CPU && src_dev == Device::CPU) {
    std::memcpy(dst, src, bytes);
    return;
  }

  // CUDA: issue the copy on the source device's current stream, then
  // synchronise on any transfer that lands in host memory -- the caller is
  // about to dereference dst and would otherwise race the copy.
  //   cudaMemcpyAsync(dst, src, bytes, kind, current_stream(gpu).handle);
  //   if (dst_dev == Device::CPU) current_stream(gpu).synchronize();
  throw std::runtime_error("copy_bytes: CUDA backend not built");
}

void memset_bytes(void* dst, Device dst_dev, int value, size_t bytes) {
  if (bytes == 0) return;

  if (dst_dev == Device::CPU) {
    std::memset(dst, value, bytes);
    return;
  }

  // CUDA: issue the memset on the device's current stream
  //   cudaMemsetAsync(dst, value, bytes, current_stream(gpu).handle);
  throw std::runtime_error("memset_bytes: CUDA backend not built");
}

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