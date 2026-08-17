#include <nn/core/allocator.h>

#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "cuda_common.h"

namespace nn {

namespace {

void configure_pool_once() {
  [[maybe_unused]] static const bool once = [] {
    int device = 0;
    NN_CUDA_CHECK(cudaGetDevice(&device));

    cudaMemPool_t pool{};
    NN_CUDA_CHECK(cudaDeviceGetDefaultMemPool(&pool, device));

    uint64_t threshold = UINT64_MAX;
    NN_CUDA_CHECK(cudaMemPoolSetAttribute(
      pool, cudaMemPoolAttrReleaseThreshold, &threshold
    ));
    return true;
  }();
}

}

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

struct CudaAllocator : public Allocator {
  void* alloc(size_t bytes) override {
    configure_pool_once();
    if (bytes == 0) {
      return nullptr;
    }
    void* ptr;
    NN_CUDA_CHECK(
      cudaMallocAsync(
        &ptr,
        bytes,
        cudaStream_t{nullptr}
      )
    );
    return ptr;
  }

  void free(void* ptr) override {
    if (!ptr) return;

    cudaFreeAsync(
      ptr,
      cudaStream_t{nullptr}
    );
  }
};

void copy_bytes(void* dst, Device dst_dev,
                const void* src, Device src_dev, size_t bytes) {
  if (bytes == 0) return;

  if (dst_dev == Device::CPU && src_dev == Device::CPU) {
    std::memcpy(dst, src, bytes);
    return;
  }

  cudaMemcpyKind kind;
  if      (src_dev == Device::CPU  && dst_dev == Device::CUDA)
    kind = cudaMemcpyKind::cudaMemcpyHostToDevice;
  else if (src_dev == Device::CUDA && dst_dev == Device::CUDA)
    kind = cudaMemcpyKind::cudaMemcpyDeviceToDevice;
  else
    kind = cudaMemcpyKind::cudaMemcpyDeviceToHost;

  NN_CUDA_CHECK(
    cudaMemcpyAsync(dst, src, bytes, kind, cudaStream_t{nullptr})
  );

  if (dst_dev == Device::CPU) current_stream(Device::CUDA).synchronize();
}

void memset_bytes(void* dst, Device dst_dev, int value, size_t bytes) {
  if (bytes == 0) return;

  if (dst_dev == Device::CPU) {
    std::memset(dst, value, bytes);
    return;
  }

  NN_CUDA_CHECK(
    cudaMemsetAsync(dst, value, bytes, cudaStream_t{nullptr})
  );
}

Allocator& allocator_for(Device d) {
  switch (d) {
    case Device::CPU: {
      static CpuAllocator cpu_allocator;
      return cpu_allocator;
    }
    case Device::CUDA: {
      static CudaAllocator gpu_allocator;
      return gpu_allocator;
    }
    default:
      throw std::runtime_error("Unknown device");
  }
}

}