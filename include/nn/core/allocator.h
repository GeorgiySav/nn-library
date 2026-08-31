#pragma once

#include <cstddef>

#include "device.h"

namespace nn {

struct Allocator {
  virtual ~Allocator() = default;
  virtual void* alloc(size_t bytes) = 0;
  virtual void free(void* ptr) = 0;
};

Allocator& allocator_for(Device d);

// Copies raw bytes between two devices (or within one). Blocks only when the
// destination is the host; a host-to-device or device-to-device copy is
// queued asynchronously, so src must stay valid until the caller
// synchronizes.
void copy_bytes(void* dst, Device dst_dev,
                const void* src, Device src_dev, size_t bytes);

void memset_bytes(void* dst, Device dst_dev, int value, size_t bytes);

}