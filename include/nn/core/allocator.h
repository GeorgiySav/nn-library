#pragma once

#include "device.h"

namespace nn {

struct Allocator {
  virtual ~Allocator() = default;
  virtual void* alloc(size_t bytes) = 0;
  virtual void free(void* ptr) = 0;
};

// throws for CUDA in a CPU build
Allocator& allocator_for(Device d);

}