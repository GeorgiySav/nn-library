#include <nn/core/storage.h>

#include <nn/core/allocator.h>

namespace nn {

Storage::Storage(size_t bytes, Device d) : bytes_(bytes), device_(d) {
  ptr_ = allocator_for(d).alloc(bytes);
}

// Frees through the same device's allocator that made the allocation, so a
// Storage must never move to another device after construction.
Storage::~Storage() {
  if (ptr_) {
    allocator_for(device_).free(ptr_);
  }
}

}