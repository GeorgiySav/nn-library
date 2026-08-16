#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace nn {

class Arena {
public:
  explicit Arena(size_t block_bytes = 64 * 1024) : block_bytes_(block_bytes) {}

  Arena(const Arena&) = delete;
  Arena& operator=(const Arena&) = delete;

  Arena(Arena&&) = default;
  Arena& operator=(Arena&&) = default;

  void* alloc(size_t bytes, size_t align) {
    for (; block_ < blocks_.size(); ++block_, offset_ = 0) {
      Block& b = blocks_[block_];
      const auto base = reinterpret_cast<uintptr_t>(b.data.get());
      const uintptr_t p = align_up(base + offset_, align);

      if (p + bytes <= base + b.size) {
        offset_ = (p + bytes) - base;
        return reinterpret_cast<void*>(p);
      }
    }

    const size_t want = bytes + align;
    add_block((want > block_bytes_) ? want : block_bytes_);
    return alloc(bytes, align);
  }

  void reset() { block_ = 0; offset_ = 0; }

  void release() { blocks_.clear(); reset(); }

  size_t bytes_reserved() const {
    size_t n = 0;
    for (const Block& b : blocks_) n += b.size;
    return n;
  }

private:
  struct Block {
    std::unique_ptr<std::byte[]> data;
    size_t size;
  };

  static uintptr_t align_up(uintptr_t n, size_t a) {
    return (n + a - 1) & ~(static_cast<uintptr_t>(a) - 1);
  }

  void add_block(size_t size) {
    blocks_.push_back({std::make_unique<std::byte[]>(size), size});
    block_ = blocks_.size() - 1;
    offset_ = 0;
  }

  std::vector<Block> blocks_;
  size_t block_  = 0;
  size_t offset_ = 0;
  size_t block_bytes_;
};

}