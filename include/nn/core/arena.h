#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace nn {

// Bump allocator over a growable list of blocks. Individual allocations are
// never freed; reset() rewinds to the start without releasing the blocks, so
// the same backing memory can be reused across repeated passes (e.g. one
// arena per forward pass). release() actually frees everything.
class Arena {
public:
  explicit Arena(size_t block_bytes = 64 * 1024) : block_bytes_(block_bytes) {}

  Arena(const Arena&) = delete;
  Arena& operator=(const Arena&) = delete;

  Arena(Arena&&) = default;
  Arena& operator=(Arena&&) = default;

  void* alloc(size_t bytes, size_t align) {
    // Scan forward from the current block (not always block 0, since a
    // previous alloc may have moved on) for one with room left.
    for (; block_ < blocks_.size(); ++block_, offset_ = 0) {
      Block& b = blocks_[block_];
      const auto base = reinterpret_cast<uintptr_t>(b.data.get());
      const uintptr_t p = align_up(base + offset_, align);

      if (p + bytes <= base + b.size) {
        offset_ = (p + bytes) - base;
        return reinterpret_cast<void*>(p);
      }
    }

    // Nothing left fits; grow by a new block sized for this request if it
    // wouldn't fit in a default-sized block.
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

  // a must be a power of two.
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