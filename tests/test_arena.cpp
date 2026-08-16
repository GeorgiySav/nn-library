#include "test_harness.h"

#include <nn/core/arena.h>

NN_TEST(arena_aligns_each_request) {
  nn::Arena a(256);
 
  void* p1 = a.alloc(1, 1);
  void* p2 = a.alloc(8, 8);
  void* p3 = a.alloc(4, 16);

  NN_CHECK(reinterpret_cast<uintptr_t>(p2) % 8 == 0);
  NN_CHECK(reinterpret_cast<uintptr_t>(p3) % 16 == 0);
  NN_CHECK(p1 != p2 && p2 != p3);
}

NN_TEST(arena_grows_past_one_block) {
  nn::Arena a(64);

  for (int i{0}; i < 100; i++) NN_CHECK(a.alloc(32, 8) != nullptr);

  NN_CHECK(a.bytes_reserved() >= 100 * 32);
}

NN_TEST(arena_serves_a_request_larger_than_a_block) {
  nn::Arena a(64);
  NN_CHECK(a.alloc(4096, 16) != nullptr);
}

NN_TEST(arena_reset_reuses_its_blocks) {
  nn::Arena a(1024);
  void* first = a.alloc(16, 8);
  for (int i{0}; i < 200; ++i) a.alloc(16, 8);
  const size_t peak = a.bytes_reserved();

  a.reset();
  NN_CHECK(a.alloc(16, 8) == first);
  for (int i{0}; i < 200; ++i) a.alloc(16, 8);
  NN_CHECK(a.bytes_reserved() == peak);
}