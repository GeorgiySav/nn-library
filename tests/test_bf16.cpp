#include "test_harness.h"
#include "devices.h"

#include <cmath>
#include <vector>

#include <nn/autograd/functions.h>
#include <nn/autograd/tape.h>
#include <nn/core/allocator.h>
#include <nn/core/bf16.h>
#include <nn/core/dtype.h>
#include <nn/core/rng.h>
#include <nn/core/storage.h>
#include <nn/core/tensor.h>

NN_TEST(bf16_dtype_size_and_name) {
  NN_CHECK(nn::dtype_size(nn::DType::BF16) == 2);
  NN_CHECK(std::string(nn::dtype_name(nn::DType::BF16)) == "bfloat16");
}

NN_TEST(bf16_pod_is_two_bytes) {
  NN_CHECK(sizeof(nn::bf16) == 2);
}