#include "test_harness.h"

#include <nn/core/small_vec.h>
#include <nn/core/shape.h>
#include <nn/core/rng.h>

NN_TEST(test_small_vec) {
  nn::SmallVec<int, 4> v;
  NN_CHECK(v.empty());
  NN_CHECK(v.size() == 0);

  v.push_back(1);
  NN_CHECK(!v.empty());
  NN_CHECK(v.size() == 1);
  NN_CHECK(v[0] == 1);

  v.push_back(2);
  v.push_back(3);
  v.push_back(4);
  NN_CHECK(v.size() == 4);
  NN_CHECK(v[0] == 1);
  NN_CHECK(v[1] == 2);
  NN_CHECK(v[2] == 3);
  NN_CHECK(v[3] == 4);

  v.pop_back();
  NN_CHECK(v.size() == 3);

  NN_CHECK(std::equal(v.begin(), v.end(), std::vector<int>{1, 2, 3}.begin()));
}

NN_TEST(test_shape) {
  nn::Shape s1{2, 3, 4};
  NN_CHECK(s1.rank() == 3);
  NN_CHECK(s1.dim(0) == 2);
  NN_CHECK(s1.dim(1) == 3);
  NN_CHECK(s1.dim(2) == 4);
  NN_CHECK(s1.numel() == 24);

  nn::Shape s2{2, 3, 4};
  nn::Shape s3{2, 3, 5};
  NN_CHECK(s1 == s2);
  NN_CHECK(s1 != s3);

  std::string str = s1.str();
  NN_CHECK(str == "[2, 3, 4]");
}

NN_TEST(test_pcg32) {
  nn::Pcg32 rng1(12345);
  nn::Pcg32 rng2(12345);

  for (int i = 0; i < 10; ++i) {
    uint32_t val1 = rng1.next_uint32();
    uint32_t val2 = rng2.next_uint32();
    NN_CHECK(val1 == val2);
  }

  nn::Pcg32 rng3(67890);
  for (int i = 0; i < 10; ++i) {
    uint32_t val1 = rng1.next_uint32();
    uint32_t val3 = rng3.next_uint32();
    NN_CHECK(val1 != val3);
  }

  for (int i = 0; i < 10; ++i) {
    float uniform_val = rng1.next_uniform();
    NN_CHECK(uniform_val >= 0.0f && uniform_val < 1.0f);
  }

  // mean and variance within 0.02 of 0 and 1 respectively for 100000 samples
  const int num_samples = 100000;
  float sum = 0.0f;
  float sum_sq = 0.0f;
  for (int i = 0; i < num_samples; ++i) {
    float normal_val = rng1.next_normal();
    sum += normal_val;
    sum_sq += normal_val * normal_val;
  }
  float mean = sum / num_samples;
  float variance = sum_sq / num_samples - mean * mean;
  NN_CHECK(std::abs(mean) < 0.02f);
  NN_CHECK(std::abs(variance - 1.0f) < 0.02f);
}