#include "test_harness.h"

#include <nn/core/small_vec.h>

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