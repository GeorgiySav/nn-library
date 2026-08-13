#include "test_harness.h"

#include <nn/version.h>
#include <string>

int main(int argc, char** argv) {
  return nn::test::run_all(argc, argv);
}

NN_TEST(smoke) {
  NN_CHECK(1 + 1 == 2);
}

NN_TEST(version) {
  NN_CHECK(std::string(nn::version()) == "0.1.0");
}