#include "test_harness.h"

#include <nn/version.h>
#include <string>
#include <typeinfo>

int main(int argc, char** argv) {
  try { return nn::test::run_all(argc, argv); }
  catch (const std::exception& e) {
    std::printf("\n!! Uncaught %s: %s\n", typeid(e).name(), e.what());
    return 2;
  }
}

NN_TEST(smoke) {
  NN_CHECK(1 + 1 == 2);
}

NN_TEST(version) {
  NN_CHECK(std::string(nn::version()) == "0.1.0");
}