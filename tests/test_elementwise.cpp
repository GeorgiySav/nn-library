#include "test_harness.h"

#include <nn/kernels/kernel_api.h>

const nn::kernels::KernelTable& init() {
  nn::kernels::init_kernels();
  const nn::kernels::KernelTable& kernels = nn::kernels::kernels(nn::Device::CPU);
  NN_CHECK(kernels.gemm != nullptr);
  NN_CHECK(kernels.add_row_bias != nullptr);
  NN_CHECK(kernels.col_sum != nullptr);
  NN_CHECK(kernels.relu != nullptr);
  NN_CHECK(kernels.relu_backward != nullptr);
  NN_CHECK(kernels.add != nullptr);
  NN_CHECK(kernels.scale != nullptr);
  NN_CHECK(kernels.axpy != nullptr);
  NN_CHECK(kernels.fill != nullptr);
  return kernels;
}

NN_TEST(test_fill) {
  const auto& k = init();

  std::vector<float> X(10, -1.0f);
  k.fill(42.0f, X.data(), static_cast<int64_t>(X.size()));
  for (const auto& x : X) {
    NN_CHECK(x == 42.0f);
  }
}

NN_TEST(test_scale) {
  const auto& k = init();

  std::vector<float> X{1, 2, 3};
  k.scale(2.0f, X.data(), static_cast<int64_t>(X.size()));
  NN_CHECK(X[0] == 2.0f);
  NN_CHECK(X[1] == 4.0f);
  NN_CHECK(X[2] == 6.0f);
}

NN_TEST(test_axpy) {
  const auto& k = init();

  std::vector<float> X{1, 2};
  std::vector<float> Y{10, 20};
  k.axpy(2.0f, X.data(), Y.data(), static_cast<int64_t>(X.size()));
  NN_CHECK(Y[0] == 12.0f);
  NN_CHECK(Y[1] == 24.0f);
}

NN_TEST(test_add) {
  const auto& k = init();

  std::vector<float> A{1, 2};
  std::vector<float> B{10, 20};
  std::vector<float> C(2);
  k.add(A.data(), B.data(), C.data(), static_cast<int64_t>(A.size()));
  NN_CHECK(C[0] == 11.0f);
  NN_CHECK(C[1] == 22.0f);
}

NN_TEST(test_relu) {
  const auto& k = init();

  std::vector<float> X{-1, 0, 1, 2};
  std::vector<float> Y(4);
  k.relu(X.data(), Y.data(), static_cast<int64_t>(X.size()));
  NN_CHECK(Y[0] == 0.0f);
  NN_CHECK(Y[1] == 0.0f);
  NN_CHECK(Y[2] == 1.0f);
  NN_CHECK(Y[3] == 2.0f);
}

NN_TEST(test_relu_backward) {
  const auto& k = init();

  std::vector<float> X{-1, 0, 1, 2};
  std::vector<float> gY{10, 20, 30, 40};
  std::vector<float> gX(4);
  k.relu_backward(X.data(), gY.data(), gX.data(), static_cast<int64_t>(X.size()));
  NN_CHECK(gX[0] == 0.0f);
  NN_CHECK(gX[1] == 0.0f);
  NN_CHECK(gX[2] == 30.0f);
  NN_CHECK(gX[3] == 40.0f);
}

NN_TEST(test_add_row_bias) {
  const auto& k = init();

  std::vector<float> X{1, 1, 1, 1, 1, 1};
  std::vector<float> b{10, 20, 30};
  std::vector<float> Y(6);
  k.add_row_bias(X.data(), b.data(), Y.data(), 2, 3);
  NN_CHECK(Y[0] == 11.0f);
  NN_CHECK(Y[1] == 21.0f);
  NN_CHECK(Y[2] == 31.0f);
  NN_CHECK(Y[3] == 11.0f);
  NN_CHECK(Y[4] == 21.0f);
  NN_CHECK(Y[5] == 31.0f);
}

NN_TEST(test_col_sum) {
  const auto& k = init();

  std::vector<float> X{1, 2, 3, 4, 5, 6};
  std::vector<float> out(3);
  k.col_sum(X.data(), out.data(), 2, 3);
  NN_CHECK(out[0] == 5.0f);
  NN_CHECK(out[1] == 7.0f);
  NN_CHECK(out[2] == 9.0f);
}