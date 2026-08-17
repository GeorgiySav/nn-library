#include "test_harness.h"
#include "utilities.h"

#include <nn/core/rng.h>
#include <nn/kernels/kernel_api.h>

nn::kernels::GemmFn gemm() {
  nn::kernels::init_kernels();
  const auto& k = nn::kernels::kernels(nn::Device::CPU);
  NN_CHECK(k.gemm != nullptr);
  return k.gemm;
}

void ref_gemm(const float* A, const float* B, float* C, int M, int N, int K) {
  for (int m{0}; m < M; ++m) {
    for (int n{0}; n < N; ++n) {
      float sum = 0.0f;
      for (int k{0}; k < K; ++k) {
        sum += A[m*K + k] * B[k*N + n];
      }
      C[m*N + n] = sum;
    }
  }
}

std::vector<float> transpose(const std::vector<float>& data, int rows, int cols) {
  std::vector<float> transposed(data.size());
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      transposed[c * rows + r] = data[r * cols + c];
    }
  }
  return transposed;
}

NN_TEST(test_gemm) {
  const std::vector<float> A = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  const std::vector<float> B = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
  std::vector<float> C(4, -1.0f);

  gemm()(cpu_stream(), A.data(), B.data(), C.data(), 2, 2, 3, false, false);

  NN_CHECK_CLOSE(C[0], 58.0f, 1e-6);
  NN_CHECK_CLOSE(C[1], 64.0f, 1e-6);
  NN_CHECK_CLOSE(C[2], 139.0f, 1e-6);
  NN_CHECK_CLOSE(C[3], 154.0f, 1e-6);
}

NN_TEST(test_gemm_transA) {
  const std::vector<float> A = {1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f};
  const std::vector<float> B = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
  std::vector<float> C(4, -1.0f);

  gemm()(cpu_stream(), A.data(), B.data(), C.data(), 2, 2, 3, true, false);

  NN_CHECK_CLOSE(C[0], 58.0f, 1e-6);
  NN_CHECK_CLOSE(C[1], 64.0f, 1e-6);
  NN_CHECK_CLOSE(C[2], 139.0f, 1e-6);
  NN_CHECK_CLOSE(C[3], 154.0f, 1e-6);
}

NN_TEST(test_gemm_transB) {
  const std::vector<float> A = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  const std::vector<float> B = {7.0f, 9.0f, 11.0f, 8.0f, 10.0f, 12.0f};
  std::vector<float> C(4, -1.0f);

  gemm()(cpu_stream(), A.data(), B.data(), C.data(), 2, 2, 3, false, true);

  NN_CHECK_CLOSE(C[0], 58.0f, 1e-6);
  NN_CHECK_CLOSE(C[1], 64.0f, 1e-6);
  NN_CHECK_CLOSE(C[2], 139.0f, 1e-6);
  NN_CHECK_CLOSE(C[3], 154.0f, 1e-6);
}

NN_TEST(test_gemm_trans_consistency) {
  const int M = 4, N = 6, K = 5;
  nn::Pcg32 rng(1234);

  std::vector<float> A(M*K), B(K*N);
  fill_random(A, rng);
  fill_random(B, rng);

  const std::vector<float> At = transpose(A, M, K);
  const std::vector<float> Bt = transpose(B, K, N);

  std::vector<float> base(M*N), c2(M*N), c3(M*N), c4(M*N);
  gemm()(cpu_stream(), A.data(), B.data(), base.data(), M, N, K, false, false);
  gemm()(cpu_stream(), At.data(), B.data(), c2.data(), M, N, K, true, false);
  gemm()(cpu_stream(), A.data(), Bt.data(), c3.data(), M, N, K, false, true);
  gemm()(cpu_stream(), At.data(), Bt.data(), c4.data(), M, N, K, true, true);

  for (int i{0}; i < M*N; ++i) {
    NN_CHECK_CLOSE(base[i], c2[i], 1e-6);
    NN_CHECK_CLOSE(base[i], c3[i], 1e-6);
    NN_CHECK_CLOSE(base[i], c4[i], 1e-6);
  }
}

NN_TEST(test_gemm_outer_product) {
  const std::vector<float> A{2, 3, 5};
  const std::vector<float> B{7, 11, 13, 17};
  std::vector<float> C(12, -1.0f);

  gemm()(cpu_stream(), A.data(), B.data(), C.data(), 3, 4, 1, false, false);

  for (int i{0}; i < 3; ++i) {
    for (int j{0}; j < 4; ++j) {
      NN_CHECK_CLOSE(C[i*4 + j], A[i] * B[j], 1e-6);
    }
  }
}

NN_TEST(test_gemm_single_row) {
  const int M = 1, N = 3, K = 5;
  nn::Pcg32 rng(1234);

  std::vector<float> A(M*K), B(K*N), C(M*N, -1.0f), expected(M*N);
  fill_random(A, rng);
  fill_random(B, rng);

  gemm()(cpu_stream(), A.data(), B.data(), C.data(), M, N, K, false, false);
  ref_gemm(A.data(), B.data(), expected.data(), M, N, K);

  for (int i{0}; i < M*N; ++i) {
    NN_CHECK_CLOSE(expected[i], C[i], 1e-6);
  }
}

NN_TEST(test_gemm_identity) {
  const int n = 7;
  nn::Pcg32 rng(1234);

  std::vector<float> A(n*n), I(n*n, 0.0f), C(n*n, -1.0f);
  fill_random(A, rng);
  for (int i{0}; i < n; ++i) {
    I[i*n + i] = 1.0f;
  }

  gemm()(cpu_stream(), A.data(), I.data(), C.data(), n, n, n, false, false);

  for (int i{0}; i < n*n; ++i) {
    NN_CHECK_CLOSE(A[i], C[i], 1e-6);
  }
}

NN_TEST(test_gemm_associativity) {
  const int M = 4, K = 5, P = 6, N = 7;
  nn::Pcg32 rng(1234);

  std::vector<float> A(M*K), B(K*P), C(P*N);
  fill_random(A, rng);
  fill_random(B, rng);
  fill_random(C, rng);

  std::vector<float> AB(M*P), AB_C(M*N);
  gemm()(cpu_stream(), A.data(), B.data(), AB.data(), M, P, K, false, false);
  gemm()(cpu_stream(), AB.data(), C.data(), AB_C.data(), M, N, P, false, false);

  std::vector<float> BC(K*N), A_BC(M*N);
  gemm()(cpu_stream(), B.data(), C.data(), BC.data(), K, N, P, false, false);
  gemm()(cpu_stream(), A.data(), BC.data(), A_BC.data(), M, N, K, false, false);

  for (int i{0}; i < M*N; ++i) {
    NN_CHECK_CLOSE(AB_C[i], A_BC[i], 1e-6);
  }
}