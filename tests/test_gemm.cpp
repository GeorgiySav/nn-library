#include "test_harness.h"
#include "devices.h"
#include "utilities.h"

#include <nn/core/rng.h>
#include <nn/core/tensor.h>
#include <kernels/kernel_api.h>

nn::kernels::GemmFn* gemm(nn::Device d) {
  nn::kernels::init_kernels();
  const auto& k = nn::kernels::kernels(d);
  NN_CHECK(k.gemm != nullptr);
  return k.gemm;
}

void gemm_dense(nn::Device d, const nn::Stream& s, const float* A, const float* B,
                float* C, int M, int N, int K, bool transA, bool transB) {
  gemm(d)(s, A, B, C, M, N, K,
          /*lda=*/transA ? M : K, /*ldb=*/transB ? K : N, /*ldc=*/N,
          transA, transB, /*batch=*/1, 0, 0, 0);
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

// Random host matrix plus its device copy, so tests that need a host-side
// reference and a device-side input do not generate the numbers twice.
struct Matrix {
  std::vector<float> host;
  nn::Tensor dev;
};

Matrix random_matrix(int rows, int cols, nn::Pcg32& rng, nn::Device d) {
  std::vector<float> h(size_t(rows) * size_t(cols));
  fill_random(h, rng);
  return {h, nn::Tensor::from(h, nn::Shape({rows, cols}), d)};
}

NN_TEST(test_gemm) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    const nn::Tensor A = nn::Tensor::from({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, dev);
    const nn::Tensor B = nn::Tensor::from({7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f}, dev);
    nn::Tensor C = nn::Tensor::full({4}, -1.0f, dev);

    gemm_dense(dev, nn::current_stream(dev), A.device_ptr(), B.device_ptr(),
              C.device_ptr(), 2, 2, 3, false, false);

    const nn::Tensor h = C.to(nn::Device::CPU);
    NN_CHECK_CLOSE(h.host_data()[0], 58.0f, 1e-6);
    NN_CHECK_CLOSE(h.host_data()[1], 64.0f, 1e-6);
    NN_CHECK_CLOSE(h.host_data()[2], 139.0f, 1e-6);
    NN_CHECK_CLOSE(h.host_data()[3], 154.0f, 1e-6);
  }
}

NN_TEST(test_gemm_transA) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    const nn::Tensor A = nn::Tensor::from({1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f}, dev);
    const nn::Tensor B = nn::Tensor::from({7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f}, dev);
    nn::Tensor C = nn::Tensor::full({4}, -1.0f, dev);

    gemm_dense(dev, nn::current_stream(dev), A.device_ptr(), B.device_ptr(),
              C.device_ptr(), 2, 2, 3, true, false);

    const nn::Tensor h = C.to(nn::Device::CPU);
    NN_CHECK_CLOSE(h.host_data()[0], 58.0f, 1e-6);
    NN_CHECK_CLOSE(h.host_data()[1], 64.0f, 1e-6);
    NN_CHECK_CLOSE(h.host_data()[2], 139.0f, 1e-6);
    NN_CHECK_CLOSE(h.host_data()[3], 154.0f, 1e-6);
  }
}

NN_TEST(test_gemm_transB) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    const nn::Tensor A = nn::Tensor::from({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, dev);
    const nn::Tensor B = nn::Tensor::from({7.0f, 9.0f, 11.0f, 8.0f, 10.0f, 12.0f}, dev);
    nn::Tensor C = nn::Tensor::full({4}, -1.0f, dev);

    gemm_dense(dev, nn::current_stream(dev), A.device_ptr(), B.device_ptr(),
              C.device_ptr(), 2, 2, 3, false, true);

    const nn::Tensor h = C.to(nn::Device::CPU);
    NN_CHECK_CLOSE(h.host_data()[0], 58.0f, 1e-6);
    NN_CHECK_CLOSE(h.host_data()[1], 64.0f, 1e-6);
    NN_CHECK_CLOSE(h.host_data()[2], 139.0f, 1e-6);
    NN_CHECK_CLOSE(h.host_data()[3], 154.0f, 1e-6);
  }
}

NN_TEST(test_gemm_trans_consistency) {
  const int M = 4, N = 6, K = 5;

  NN_TEST_FOR_EACH_DEVICE(dev) {
    nn::Pcg32 rng(1234);
    const Matrix A = random_matrix(M, K, rng, dev);
    const Matrix B = random_matrix(K, N, rng, dev);

    const nn::Tensor At = nn::Tensor::from(transpose(A.host, M, K), nn::Shape({K, M}), dev);
    const nn::Tensor Bt = nn::Tensor::from(transpose(B.host, K, N), nn::Shape({N, K}), dev);

    nn::Tensor base = nn::Tensor::zeros({M, N}, dev);
    nn::Tensor c2   = nn::Tensor::zeros({M, N}, dev);
    nn::Tensor c3   = nn::Tensor::zeros({M, N}, dev);
    nn::Tensor c4   = nn::Tensor::zeros({M, N}, dev);

    const nn::Stream& s = nn::current_stream(dev);
    gemm_dense(dev, s, A.dev.device_ptr(), B.dev.device_ptr(), base.device_ptr(), M, N, K, false, false);
    gemm_dense(dev, s, At.device_ptr(),    B.dev.device_ptr(), c2.device_ptr(),   M, N, K, true,  false);
    gemm_dense(dev, s, A.dev.device_ptr(), Bt.device_ptr(),    c3.device_ptr(),   M, N, K, false, true);
    gemm_dense(dev, s, At.device_ptr(),    Bt.device_ptr(),    c4.device_ptr(),   M, N, K, true,  true);

    const nn::Tensor hb = base.to(nn::Device::CPU);
    const nn::Tensor h2 = c2.to(nn::Device::CPU);
    const nn::Tensor h3 = c3.to(nn::Device::CPU);
    const nn::Tensor h4 = c4.to(nn::Device::CPU);

    for (int i{0}; i < M*N; ++i) {
      NN_CHECK_CLOSE(hb.host_data()[i], h2.host_data()[i], 1e-6);
      NN_CHECK_CLOSE(hb.host_data()[i], h3.host_data()[i], 1e-6);
      NN_CHECK_CLOSE(hb.host_data()[i], h4.host_data()[i], 1e-6);
    }
  }
}

NN_TEST(test_gemm_outer_product) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    const std::vector<float> a{2, 3, 5};
    const std::vector<float> b{7, 11, 13, 17};

    const nn::Tensor A = nn::Tensor::from(a, nn::Shape({3, 1}), dev);
    const nn::Tensor B = nn::Tensor::from(b, nn::Shape({1, 4}), dev);
    nn::Tensor C = nn::Tensor::full({3, 4}, -1.0f, dev);

    gemm_dense(dev, nn::current_stream(dev), A.device_ptr(), B.device_ptr(),
              C.device_ptr(), 3, 4, 1, false, false);

    const nn::Tensor h = C.to(nn::Device::CPU);
    for (int i{0}; i < 3; ++i) {
      for (int j{0}; j < 4; ++j) {
        NN_CHECK_CLOSE(h.host_data()[i*4 + j], a[i] * b[j], 1e-6);
      }
    }
  }
}

NN_TEST(test_gemm_single_row) {
  const int M = 1, N = 3, K = 5;

  NN_TEST_FOR_EACH_DEVICE(dev) {
    nn::Pcg32 rng(1234);
    const Matrix A = random_matrix(M, K, rng, dev);
    const Matrix B = random_matrix(K, N, rng, dev);
    nn::Tensor C = nn::Tensor::full({M, N}, -1.0f, dev);

    gemm_dense(dev, nn::current_stream(dev), A.dev.device_ptr(), B.dev.device_ptr(),
              C.device_ptr(), M, N, K, false, false);

    std::vector<float> expected(M*N);
    ref_gemm(A.host.data(), B.host.data(), expected.data(), M, N, K);

    const nn::Tensor h = C.to(nn::Device::CPU);
    for (int i{0}; i < M*N; ++i) {
      NN_CHECK_CLOSE(expected[i], h.host_data()[i], 1e-6);
    }
  }
}

NN_TEST(test_gemm_identity) {
  const int n = 7;

  NN_TEST_FOR_EACH_DEVICE(dev) {
    nn::Pcg32 rng(1234);
    const Matrix A = random_matrix(n, n, rng, dev);

    std::vector<float> identity(size_t(n)*size_t(n), 0.0f);
    for (int i{0}; i < n; ++i) identity[size_t(i)*n + i] = 1.0f;
    const nn::Tensor I = nn::Tensor::from(identity, nn::Shape({n, n}), dev);

    nn::Tensor C = nn::Tensor::full({n, n}, -1.0f, dev);
    gemm_dense(dev, nn::current_stream(dev), A.dev.device_ptr(), I.device_ptr(),
              C.device_ptr(), n, n, n, false, false);

    const nn::Tensor h = C.to(nn::Device::CPU);
    for (int i{0}; i < n*n; ++i) {
      NN_CHECK_CLOSE(A.host[i], h.host_data()[i], 1e-6);
    }
  }
}

NN_TEST(test_gemm_associativity) {
  const int M = 4, K = 5, P = 6, N = 7;

  NN_TEST_FOR_EACH_DEVICE(dev) {
    nn::Pcg32 rng(1234);
    const Matrix A = random_matrix(M, K, rng, dev);
    const Matrix B = random_matrix(K, P, rng, dev);
    const Matrix C = random_matrix(P, N, rng, dev);

    const nn::Stream& s = nn::current_stream(dev);

    nn::Tensor AB   = nn::Tensor::zeros({M, P}, dev);
    nn::Tensor AB_C = nn::Tensor::zeros({M, N}, dev);
    gemm_dense(dev, s, A.dev.device_ptr(), B.dev.device_ptr(), AB.device_ptr(),   M, P, K, false, false);
    gemm_dense(dev, s, AB.device_ptr(),    C.dev.device_ptr(), AB_C.device_ptr(), M, N, P, false, false);

    nn::Tensor BC   = nn::Tensor::zeros({K, N}, dev);
    nn::Tensor A_BC = nn::Tensor::zeros({M, N}, dev);
    gemm_dense(dev, s, B.dev.device_ptr(), C.dev.device_ptr(), BC.device_ptr(),   K, N, P, false, false);
    gemm_dense(dev, s, A.dev.device_ptr(), BC.device_ptr(),    A_BC.device_ptr(), M, N, K, false, false);

    const nn::Tensor h1 = AB_C.to(nn::Device::CPU);
    const nn::Tensor h2 = A_BC.to(nn::Device::CPU);
    for (int i{0}; i < M*N; ++i) {
      NN_CHECK_CLOSE(h1.host_data()[i], h2.host_data()[i], 1e-6);
    }
  }
}
