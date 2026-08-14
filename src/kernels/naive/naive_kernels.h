#pragma once

#include <cstdint>

namespace nn::kernels {

void naive_gemm(const float* A, const float* B, float* C, int M, int N, int K, bool transA, bool transB);

}