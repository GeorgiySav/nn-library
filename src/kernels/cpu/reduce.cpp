#include "cpu_kernels.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <nn/core/device.h>
#include <kernels/random.h>

#include "../strided_index.h"

namespace nn::kernels {

void cpu_argmax_rows(const Stream&, const float* X, int32_t* out, int M, int N, int64_t sx) {
  for (int i = 0; i < M; ++i) {
    const float* row = X + int64_t(i) * sx;
    int best = 0;
    for (int j = 1; j < N; ++j) {
      if (row[j] > row[best]) best = j;
    }
    out[i] = int32_t(best);
  }
}

// sums a gradient back down to a broadcast operand's original shape. keep
// indexes the axes that survive (the operand's own shape) and red indexes
// the axes that were broadcast away; for each surviving position j, offsets
// from keep and red are added together to walk every g element that
// broadcast from it, and those are summed into out[j].
void cpu_sum_to(const Stream&, const float* g, TensorView keep, TensorView red,
                  float* out, int64_t n_out, int64_t n_red) {
  for (int64_t j = 0; j < n_out; ++j) {
    const int64_t base = offset_of(keep, j);
    float acc = 0.0f;
    for (int64_t k = 0; k < n_red; ++k) acc += g[base + offset_of(red, k)];
    out[j] = acc;
  }
}

// accumulates in double despite the float inputs/output, since summing a
// large tensor in float loses precision as the running total grows relative
// to each new term.
void cpu_sum_all(const Stream&, const float* X, TensorView v, Accum a,
                   float* out, float*, int64_t n) {
  double acc = 0.0;
  for (int64_t i = 0; i < n; ++i) acc += double(apply_accum(a, X[offset_of(v, i)]));
  *out = float(acc);
}

void cpu_topk_rows(const Stream&, const float* X, int M, int N, int k,
                   float* values, int32_t* indices, int64_t sx) {
  for (int i = 0; i < M; ++i) {
    const float* row = X + int64_t(i) * sx;
    std::vector<std::pair<float, int>> pairs(N);
    for (int j = 0; j < N; ++j) pairs[j] = {row[j], j};
    std::partial_sort(pairs.begin(), pairs.begin() + k, pairs.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });
    for (int j = 0; j < k; ++j) {
      values[i * k + j] = pairs[j].first;
      indices[i * k + j] = pairs[j].second;
    }
  }
}

namespace {

template <class T>
void gather_rows_impl(const T* src, const int32_t* idx, T* out, int M, int64_t sx) {
  for (int i = 0; i < M; ++i) out[i] = src[int64_t(i) * sx + idx[i]];
}

}  // namespace

// idx is range-checked by ops::gather_rows before this ever runs, so the
// kernel itself just reads without bounds checks.
void cpu_gather_rows(const Stream&, const float* src, const int32_t* idx,
                       float* out, int M, int64_t sx) {
  gather_rows_impl(src, idx, out, M, sx);
}

void cpu_gather_rows_i32(const Stream&, const int32_t* src, const int32_t* idx,
                           int32_t* out, int M, int64_t sx) {
  gather_rows_impl(src, idx, out, M, sx);
}

void cpu_multinomial(const Stream&, const float* W, int32_t* out, int M, int N, int64_t sx,
                       uint64_t seed, uint64_t offset) {
  for (int i = 0; i < M; ++i) {
    const float* row = W + int64_t(i) * sx;
    float total = 0.0f;
    for (int j = 0; j < N; ++j) total += row[j];
    if (!(total > 0.0f)) {
      throw std::invalid_argument("multinomial: row " + std::to_string(i) +
                                  " has no positive weight");
    }

    // inverse-CDF sampling. draw target uniformly over [0, total) and walk
    // the running sum until it passes target.
    const float target = random_uniform(seed, offset + uint64_t(i)) * total;
    float cum = 0.0f;
    int chosen = N - 1;   // the slot a rounding error could leave uncovered
    for (int j = 0; j < N; ++j) {
      cum += row[j];
      if (target < cum) { chosen = j; break; }
    }
    out[i] = chosen;
  }
}

}
