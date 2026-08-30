// Reductions: sum_to is the primitive, every other form is a shape around it.

#include <nn/ops/ops.h>

#include <stdexcept>
#include <string>
#include <vector>

#include <kernels/kernel_api.h>

#include "ops_common.h"

namespace nn::ops {

// Sum g down to `target`, which must broadcast up to g's shape. This is the
// backward of every broadcast: an axis that was stretched to feed many
// outputs collects the gradient from all of them.
Tensor sum_to(const Tensor& g, const Shape& target) {
  if (g.shape() == target) return g;
  if (target.rank() > g.shape().rank()) {
    throw std::invalid_argument("sum_to: " + target.str() + " has more axes than " +
                                g.shape().str());
  }

  const int r = g.shape().rank();
  const int lead = r - target.rank();

  TensorView keep{}, red{};
  keep.rank = red.rank = r;
  int64_t n_out = 1, n_red = 1;
  for (int i = 0; i < r; ++i) {
    const int ti = i - lead;
    const int td = (ti >= 0) ? target.dim(ti) : 1;
    const int gd = g.shape().dim(i);
    if (td != gd && td != 1) {
      throw std::invalid_argument("sum_to: " + g.shape().str() + " does not reduce to " +
                                  target.str());
    }
    const bool reduced = (td == 1 && gd > 1);
    keep.shape[i]  = td;
    keep.stride[i] = g.stride(i);
    red.shape[i]   = reduced ? gd : 1;
    red.stride[i]  = reduced ? g.stride(i) : 0;
    n_out *= td;
    n_red *= red.shape[i];
  }

  Tensor out(target, g.device(), g.dtype());
  const auto& kk = nn::kernels::kernels(g.device());
  kk.sum_to(current_stream(g.device()), g.device_ptr(), keep, red,
            out.device_ptr(), n_out, n_red);
  return out;
}

Tensor sum_dim(const Tensor& x, int dim, bool keepdim) {
  const int r = x.shape().rank();
  const int d = x.shape().resolve_dim(dim, "sum");

  Shape kept = x.shape();
  kept.set_dim(d, 1);
  Tensor out = sum_to(x, kept);
  if (keepdim) return out;

  int dims[kMaxShapeRank] = {0};
  int n = 0;
  for (int i = 0; i < r; ++i) {
    if (i != d) dims[n++] = x.shape().dim(i);
  }
  return out.reshape_view(Shape(std::span<const int>(dims, n)));
}

Tensor mean_dim(const Tensor& x, int dim, bool keepdim) {
  const int d = x.shape().resolve_dim(dim, "mean");
  const int n = x.shape().dim(d);
  return scalar(ScalarOp::MulScalar, sum_dim(x, d, keepdim), 1.0f / float(n));
}

Tensor sum_all(const Tensor& x, Accum a) {
  Tensor out(Shape{}, x.device(), x.dtype());
  Tensor workspace(Shape{nn::kernels::kSumAllWorkspace}, x.device(), x.dtype());

  const auto& k = nn::kernels::kernels(x.device());
  k.sum_all(current_stream(x.device()), x.device_ptr(), view_of(x), a,
            out.device_ptr(), workspace.device_ptr(), x.numel());
  return out;
}

Tensor mean_all(const Tensor& x) {
  const int64_t n = x.numel();
  if (n == 0) throw std::invalid_argument("mean: empty tensor");
  return scalar(ScalarOp::MulScalar, sum_all(x), 1.0f / float(n));
}

Tensor argmax_rows(const Tensor& x) {
  if (x.shape().rank() != 2) throw std::invalid_argument("argmax rows: x must be 2D");
  Tensor out(Shape{x.shape().dim(0)}, x.device(), DType::I32);
  const auto& k = nn::kernels::kernels(x.device());
  k.argmax_rows(current_stream(x.device()), x.device_ptr(), out.device_ptr_i32(),
                x.shape().dim(0), x.shape().dim(1), row_stride_of(x, "argmax_rows"));
  return out;
}

void topk_rows(const Tensor& x, int k, Tensor& values, Tensor& indices) {
  if (x.shape().rank() != 2) throw std::invalid_argument("topk_rows: x must be 2D");
  const int N = x.shape().dim(1);
  if (k <= 0 || k > N) {
    throw std::invalid_argument("topk_rows: k must be in (0, " + std::to_string(N) + "]");
  }

  values = Tensor(Shape{x.shape().dim(0), k}, x.device(), x.dtype());
  indices = Tensor(Shape{x.shape().dim(0), k}, x.device(), DType::I32);

  const auto& kk = nn::kernels::kernels(x.device());
  kk.topk_rows(current_stream(x.device()), x.device_ptr(), x.shape().dim(0), N, k,
               values.device_ptr(), indices.device_ptr_i32(), row_stride_of(x, "topk_rows"));
}

Tensor multinomial(const Tensor& weights, Pcg32& rng) {
  if (weights.shape().rank() != 2) {
    throw std::invalid_argument("multinomial: weights must be 2D");
  }

  const int M = weights.shape().dim(0);
  const int N = weights.shape().dim(1);
  const Tensor host = weights.pack().to(Device::CPU);
  const float* w = host.host_data();

  const size_t rows = M, cols = N;
  std::vector<int32_t> sampled(rows);

  for (size_t i = 0; i < rows; ++i) {
    const float* row = w + i * cols;
    float total = 0.0f;
    for (size_t j = 0; j < cols; ++j) total += row[j];
    if (!(total > 0.0f)) {
      throw std::invalid_argument("multinomial: row " + std::to_string(i) +
                                  " has no positive weight");
    }

    const float target = rng.next_uniform() * total;
    float cum = 0.0f;
    size_t chosen = cols - 1;   // the last slot a rounding error could leave uncovered
    for (size_t j = 0; j < cols; ++j) {
      cum += row[j];
      if (target < cum) { chosen = j; break; }
    }
    sampled[i] = int32_t(chosen);
  }

  return Tensor::from_i32(sampled, Shape{M}, weights.device());
}

Tensor gather_rows(const Tensor& src, const Tensor& idx) {
  if (src.shape().rank() != 2) throw std::invalid_argument("gather_rows: src must be 2D");
  const int M = src.shape().dim(0);
  const int N = src.shape().dim(1);
  if (idx.shape().rank() != 1 || idx.shape().dim(0) != M) {
    throw std::invalid_argument("gather_rows: idx must be 1D with one entry per row of src");
  }
  same_device(src, idx, "gather_rows");

  const Tensor host_src = src.pack().to(Device::CPU);
  const Tensor host_idx = idx.pack().to(Device::CPU);
  const int32_t* hi = host_idx.host_data_i32();

  const size_t rows = M, cols = N;
  for (size_t i = 0; i < rows; ++i) {
    if (hi[i] < 0 || size_t(hi[i]) >= cols) {
      throw std::invalid_argument("gather_rows: idx[" + std::to_string(i) + "] = " +
                                  std::to_string(hi[i]) + " is out of range for a row of " +
                                  std::to_string(N));
    }
  }

  if (src.dtype() == DType::I32) {
    const int32_t* hs = host_src.host_data_i32();
    std::vector<int32_t> out(rows);
    for (size_t i = 0; i < rows; ++i) out[i] = hs[i * cols + size_t(hi[i])];
    return Tensor::from_i32(out, Shape{M}, src.device());
  }

  const float* hs = host_src.host_data();
  std::vector<float> out(rows);
  for (size_t i = 0; i < rows; ++i) out[i] = hs[i * cols + size_t(hi[i])];
  return Tensor::from(out, Shape{M}, src.device());
}

}  // namespace nn::ops
