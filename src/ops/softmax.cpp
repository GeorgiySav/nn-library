// Softmax over the last axis, and the fused softmax + cross-entropy.

#include <nn/ops/ops.h>

#include <stdexcept>
#include <string>

#include <kernels/kernel_api.h>

#include "ops_common.h"

namespace nn::ops {

// Private to softmax: collapsing any rank into [M, N] rows.
namespace {

struct Rows {
  Tensor t;         // keeps a materialised copy alive, when one was needed
  int M = 0;
  int N = 0;
  int64_t stride = 0;
};

Rows rows_of(const Tensor& x) {
  const int r = x.shape().rank();
  const int N = (r == 0) ? 1 : x.shape().dim(r - 1);
  const int M = (N > 0) ? int(x.numel() / N) : 0;

  if (x.is_contiguous()) return {x, M, N, N};

  const TensorView v = view_of(x);
  if (v.rank == 2 && v.shape[1] == N && v.stride[1] == 1) {
    return {x, M, N, v.stride[0]};
  }
  return {x.pack(), M, N, N};
}

}  // namespace

Tensor softmax_rows(const Tensor& x) {
  const Rows r = rows_of(x);
  Tensor out(x.shape(), x.device(), x.dtype());
  const auto& k = nn::kernels::kernels(x.device());
  k.softmax_rows(current_stream(x.device()), r.t.device_ptr(), out.device_ptr(),
                 r.M, r.N, r.stride);
  return out;
}

Tensor softmax_rows_backward(const Tensor& y, const Tensor& g) {
  same_device(y, g, "softmax_backward");
  same_shape(y, g, "softmax_backward");

  const Rows ry = rows_of(y);
  const Rows rg = rows_of(g);
  Tensor gx(y.shape(), y.device(), y.dtype());
  const auto& k = nn::kernels::kernels(y.device());
  k.softmax_rows_backward(current_stream(y.device()), ry.t.device_ptr(),
                          rg.t.device_ptr(), gx.device_ptr(),
                          ry.M, ry.N, ry.stride, rg.stride);
  return gx;
}

void softmax_ce(const Tensor& logits, const Tensor& labels, Tensor& loss_out, Tensor& probs) {
  same_device(logits, labels, "softmax_ce");
  same_device(labels, loss_out, "softmax_ce");
  same_device(probs, loss_out, "softmax_ce");

  if (logits.shape().rank() != 2 || labels.shape().rank() != 1) {
    throw std::invalid_argument("logits must be 2D and labels must be 1D");
  }
  if (logits.shape().dim(0) != labels.shape().dim(0)) {
    throw std::invalid_argument("Number of samples in logits and labels must match");
  }
  if (loss_out.shape().rank() != 0) {
    throw std::invalid_argument("loss_out must be a scalar tensor");
  }
  if (probs.shape() != logits.shape()) {
    throw std::invalid_argument("probs must have the same shape as logits");
  }

  require_contiguous(probs, "softmax_ce (probs)");

  const auto& k = nn::kernels::kernels(logits.device());
  k.softmax_ce(current_stream(logits.device()), logits.device_ptr(), labels.device_ptr_i32(),
               loss_out.device_ptr(), probs.device_ptr(),
               logits.shape().dim(0), logits.shape().dim(1),
               row_stride_of(logits, "softmax_ce"));
}

Tensor softmax_ce_backward(const Tensor& probs, const Tensor& labels, const Tensor& g_loss) {
  same_device(probs, labels, "softmax_ce_backward");
  same_device(g_loss, labels, "softmax_ce_backward");

  if (probs.shape().rank() != 2 || labels.shape().rank() != 1) {
    throw std::invalid_argument("probs must be 2D and labels must be 1D");
  }
  if (probs.shape().dim(0) != labels.shape().dim(0)) {
    throw std::invalid_argument("Number of samples in probs and lebls must match");
  }
  if (g_loss.shape().rank() != 0) {
    throw std::invalid_argument("g_loss must be a scalar tensor");
  }

  Tensor g_logits(probs.shape(), probs.device(), probs.dtype());
  const auto& k = nn::kernels::kernels(g_logits.device());
  k.softmax_ce_backward(current_stream(probs.device()), probs.device_ptr(), labels.device_ptr_i32(),
                        g_loss.device_ptr(), g_logits.device_ptr(),
                        probs.shape().dim(0), probs.shape().dim(1),
                        row_stride_of(probs, "softmax_ce_backward"));
  return g_logits;
}

void softmax_ce_weighted(const Tensor& logits, const Tensor& labels, const Tensor& weights,
                         Tensor& loss_out, Tensor& probs) {
  same_device(logits, labels, "softmax_ce_weighted");
  same_device(labels, weights, "softmax_ce_weighted");
  same_device(labels, loss_out, "softmax_ce_weighted");
  same_device(probs, loss_out, "softmax_ce_weighted");

  if (logits.shape().rank() != 2 || labels.shape().rank() != 1) {
    throw std::invalid_argument("logits must be 2D and labels must be 1D");
  }
  if (logits.shape().dim(0) != labels.shape().dim(0)) {
    throw std::invalid_argument("Number of samples in logits and labels must match");
  }
  if (weights.shape().rank() != 1) {
    throw std::invalid_argument("weights must be 1D");
  }
  if (weights.shape().dim(0) != logits.shape().dim(0)) {
    throw std::invalid_argument("Number of samples in logits and weights must match");
  }
  if (loss_out.shape().rank() != 0) {
    throw std::invalid_argument("loss_out must be a scalar tensor");
  }
  if (probs.shape() != logits.shape()) {
    throw std::invalid_argument("probs must have the same shape as logits");
  }

  require_contiguous(probs, "softmax_ce_weighted (probs)");

  const auto& k = nn::kernels::kernels(logits.device());
  k.softmax_ce_weighted(current_stream(logits.device()), logits.device_ptr(), labels.device_ptr_i32(),
                        weights.device_ptr(), loss_out.device_ptr(), probs.device_ptr(),
                        logits.shape().dim(0), logits.shape().dim(1),
                        row_stride_of(logits, "softmax_ce_weighted"));
}

Tensor softmax_ce_weighted_backward(const Tensor& probs, const Tensor& labels,
                                    const Tensor& weights, const Tensor& g_loss) {
  same_device(probs, labels, "softmax_ce_weighted_backward");
  same_device(labels, weights, "softmax_ce_weighted_backward");
  same_device(g_loss, labels, "softmax_ce_weighted_backward");

  if (probs.shape().rank() != 2 || labels.shape().rank() != 1) {
    throw std::invalid_argument("probs must be 2D and labels must be 1D");
  }
  if (probs.shape().dim(0) != labels.shape().dim(0)) {
    throw std::invalid_argument("Number of samples in probs and lebls must match");
  }
  if (weights.shape().rank() != 1) {
    throw std::invalid_argument("weights must be 1D");
  }
  if (weights.shape().dim(0) != probs.shape().dim(0)) {
    throw std::invalid_argument("Number of samples in probs and weights must match");
  }
  if (g_loss.shape().rank() != 0) {
    throw std::invalid_argument("g_loss must be a scalar tensor");
  }

  Tensor g_logits(probs.shape(), probs.device(), probs.dtype());
  const auto& k = nn::kernels::kernels(g_logits.device());
  k.softmax_ce_weighted_backward(current_stream(probs.device()), probs.device_ptr(), labels.device_ptr_i32(),
                                 weights.device_ptr(), g_loss.device_ptr(), g_logits.device_ptr(),
                                 probs.shape().dim(0), probs.shape().dim(1),
                                 row_stride_of(probs, "softmax_ce_weighted_backward"));
  return g_logits;
}

}  // namespace nn::ops
