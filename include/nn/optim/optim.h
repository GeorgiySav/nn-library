#pragma once

#include <cmath>
#include <functional>
#include <span>
#include <stdexcept>
#include <vector>

#include <nn/core/tensor.h>
#include <nn/autograd/tape.h>
#include <nn/ops/ops.h>

namespace nn::optim {

namespace detail {

// The gradients that exist, skipping parameters backward never reached.
inline Tensor* grad_of(Tensor* p) {
  AutogradMeta* m = p->meta();
  return (m && m->grad.defined()) ? &m->grad : nullptr;
}

}  // namespace detail

// The L2 norm over every gradient at once, as if they were one long vector.
// One reduction per parameter and one host sync for the whole call.
inline float grad_norm(std::span<Tensor* const> params) {
  autograd::NoGradScope no_grad;

  Tensor total;
  for (Tensor* p : params) {
    Tensor* g = detail::grad_of(p);
    if (!g) continue;
    Tensor sq = ops::sum_all(*g, ops::Accum::SumSq);
    if (!total.defined()) total = std::move(sq);
    else                  ops::add_inplace(total, sq);
  }

  return total.defined() ? std::sqrt(total.item()) : 0.0f;
}

// Scales every gradient by one common factor so their combined L2 norm is at
// most max_norm, and returns the norm as it was before clipping.
inline float clip_grad_norm(std::span<Tensor* const> params, float max_norm,
                            float eps = 1e-6f) {
  if (!(max_norm > 0.0f)) {
    throw std::invalid_argument("clip_grad_norm: max_norm must be positive");
  }
  autograd::NoGradScope no_grad;

  const float norm = grad_norm(params);
  const float scale = max_norm / (norm + eps);
  if (!std::isfinite(norm) || scale >= 1.0f) return norm;   // covers norm == 0

  for (Tensor* p : params) {
    if (Tensor* g = detail::grad_of(p)) ops::scale_inplace(*g, scale);
  }
  return norm;
}

// clamp each gradient element to [-value, value]
inline void clip_grad_value(std::span<Tensor* const> params, float value) {
  if (!(value > 0.0f)) {
    throw std::invalid_argument("clip_grad_value: value must be positive");
  }
  autograd::NoGradScope no_grad;

  for (Tensor* p : params) {
    Tensor* g = detail::grad_of(p);
    if (!g) continue;
    ops::scalar_inplace(*g, ops::ScalarOp::ClampMin, -value);
    ops::scalar_inplace(*g, ops::ScalarOp::ClampMax,  value);
  }
}

class Optimizer {
public:
  virtual ~Optimizer() = default;
  virtual void step() = 0;
  virtual void zero_grad() = 0;
};

class SGD : public Optimizer {
public:
  SGD(std::vector<Tensor*> params, float lr)
    : params_(std::move(params)), lr_(lr) {}

  void step() {
    autograd::NoGradScope no_grad;
    for (Tensor* p : params_) {
      AutogradMeta* m = p->meta();
      if (!m || !m->grad.defined()) continue;
      ops::axpy_inplace(*p, -lr_, m->grad); // p -= lr * g
    }
  }

  void zero_grad() { for (Tensor* p : params_) p->zero_grad(); }

private:
  std::vector<Tensor*> params_;
  float lr_;
};

inline bool decay_by_rank(const Tensor& p) { return p.shape().rank() >= 2; }

class AdamW : public Optimizer {
public:
  AdamW(std::vector<Tensor*> params, float lr, float weight_decay = 0.01f,
        float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f,
        const std::function<bool(const Tensor&)>& should_decay = decay_by_rank)
    : params_(std::move(params)), lr_(lr), beta1_(beta1), beta2_(beta2), eps_(eps) {
    for (Tensor* p : params_) {
      m_.emplace_back(Tensor::zeros(p->shape(), p->device(), p->dtype()));
      v_.emplace_back(Tensor::zeros(p->shape(), p->device(), p->dtype()));
      wd_.push_back(should_decay(*p) ? weight_decay : 0.0f);
    }
  }

  void step() override {
    autograd::NoGradScope no_grad;
    step_++;
    for (size_t i{0u}; i < params_.size(); ++i) {
      Tensor* p = params_[i];

      AutogradMeta* m = p->meta();
      if (!m || !m->grad.defined()) continue;

      ops::adam(*p, m->grad, m_[i], v_[i],
                lr_, beta1_, beta2_, eps_, wd_[i], step_);
    }
  }

  void zero_grad() override { for (Tensor* p : params_) p->zero_grad(); }

  float lr() const { return lr_; }
  void set_lr(float lr) { lr_ = lr; }

  std::span<const float> weight_decays() const { return wd_; }

private:
  std::vector<Tensor*> params_;
  std::vector<Tensor> m_, v_;
  std::vector<float> wd_;
  int step_ = 0;
  float lr_;
  float beta1_ = 0.9f;
  float beta2_ = 0.999f;
  float eps_ = 1e-8f;
};

class Adam : public AdamW {
public:
  Adam(std::vector<Tensor*> params, float lr, float beta1 = 0.9f,
       float beta2 = 0.999f, float eps = 1e-8f)
    : AdamW(std::move(params), lr, /*weight_decay=*/0.0f, beta1, beta2, eps) {}
};


}