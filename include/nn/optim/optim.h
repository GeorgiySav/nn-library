#pragma once

#include <algorithm>
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
  explicit Optimizer(float lr) : lr_(lr) {}
  virtual ~Optimizer() = default;
  virtual void step() = 0;
  virtual void zero_grad() = 0;

  // The learning rate lives here rather than in each subclass so a Schedule
  // can drive any optimiser. Changing it between steps is the point; nothing
  // else about the optimiser's state depends on it.
  float lr() const { return lr_; }
  void set_lr(float lr) { lr_ = lr; }

protected:
  float lr_;
};

class SGD : public Optimizer {
public:
  SGD(std::vector<Tensor*> params, float lr)
    : Optimizer(lr), params_(std::move(params)) {}

  void step() override {
    autograd::NoGradScope no_grad;
    for (Tensor* p : params_) {
      AutogradMeta* m = p->meta();
      if (!m || !m->grad.defined()) continue;
      ops::axpy_inplace(*p, -lr_, m->grad); // p -= lr * g
    }
  }

  void zero_grad() override { for (Tensor* p : params_) p->zero_grad(); }

private:
  std::vector<Tensor*> params_;
};

inline bool decay_by_rank(const Tensor& p) { return p.shape().rank() >= 2; }

class AdamW : public Optimizer {
public:
  AdamW(std::vector<Tensor*> params, float lr, float weight_decay = 0.01f,
        float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f,
        const std::function<bool(const Tensor&)>& should_decay = decay_by_rank)
    : Optimizer(lr), params_(std::move(params)),
      beta1_(beta1), beta2_(beta2), eps_(eps) {
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

  // The decay each parameter actually got, in the order they were given.
  std::span<const float> weight_decays() const { return wd_; }

private:
  std::vector<Tensor*> params_;
  std::vector<Tensor> m_, v_;
  std::vector<float> wd_;
  int step_ = 0;
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


enum class Decay { Constant, Linear, Cosine };

//   nn::optim::Schedule sched(3e-4f, /*total=*/10000, /*warmup=*/500);
//   for (int64_t s = 0; s < 10000; ++s) {
//     opt.set_lr(sched.at(s));
//     ...  forward, backward, clip  ...
//     opt.step();
//   }
class Schedule {
public:
  Schedule(float peak_lr, int64_t total, int64_t warmup = 0,
           Decay decay = Decay::Cosine, float min_lr = 0.0f)
    : peak_(peak_lr), min_(min_lr), total_(total), warmup_(warmup), decay_(decay) {
    if (total <= 0)          throw std::invalid_argument("Schedule: total must be positive");
    if (warmup < 0)          throw std::invalid_argument("Schedule: warmup cannot be negative");
    if (warmup > total)      throw std::invalid_argument("Schedule: warmup exceeds total");
    if (!(peak_lr >= min_lr)) throw std::invalid_argument("Schedule: peak_lr is below min_lr");
    if (min_lr < 0.0f)       throw std::invalid_argument("Schedule: min_lr cannot be negative");
  }

  float at(int64_t step) const {
    if (step < 0) throw std::invalid_argument("Schedule: step cannot be negative");

    if (step < warmup_) {
      return peak_ * float(step + 1) / float(warmup_);
    }

    const int64_t span = total_ - warmup_;
    if (span <= 0) return min_;
    const float t = std::min(1.0f, float(step - warmup_) / float(span));

    switch (decay_) {
      case Decay::Constant: return peak_;
      case Decay::Linear:   return min_ + (peak_ - min_) * (1.0f - t);
      case Decay::Cosine:   break;
    }
    return min_ + 0.5f * (peak_ - min_) * (1.0f + std::cos(3.14159265358979f * t));
  }

  float operator()(int64_t step) const { return at(step); }

  float peak_lr() const { return peak_; }
  float min_lr()  const { return min_; }
  int64_t total() const { return total_; }
  int64_t warmup() const { return warmup_; }

private:
  float peak_, min_;
  int64_t total_, warmup_;
  Decay decay_;
};


}