#pragma once

#include <algorithm>
#include <cmath>
#include <functional>
#include <string>
#include <span>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include <nn/core/state.h>
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

  // set_to_none frees each parameter's gradient instead of zeroing it in
  // place, matching PyTorch's zero_grad(set_to_none=True) default.
  // When every parameter is touched every step, leave this false, since
  // freeing and reallocating each grad buffer every step would be pure
  // churn for no benefit.
  virtual void zero_grad(bool set_to_none = false) = 0;

  virtual void collect_state(const std::string& prefix,
                             std::vector<NamedTensor>& tensors,
                             std::vector<NamedScalar>& scalars) = 0;

  // Called with the set of tensor names actually present in a checkpoint
  // (already prefixed, e.g. "opt.m.3") before collect_state() is used to
  // load it. An optimizer with lazily allocated state (AdamW's m/v moments,
  // see AdamW::collect_state) needs this to pre-allocate exactly the
  // entries the file has, since collect_state() only hands out pointers for
  // state it has already allocated, and load() requires every tensor it is
  // asked for to already exist at the right shape. An optimizer with no
  // lazy state (SGD) has nothing to do here.
  virtual void prepare_for_load(const std::string& prefix,
                                const std::unordered_set<std::string>& available) {
    (void)prefix; (void)available;
  }

  virtual void apply_state(const std::string& prefix,
                           std::span<const NamedScalar> scalars) = 0;

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

  void zero_grad(bool set_to_none = false) override {
    for (Tensor* p : params_) p->zero_grad(set_to_none);
  }

  // Plain SGD carries no state; the rate is the caller's or a schedule's.
  void collect_state(const std::string& prefix, std::vector<NamedTensor>&,
                     std::vector<NamedScalar>& scalars) override {
    scalars.push_back({prefix + "lr", double(lr_)});
  }

  void apply_state(const std::string& prefix,
                   std::span<const NamedScalar> scalars) override {
    lr_ = float(scalar_value(scalars, prefix + "lr"));
  }

private:
  std::vector<Tensor*> params_;
};

// excludes biases and norm gains (rank < 2) from weight decay, the common
// default for transformer-style models
inline bool decay_by_rank(const Tensor& p) { return p.shape().rank() >= 2; }

class AdamW : public Optimizer {
public:
  AdamW(std::vector<Tensor*> params, float lr, float weight_decay = 0.01f,
        float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f,
        const std::function<bool(const Tensor&)>& should_decay = decay_by_rank)
    : Optimizer(lr), params_(std::move(params)),
      m_(params_.size()), v_(params_.size()),
      beta1_(beta1), beta2_(beta2), eps_(eps) {
    for (Tensor* p : params_) {
      wd_.push_back(should_decay(*p) ? weight_decay : 0.0f);
    }
    // m_ and v_ start undefined and are allocated lazily in step(), the
    // first time a parameter actually has a gradient to update
  }

  void step() override {
    autograd::NoGradScope no_grad;
    step_++;
    for (size_t i{0u}; i < params_.size(); ++i) {
      Tensor* p = params_[i];

      AutogradMeta* m = p->meta();
      if (!m || !m->grad.defined()) continue;

      // first gradient this parameter has ever produced, allocate its
      // moments now
      if (!m_[i].defined()) {
        m_[i] = Tensor::zeros(p->shape(), p->device(), p->dtype());
        v_[i] = Tensor::zeros(p->shape(), p->device(), p->dtype());
      }

      // updates the first and second moment estimates, applies bias
      // correction for step_, and does the decoupled weight decay update
      // for this parameter
      ops::adam(*p, m->grad, m_[i], v_[i],
                lr_, beta1_, beta2_, eps_, wd_[i], step_);
    }
  }

  void zero_grad(bool set_to_none = false) override {
    for (Tensor* p : params_) p->zero_grad(set_to_none);
  }

  void collect_state(const std::string& prefix, std::vector<NamedTensor>& tensors,
                     std::vector<NamedScalar>& scalars) override {
    for (size_t i = 0; i < params_.size(); ++i) {
      // skip moments never allocated, meaning this parameter has yet to be
      // stepped even once
      if (!m_[i].defined()) continue;
      tensors.push_back({prefix + "m." + std::to_string(i), &m_[i]});
      tensors.push_back({prefix + "v." + std::to_string(i), &v_[i]});
    }
    scalars.push_back({prefix + "step", double(step_)});
    scalars.push_back({prefix + "lr", double(lr_)});
  }

  void prepare_for_load(const std::string& prefix,
                        const std::unordered_set<std::string>& available) override {
    for (size_t i = 0; i < params_.size(); ++i) {
      if (m_[i].defined()) continue;
      if (!available.count(prefix + "m." + std::to_string(i))) continue; // the file never touched this one either
      Tensor* p = params_[i];
      m_[i] = Tensor::zeros(p->shape(), p->device(), p->dtype());
      v_[i] = Tensor::zeros(p->shape(), p->device(), p->dtype());
    }
  }

  void apply_state(const std::string& prefix,
                   std::span<const NamedScalar> scalars) override {
    step_ = int(scalar_value(scalars, prefix + "step"));
    lr_   = float(scalar_value(scalars, prefix + "lr"));
  }

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