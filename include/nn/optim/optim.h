#pragma once

#include <vector>

#include <nn/core/tensor.h>
#include <nn/autograd/tape.h>
#include <nn/ops/ops.h>

namespace nn::optim {

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

class Adam : public Optimizer {
public:
  Adam(std::vector<Tensor*> params, float lr, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f)
    : params_(std::move(params)), lr_(lr), beta1_(beta1), beta2_(beta2), eps_(eps) {
    for (Tensor* p : params_) {
      m_.emplace_back(Tensor::zeros(p->shape(), p->device(), p->dtype()));
      v_.emplace_back(Tensor::zeros(p->shape(), p->device(), p->dtype()));
    }
  }

  void step() {
    autograd::NoGradScope no_grad;
    step_++;
    for (size_t i{0u}; i < params_.size(); ++i) {
      Tensor* p = params_[i];

      AutogradMeta* m = p->meta();
      if (!m || !m->grad.defined()) continue;

      ops::adam(*p, m->grad, m_[i], v_[i],
                 lr_, beta1_, beta2_, eps_, step_);
    }
  }

  void zero_grad() { for (Tensor* p : params_) p->zero_grad(); }

private:
  std::vector<Tensor*> params_;
  std::vector<Tensor> m_, v_;
  int step_ = 0;
  float lr_;
  float beta1_ = 0.9f;
  float beta2_ = 0.999f;
  float eps_ = 1e-8f;
};


}