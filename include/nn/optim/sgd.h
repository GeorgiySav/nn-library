#pragma once

#include <vector>

#include <nn/core/tensor.h>
#include <nn/autograd/tape.h>
#include <nn/ops/ops.h>

namespace nn::optim {

class SGD {
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

}