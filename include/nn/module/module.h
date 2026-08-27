#pragma once

#include <stdexcept>
#include <string>
#include <vector>

#include <nn/core/state.h>
#include <nn/core/tensor.h>

namespace nn {

class Module {
public:
  virtual ~Module() = default;

  virtual Tensor forward(const Tensor&) {
    throw std::logic_error("Module::forward(x): not implemented for this module");
  }

  // name what you own, under the path you were given.
  virtual void collect_named(const std::string& prefix,
                            std::vector<NamedTensor>& out) = 0;

  std::vector<NamedTensor> named_parameters() {
    std::vector<NamedTensor> v;
    collect_named("", v);
    return v;
  }

  std::vector<Tensor*> parameters() {
    std::vector<Tensor*> p;
    collect_parameters(p);
    return p;
  }

  // The flat view, for an optimiser or a clip that does not care about names.
  void collect_parameters(std::vector<Tensor*>& out) {
    for (NamedTensor& p : named_parameters()) out.push_back(p.t);
  }

  void zero_grad() { for (Tensor* p : parameters()) p->zero_grad(); }

  virtual void set_training(bool on) { training_ = on; }
  void train() { set_training(true); }
  void eval()  { set_training(false); }
  bool training() const { return training_; }

  void to(Device d) {
    for (Tensor* p : parameters()) {
      const bool rg = p->requires_grad();
      *p = p->to(d);
      p->set_requires_grad(rg);
    }
  }

protected:
  bool training_ = true;
};

}  // namespace nn
