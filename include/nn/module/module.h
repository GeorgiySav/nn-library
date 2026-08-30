#pragma once

#include <stdexcept>
#include <string>
#include <utility>
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
  // Deduplicated by AutogradMeta identity: two Tensor fields tied together
  // (e.g. an LM head sharing an Embedding's weight -- see Linear::weight())
  // alias the same storage and the same AutogradMeta, so they must count as
  // one parameter here. Left un-deduplicated, an optimiser would allocate two
  // separate moment buffers and apply two separate updates to the one
  // storage they share, and a gradient clip would count its norm twice.
  void collect_parameters(std::vector<Tensor*>& out) {
    for (NamedTensor& p : named_parameters()) {
      AutogradMeta* m = p.t->meta();
      if (m) {
        bool tied = false;
        for (Tensor* existing : out) {
          if (existing->meta() == m) { tied = true; break; }
        }
        if (tied) continue;
      }
      out.push_back(p.t);
    }
  }

  void zero_grad() { for (Tensor* p : parameters()) p->zero_grad(); }

  virtual void set_training(bool on) { training_ = on; }
  void train() { set_training(true); }
  void eval()  { set_training(false); }
  bool training() const { return training_; }

  // Moves every parameter slot to d. This walks the full named list rather
  // than the deduplicated parameters(): a tied pair is still two distinct
  // Tensor fields even though they alias one storage, and Tensor::to() across
  // devices allocates a fresh Storage rather than moving the shared one in
  // place -- so every slot needs its own reassignment, or an unvisited alias
  // would keep pointing at the old device. The tie survives the move by
  // reusing the same moved Tensor for every slot that shares its original
  // AutogradMeta, instead of moving each one independently.
  void to(Device d) {
    std::vector<std::pair<AutogradMeta*, Tensor>> moved;
    for (NamedTensor& p : named_parameters()) {
      AutogradMeta* key = p.t->meta();
      Tensor* reuse = nullptr;
      if (key) {
        for (auto& entry : moved) {
          if (entry.first == key) { reuse = &entry.second; break; }
        }
      }
      if (reuse) {
        *p.t = *reuse;
        continue;
      }

      const bool rg = p.t->requires_grad();
      Tensor out = p.t->to(d);
      out.set_requires_grad(rg);
      if (key) moved.emplace_back(key, out);
      *p.t = out;
    }
  }

protected:
  bool training_ = true;
};

}  // namespace nn
