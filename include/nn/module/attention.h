#pragma once

#include <nn/core/rng.h>
#include <nn/core/tensor.h>
#include <nn/module/module.h>
#include <nn/module/linear.h>

namespace nn {

class MultiHeadAttention : public Module {
public:
  MultiHeadAttention(int embed_dim, int num_heads, Pcg32& rng);

  Tensor forward(const Tensor& Q, const Tensor& K, const Tensor& V);

private:
  // Runs attention separately within each of num_heads_ slices of size dk_.
  Tensor scaled_dot_product_attention(const Tensor& Q,
                                      const Tensor& K,
                                      const Tensor& V);
  // [b, t, embed_dim] <-> [b, num_heads, t, dk_], splitting/merging the last
  // axis into per-head chunks so attention can batch over heads.
  Tensor split_heads(const Tensor& x);
  Tensor combine_heads(const Tensor& x);

  Linear wQ_, wK_, wV_, wO_;
  int embed_dim_, num_heads_, dk_;
};

}