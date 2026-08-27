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
  Tensor scaled_dot_product_attention(const Tensor& Q,
                                      const Tensor& K,
                                      const Tensor& V); 
  Tensor split_heads(const Tensor& x);
  Tensor combine_heads(const Tensor& x);

  Linear wQ_, wK_, wV_, wO_;
  int embed_dim_, num_heads_, dk_;
};

}