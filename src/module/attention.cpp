#include <nn/module/attention.h>

#include <nn/autograd/functions.h>

namespace nn {

MultiHeadAttention::MultiHeadAttention(int embed_dim,
                                       int num_heads,
                                       Pcg32& rng)
    : embed_dim_(embed_dim),
      num_heads_(num_heads),
      dk_(embed_dim / num_heads),
      wQ_(Linear(embed_dim, embed_dim, rng)),
      wK_(Linear(embed_dim, embed_dim, rng)),
      wV_(Linear(embed_dim, embed_dim, rng)),
      wO_(Linear(embed_dim, embed_dim, rng))
      {
  if (embed_dim_ % num_heads_ != 0) throw std::invalid_argument("MultiHeadAttention: embedding dim must be divisible by the number of heads");
}

Tensor MultiHeadAttention::forward(const Tensor& Q, const Tensor& K, const Tensor& V) {
  // Q, K, V: [b, t, e]. Project to q/k/v, split each into per-head slices,
  // attend within each head independently, then merge the heads back into
  // one embed_dim vector per token before the output projection.
  Tensor sQ = split_heads(wQ_.forward(Q)); // [b, nh, t, e/nh]
  Tensor sK = split_heads(wK_.forward(K));
  Tensor sV = split_heads(wV_.forward(V));

  Tensor attended = scaled_dot_product_attention(sQ, sK, sV);
  Tensor combined = combine_heads(attended); // [b, t, e]
  Tensor output   = wO_.forward(combined);
  return output;
}

Tensor MultiHeadAttention::split_heads(const Tensor& x) {
  // [b, t, e] -> [b, t, nh, e/nh] -> [b, nh, t, e/nh]. The transpose puts the
  // head axis before the sequence axis so attention below can batch over
  // (b, nh) and matmul over the trailing (t, e/nh) pair; contiguous() makes
  // that transposed layout dense since matmul needs a packed inner axis.
  const int r = x.rank();
  if (r+1 > kMaxShapeRank) throw std::invalid_argument("split_heads: rank too large");

  int dims[kMaxShapeRank];
  for (int i = 0; i < r - 1; ++i) dims[i] = x.extent(i);
  dims[r - 1] = num_heads_;
  dims[r]     = dk_;

  return x.reshape(Shape{std::span<const int32_t>(dims, r+1)})
          .transpose(-3, -2)
          .contiguous();
}

Tensor MultiHeadAttention::scaled_dot_product_attention(const Tensor& Q, const Tensor& K, const Tensor& V) {
  // No mask here: this module is plain bidirectional self/cross-attention.
  return nn::scaled_dot_product_attention(Q, K, V, Tensor());
}

Tensor MultiHeadAttention::combine_heads(const Tensor& x) {
  // Inverse of split_heads: [b, nh, t, e/nh] -> [b, t, nh, e/nh] -> [b, t, e].
  const int r = x.rank();
  Tensor t = x.transpose(-3, -2);

  int dims[kMaxShapeRank];
  for (int i = 0; i < r - 3; ++i) dims[i] = x.extent(i);
  dims[r-3] = x.extent(-2);
  dims[r-2] = embed_dim_;

  return t.reshape(Shape(std::span<const int>(dims, r-1)));
}

}