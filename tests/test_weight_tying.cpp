// Two modules sharing one parameter -- Tensor's storage/AutogradMeta are
// shared_ptrs, so `b.weight() = a.weight()` already aliases correctly; what
// these tests actually exercise is everything downstream that has to agree
// it's one parameter: the optimiser (collect_parameters dedup), a device
// move (Module::to's retie), and a Linear reading a tied weight transposed.

#include "test_harness.h"
#include "devices.h"
#include "gradcheck.h"

#include <string>
#include <vector>

#include <nn/autograd/functions.h>
#include <nn/autograd/tape.h>
#include <nn/core/rng.h>
#include <nn/core/tensor.h>
#include <nn/module/embedding.h>
#include <nn/module/linear.h>
#include <nn/module/module.h>
#include <nn/ops/ops.h>
#include <nn/optim/optim.h>

namespace {

std::vector<float> host_of(const nn::Tensor& t) {
  const nn::Tensor h = t.pack().to(nn::Device::CPU);
  return std::vector<float>(h.host_data(), h.host_data() + h.numel());
}

// A minimal composite so a device move can be exercised the way tying is
// meant to be used: one shared owner moved once, not each tied module moved
// on its own (see Module::to's doc comment for why that distinction matters).
struct TiedPair : nn::Module {
  nn::Linear a, b;
  TiedPair(int in, int out, nn::Pcg32& rng) : a(in, out, rng), b(in, out, rng) {
    b.weight() = a.weight();
    b.bias()   = a.bias();
  }

  void collect_named(const std::string& prefix, std::vector<nn::NamedTensor>& out) override {
    a.collect_named(prefix + "a.", out);
    b.collect_named(prefix + "b.", out);
  }
};

}  // namespace

NN_TEST(tying_aliases_storage_and_autograd_meta) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    nn::Pcg32 rng(31);
    nn::Linear a(3, 2, rng);
    nn::Linear b(3, 2, rng);
    a.to(dev); b.to(dev);

    b.weight() = a.weight();
    NN_CHECK(a.weight().meta() == b.weight().meta());

    nn::ops::fill_inplace(a.weight(), 2.5f);
    for (float v : host_of(b.weight())) NN_CHECK_CLOSE(v, 2.5f, 0.0f);
  }
}

NN_TEST(collect_parameters_counts_a_tied_weight_once) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    nn::Pcg32 rng(31);
    nn::Linear a(3, 2, rng);
    nn::Linear b(3, 2, rng);
    a.to(dev); b.to(dev);

    std::vector<nn::Tensor*> untied;
    a.collect_parameters(untied);
    b.collect_parameters(untied);
    NN_CHECK(untied.size() == 4);   // a.w, a.b, b.w, b.b: nothing shared yet

    b.weight() = a.weight();
    b.bias()   = a.bias();

    std::vector<nn::Tensor*> tied;
    a.collect_parameters(tied);
    b.collect_parameters(tied);
    NN_CHECK(tied.size() == 2);     // the shared w and the shared b, once each
  }
}

// The real correctness question: does the optimiser apply exactly one update
// to a tied weight, driven by the sum of both modules' gradients? Checked by
// comparison against a model with no tying machinery at all -- one Linear
// used twice in its own loss, which is mathematically the same situation.
NN_TEST(tied_weights_train_identically_to_one_linear_used_twice) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    nn::Pcg32 ref_rng(31);
    nn::Linear ref(3, 2, ref_rng);
    ref.to(dev);

    nn::Pcg32 rng(97);
    nn::Linear a(3, 2, rng);
    nn::Linear b(3, 2, rng);
    a.to(dev); b.to(dev);

    // Start `a` from ref's values (discarding its own random init) so the two
    // scenarios begin identically, then tie `b` to it.
    a.weight() = ref.weight().clone();
    a.bias()   = ref.bias().clone();
    a.weight().set_requires_grad(true);
    a.bias().set_requires_grad(true);
    b.weight() = a.weight();
    b.bias()   = a.bias();

    std::vector<nn::Tensor*> params;
    a.collect_parameters(params);
    b.collect_parameters(params);
    NN_CHECK(params.size() == 2);

    std::vector<nn::Tensor*> ref_params;
    ref.collect_parameters(ref_params);

    nn::optim::Adam opt(params, 0.1f);
    nn::optim::Adam ref_opt(ref_params, 0.1f);

    const nn::Tensor x = nn::Tensor::from({{1.0f, 0.5f, -0.5f}}, dev);

    for (int step = 0; step < 3; ++step) {
      opt.zero_grad();
      {
        nn::autograd::GradScope grad;
        nn::Tensor loss = nn::autograd::sum_all(a.forward(x)) +
                          nn::autograd::sum_all(b.forward(x));
        loss.backward();
      }
      opt.step();

      ref_opt.zero_grad();
      {
        nn::autograd::GradScope grad;
        nn::Tensor ref_loss = nn::autograd::sum_all(ref.forward(x)) +
                              nn::autograd::sum_all(ref.forward(x));
        ref_loss.backward();
      }
      ref_opt.step();

      const std::vector<float> got_w = host_of(a.weight()), want_w = host_of(ref.weight());
      for (size_t i = 0; i < got_w.size(); ++i) NN_CHECK_CLOSE(got_w[i], want_w[i], 1e-5f);
      const std::vector<float> got_b = host_of(a.bias()), want_b = host_of(ref.bias());
      for (size_t i = 0; i < got_b.size(); ++i) NN_CHECK_CLOSE(got_b[i], want_b[i], 1e-5f);
    }
  }
}

NN_TEST(to_device_preserves_a_tie_when_moved_through_one_shared_owner) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    nn::Pcg32 rng(51);
    TiedPair m(3, 2, rng);
    NN_CHECK(m.a.weight().meta() == m.b.weight().meta());

    m.to(dev);
    NN_CHECK(m.a.weight().device() == dev);
    NN_CHECK(m.a.weight().meta() == m.b.weight().meta());

    nn::ops::fill_inplace(m.a.weight(), 3.5f);
    for (float v : host_of(m.b.weight())) NN_CHECK_CLOSE(v, 3.5f, 0.0f);
  }
}

// The classic case: an LM head reading an Embedding's [vocab, dim] matrix as
// its own [dim, vocab] weight, without ever materialising the transpose.
NN_TEST(linear_tied_to_an_embeddings_weight_reads_it_transposed) {
  const int V = 5, D = 3;

  NN_TEST_FOR_EACH_DEVICE(dev) {
    nn::Pcg32 e_rng(41);
    nn::Embedding embed(V, D, e_rng);
    nn::Pcg32 h_rng(7);
    nn::Linear head(D, V, h_rng);
    embed.to(dev); head.to(dev);

    head.weight() = embed.weight();
    head.set_transposed_weight(true);

    const nn::Tensor idx = nn::Tensor::from_i32({0, 2}, dev);
    const nn::Tensor x = embed.forward(idx);   // [2, D]

    const nn::Tensor got = head.forward(x);    // [2, V]
    const nn::Tensor want = nn::autograd::matmul(x, embed.weight().t().contiguous());

    const std::vector<float> a = host_of(got), b = host_of(want);
    NN_CHECK(a.size() == b.size());
    for (size_t i = 0; i < a.size(); ++i) NN_CHECK_CLOSE(a[i], b[i], 1e-5f);
  }
}

// Gradient from both usages -- the embedding's gather and the head's
// transposed GEMM -- has to land in the one shared weight, correctly summed.
NN_TEST(gradcheck_weight_tied_between_an_embedding_and_its_head) {
  const int V = 5, D = 3;

  NN_TEST_FOR_EACH_DEVICE(dev) {
    nn::Pcg32 e_rng(61);
    nn::Embedding embed(V, D, e_rng);
    nn::Pcg32 h_rng(13);
    nn::Linear head(D, V, h_rng);
    embed.to(dev); head.to(dev);

    head.weight() = embed.weight();
    head.set_transposed_weight(true);

    const nn::Tensor idx = nn::Tensor::from_i32({0, 2, 4}, dev);

    nn::autograd::Tape tape;
    nn::Tensor loss;
    auto forward = [&]() -> float {
      tape.clear();
      nn::autograd::TapeScope scope(tape);
      const nn::Tensor x = embed.forward(idx);
      loss = nn::autograd::sum_all(head.forward(x));
      return loss.item();
    };
    auto backward = [&]() { tape.backward(loss, true); };

    NN_CHECK(nn::test::gradCheck(embed.weight(), forward, backward) < 3e-2f);
  }
}
