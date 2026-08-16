#include "bench.h"

#include <nn/core/rng.h>
#include <nn/core/tensor.h>
#include <nn/ops/ops.h>
#include <nn/autograd/tape.h>

int main() {
  constexpr int kNodes = 4096;
  
  nn::autograd::Tape tape;
  nn::Tensor param = nn::Tensor::scalar(0.0f);
  param.set_requires_grad(true);

  const double ns = nn::bench::time_ns([&] {
    nn::autograd::TapeScope scope(tape);
    int prev = tape.node_for(param);
    for (int i{0}; i < kNodes; ++i) {
      prev = tape.record([](const nn::Tensor&, std::span<nn::Tensor>){}, {prev}, "nop");
    }
    nn::Tensor loss = nn::Tensor::scalar(0.0f);
    tape.set_producer(loss, prev);
    tape.backward(loss);
  });

  nn::bench::report("tape record+backward", ns, "node", kNodes);
  std::printf("arena reserved: %zu bytes\n", tape.arena_size());
}