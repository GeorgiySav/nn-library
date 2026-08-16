#include "bench.h"

#include <nn/core/rng.h>
#include <nn/core/tensor.h>
#include <nn/ops/ops.h>

int main() {
  nn::Pcg32 rng(1);
  struct Case {
    int M, N, K;
    bool tA, tB;
    const char* name;
  };

  const Case cases[] = {
    {  64, 128, 784, false, false, "fc1 forward"   },
    {  64, 784, 128, false, true,  "fc1 grad-input"},
    { 784, 128,  64, true,  false, "fc1 grad-w"    },
    { 256, 256, 256, false, false, "square 256"    },
    { 512, 512, 512, false, false, "square 512"    },
  };

  for (const Case& c : cases) {
    nn::Tensor a = c.tA ? nn::Tensor::randn({c.K, c.M}, rng, 1.0f)
                        : nn::Tensor::randn({c.M, c.K}, rng, 1.0f);
    nn::Tensor b = c.tB ? nn::Tensor::randn({c.N, c.K}, rng, 1.0f)
                        : nn::Tensor::randn({c.K, c.N}, rng, 1.0f);

    const double ns = nn::bench::time_ns(
      [&] { nn::ops::matmul(a, b, c.tA, c.tB); });
    nn::bench::report(c.name, ns, 2.0 * c.M * c.N * c.K);
  }
}