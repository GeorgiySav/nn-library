#include <cstdio>
#include <string>

#ifndef NN_PROJECT_ROOT
#define NN_PROJECT_ROOT "."
#endif

#include <nn/core/rng.h>

#include <nn/nn/module.h>
#include <nn/optim/sgd.h>
#include <nn/autograd/tape.h>
#include <nn/metrics.h>

#include <nn/data/mnist.h>
#include <nn/data/dataloader.h>

float accuracy(nn::Module& model, nn::data::DataLoader<>& loader) {
  nn::autograd::NoGradScope no_grad;
  int64_t correct = 0, total = 0;
  loader.reset();
  while (loader.has_next()) {
    auto [xb, yb] = loader.next();
    correct += nn::metrics::count_correct(model.forward(xb), yb);
    total   += yb.shape().dim(0);
  }
  return float(correct) / float(total);
}

int main(int argc, char** argv) {
  std::printf("MNIST EXAMPLE\n");

  const std::string dir = (argc > 1) ? argv[1] : NN_PROJECT_ROOT "/data/mnist";

  nn::Pcg32 rng(42);

  std::printf("Loading MNIST dataset\n");
  auto train = nn::data::load_mnist(dir + "/train-images.idx3-ubyte",
                                    dir + "/train-labels.idx1-ubyte");
  auto test  = nn::data::load_mnist(dir + "/t10k-images.idx3-ubyte",
                                    dir + "/t10k-labels.idx1-ubyte");

  nn::data::DataLoader<> loader(train, 64, rng);
  nn::data::DataLoader<> eval(test, 512, rng, false, false);

  std::printf("Creating model\n");
  nn::Sequential model(
    nn::Linear(784, 128, rng),
    nn::ReLu(),
    nn::Linear(128, 10, rng) 
  );

  nn::optim::SGD opt(model.parameters(), 0.1f);
  nn::autograd::Tape tape;

  std::printf("Training\n");
  for (int epoch{0}; epoch < 10; ++epoch) {
    loader.reset();
    double running = 0.0;
    int steps = 0;

    while (loader.has_next()) {
      auto [xb, yb] = loader.next();
      opt.zero_grad();

      nn::Tensor loss;
      {
        nn::autograd::TapeScope scope(tape);
        loss = nn::autograd::cross_entropy(model.forward(xb), yb);
      }

      tape.backward(loss);
      opt.step();

      running += loss.item();
      ++steps;
    }

    std::printf("epoch %2d: loss %.4f  test acc %.4f\n",
                epoch, running / steps, accuracy(model, eval));
  }

}