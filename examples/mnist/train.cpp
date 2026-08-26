#include <cstdio>
#include <string>

#ifndef NN_PROJECT_ROOT
#define NN_PROJECT_ROOT "."
#endif

#include <nn/nn.h>
#include <nn/data/mnist.h>

float accuracy(nn::Module& model, nn::data::DataLoader<>& loader) {
  nn::autograd::NoGradScope no_grad;
  int64_t correct = 0, total = 0;
  loader.reset();
  while (loader.has_next()) {
    auto [xb, yb] = loader.next();
    correct += nn::metrics::count_correct(
                model(xb).to(nn::Device::CPU),
                yb.to(nn::Device::CPU));
    total   += yb.extent(0);
  }
  return float(correct) / float(total);
}

int main(int argc, char** argv) {
  std::printf("MNIST EXAMPLE\n");
  nn::Device device =
      nn::cuda_device_count() > 0 ? nn::Device::CUDA : nn::Device::CPU;
  std::printf("Device: %s\n", nn::device_name(device));

  const std::string dir = (argc > 1) ? argv[1] : NN_PROJECT_ROOT "/data/mnist";

  nn::Pcg32 rng(42);

  std::printf("Loading MNIST dataset\n");
  auto train = nn::data::load_mnist(dir + "/train-images.idx3-ubyte",
                                    dir + "/train-labels.idx1-ubyte");
  auto test  = nn::data::load_mnist(dir + "/t10k-images.idx3-ubyte",
                                    dir + "/t10k-labels.idx1-ubyte");

  nn::data::DataLoader<> loader(train, 64, rng, true, true, device);
  nn::data::DataLoader<> eval(test, 512, rng, false, false, device);

  std::printf("Creating model\n");
  nn::Sequential model(
    nn::Linear(784, 128, rng),
    nn::ReLu(),
    nn::Linear(128, 10, rng) 
  );
  model.to(device);

  nn::optim::Adam opt(model.parameters(), 0.001f);

  std::printf("Training\n");
  for (int epoch{0}; epoch < 10; ++epoch) {
    loader.reset();
    double running = 0.0;
    int steps = 0;

    while (loader.has_next()) {
      auto [xb, yb] = loader.next();
      opt.zero_grad();

      nn::autograd::GradScope grad;
      nn::Tensor loss = nn::cross_entropy(model(xb), yb);
      loss.backward();
      opt.step();

      running += loss.item();
      ++steps;
    }

    std::printf("epoch %2d: loss %.4f  test acc %.4f\n",
                epoch, running / steps, accuracy(model, eval));

    nn::io::save_checkpoint(NN_PROJECT_ROOT "/mnist.ckpt", model, opt, epoch + 1);
  }

  nn::Sequential reloaded(
    nn::Linear(784, 128, rng),
    nn::ReLu(),
    nn::Linear(128, 10, rng)
  );
  reloaded.to(device);
  nn::io::load_weights(NN_PROJECT_ROOT "/mnist.ckpt", reloaded);
  std::printf("reloaded:      test acc %.4f\n", accuracy(reloaded, eval));
}