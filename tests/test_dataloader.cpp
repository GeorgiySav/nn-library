#include "test_harness.h"

#include <nn/data/dataloader.h>

static std::shared_ptr<nn::data::TensorDataset<>> counting_dataset(int n, int d) {
  nn::Tensor x = nn::Tensor::zeros({n, d});
  nn::Tensor y(nn::Shape{n}, nn::Device::CPU, nn::DType::I32);
  for (int i{0}; i < n; ++i) {
    x.data()[size_t(i) * d] = float(i);
    y.data_i32()[i] = i % 10;
  }
  return std::make_shared<nn::data::TensorDataset<>>(std::move(x), std::move(y));
}

NN_TEST(dataloader_visits_every_row_once_and_keeps_labels_aligned) {
  nn::Pcg32 rng(1);
  nn::data::DataLoader<> loader(counting_dataset(100, 4), 10, rng);

  std::vector<int> seen(100, 0);
  int batches = 0;
  loader.reset();
  while (loader.has_next()) {
    auto [xb, yb] = loader.next();
    for (int i{0}; i < xb.shape().dim(0); ++i) {
      const int row = int(xb.data()[size_t(i) * 4]);
      ++seen[size_t(row)];
      NN_CHECK(yb.data_i32()[i] == row % 10);
    }
    ++batches;
  }
  NN_CHECK(batches == 10);
  for (int c : seen) NN_CHECK(c == 1);
}

NN_TEST(dataloader_shuffle_actually_permutes) {
  nn::Pcg32 rng(1);
  nn::data::DataLoader<> loader(counting_dataset(64, 1), 64, rng, true);
  auto [xb, yb] = loader.next();

  bool identity = true;
  for (int i = 0; i < 64; ++i) {
    if (xb.data()[i] != float(i)) { identity = false; break; }
  }
  NN_CHECK(!identity);
}

NN_TEST(dataloader_drop_last_discards_the_tail) {
  nn::Pcg32 rng(1);
  nn::data::DataLoader<> drops(counting_dataset(100, 1), 32, rng, false, true);
  NN_CHECK(drops.batches_per_epoch() == 3);

  nn::data::DataLoader<> keeps(counting_dataset(100, 1), 32, rng, false, false);
  NN_CHECK(keeps.batches_per_epoch() == 4);

  int last = 0;
  while (keeps.has_next()) last = keeps.next().x.shape().dim(0);
  NN_CHECK(last == 4);   // the short tail gets its own tensor
}