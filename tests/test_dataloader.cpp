#include "test_harness.h"

#include <nn/data/dataloader.h>

static std::shared_ptr<nn::data::TensorDataset> counting_dataset(int n, int d) {
  nn::Tensor x = nn::Tensor::zeros({n, d});
  nn::Tensor y(nn::Shape{n}, nn::Device::CPU, nn::DType::I32);
  for (int i{0}; i < n; ++i) {
    x.host_data()[size_t(i) * d] = float(i);
    y.host_data_i32()[i] = i % 10;
  }
  return std::make_shared<nn::data::TensorDataset>(std::move(x), std::move(y));
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
      const int row = int(xb.host_data()[size_t(i) * 4]);
      ++seen[size_t(row)];
      NN_CHECK(yb.host_data_i32()[i] == row % 10);
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
    if (xb.host_data()[i] != float(i)) { identity = false; break; }
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
  while (keeps.has_next()) last = keeps.next()[0].extent(0);
  NN_CHECK(last == 4);   // the short tail gets its own tensor
}

NN_TEST(dataloader_batches_do_not_alias_each_other) {
  nn::Pcg32 rng(1);
  nn::data::DataLoader<> loader(counting_dataset(8, 1), 2, rng, false);

  auto [x1, y1] = loader.next();
  const float first = x1.host_data()[0];

  auto [x2, y2] = loader.next();
  NN_CHECK(x1.host_data() != x2.host_data());
  NN_CHECK(x1.host_data()[0] == first);
  NN_CHECK(x2.host_data()[0] != first);
}

NN_TEST(dataset_carries_integer_inputs) {
  const int N = 8, T = 4;
  nn::Tensor ids(nn::Shape{N, T}, nn::Device::CPU, nn::DType::I32);
  nn::Tensor nxt(nn::Shape{N, T}, nn::Device::CPU, nn::DType::I32);
  for (int i{0}; i < N * T; ++i) {
    ids.host_data_i32()[i] = i;
    nxt.host_data_i32()[i] = i + 1;
  }

  nn::Pcg32 rng(7);
  auto ds = std::make_shared<nn::data::TensorDataset>(std::move(ids), std::move(nxt));
  nn::data::DataLoader<> loader(ds, 4, rng, false);

  auto [x, y] = loader.next();
  NN_CHECK(x.dtype() == nn::DType::I32);
  NN_CHECK(y.dtype() == nn::DType::I32);
  NN_CHECK(x.shape() == nn::Shape({4, T}));
  for (int i{0}; i < 4 * T; ++i) {   // the whole batch, not just its first row
    NN_CHECK(y.host_data_i32()[i] == x.host_data_i32()[i] + 1);
  }
}

NN_TEST(dataset_carries_multi_axis_samples) {
  const int N = 6;
  nn::Tensor imgs = nn::Tensor::zeros({N, 2, 3});
  nn::Tensor lbls(nn::Shape{N}, nn::Device::CPU, nn::DType::I32);
  for (int i = 0; i < N; ++i) {
    imgs.host_data()[size_t(i) * 6] = float(i);
    lbls.host_data_i32()[i] = i;
  }

  nn::Pcg32 rng(3);
  auto ds = std::make_shared<nn::data::TensorDataset>(std::move(imgs), std::move(lbls));
  nn::data::DataLoader<> loader(ds, 3, rng, false);

  auto [xb, yb] = loader.next();
  NN_CHECK(xb.shape() == nn::Shape({3, 2, 3}));
  NN_CHECK(yb.shape() == nn::Shape({3}));
  for (int i = 0; i < 3; ++i) {
    NN_CHECK(xb.host_data()[size_t(i) * 6] == float(yb.host_data_i32()[i]));
  }
}