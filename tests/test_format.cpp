#include "test_harness.h"
#include "devices.h"

#include <string>
#include <vector>

#include <nn/core/rng.h>
#include <nn/core/tensor.h>

namespace {

std::string body(const nn::Tensor& t) {
  const std::string s = nn::to_string(t);
  const size_t nl = s.find('\n');
  return nl == std::string::npos ? std::string{} : s.substr(nl + 1);
}

std::string header(const nn::Tensor& t) {
  const std::string s = nn::to_string(t);
  return s.substr(0, s.find('\n'));
}

bool has(const std::string& hay, const char* needle) {
  return hay.find(needle) != std::string::npos;
}

}  // namespace

NN_TEST(format_lays_out_a_small_matrix) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    const nn::Tensor t =
        nn::Tensor::from({{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}}, dev);

    NN_CHECK(body(t) == "[[1.0000, 2.0000, 3.0000],\n"
                        " [4.0000, 5.0000, 6.0000]]");

    const std::string h = header(t);
    NN_CHECK(has(h, "[2, 3]"));
    NN_CHECK(has(h, "float32"));
    NN_CHECK(has(h, nn::device_name(dev)));
  }
}

NN_TEST(format_aligns_columns) {
  const nn::Tensor t =
      nn::Tensor::from({{1.0f, -2.0f}, {30.0f, 4.0f}}, nn::Device::CPU);
  NN_CHECK(body(t) == "[[ 1.0000, -2.0000],\n"
                      " [30.0000,  4.0000]]");
}

NN_TEST(format_reports_layout_only_when_it_is_surprising) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    const nn::Tensor t = nn::Tensor::zeros({2, 3, 4}, dev);
    NN_CHECK(!has(header(t), "strides"));
    NN_CHECK(!has(header(t), "noncontiguous"));

    const nn::Tensor v = t.transpose(1, 2);
    NN_CHECK(has(header(v), "strides=(12, 1, 4)"));
    NN_CHECK(has(header(v), "noncontiguous"));

    const nn::Tensor s = t.slice(2, 1, 2);
    NN_CHECK(has(header(s), "offset=1"));
  }
}

NN_TEST(format_follows_strides_not_storage) {
  NN_TEST_FOR_EACH_DEVICE(dev) {
    std::vector<float> h(24);
    for (size_t i = 0; i < h.size(); ++i) h[i] = float(i);
    const nn::Tensor t = nn::Tensor::from(h, nn::Shape({2, 3, 4}), dev);

    const int order[3] = {2, 0, 1};
    const nn::Tensor v = t.permute(std::span<const int>(order, 3));
    NN_CHECK(body(v) == body(v.contiguous()));
    NN_CHECK(body(t.transpose(1, 2)) == body(t.transpose(1, 2).contiguous()));
    NN_CHECK(body(t.slice(1, 1, 2)) == body(t.slice(1, 1, 2).contiguous()));
  }
}

NN_TEST(format_elides_long_axes) {
  const nn::Tensor t =
      nn::Tensor::from(std::vector<float>(100, 1.25f), nn::Shape({100}));
  const std::string b = body(t);
  NN_CHECK(has(b, "..."));
  NN_CHECK(b.size() < 100);              // not 100 numbers

  // A rank-4 tensor cannot be allowed to print (2*edge+1)^4 numbers unbounded.
  const nn::Tensor big = nn::Tensor::zeros({40, 40, 40, 40});
  NN_CHECK(nn::to_string(big).size() < 64 * 1024);
}

NN_TEST(format_counts_nan_and_inf) {
  const float nan = 0.0f / 0.0f, inf = 1.0f / 0.0f;
  const nn::Tensor t =
      nn::Tensor::from({nan, 1.0f, inf, nan}, nn::Device::CPU);

  const std::string h = header(t);
  NN_CHECK(has(h, "2 nan"));
  NN_CHECK(has(h, "1 inf"));
  NN_CHECK(has(body(t), "nan"));
}

NN_TEST(format_switches_to_scientific_for_extreme_magnitudes) {
  NN_CHECK(has(body(nn::Tensor::full({2}, 1.0e6f)), "e+06"));
  NN_CHECK(has(body(nn::Tensor::full({2}, 1.0e-6f)), "e-06"));
  NN_CHECK(has(body(nn::Tensor::full({2}, 12.5f)), "12.5000"));
}

NN_TEST(format_edge_shapes) {
  NN_CHECK(nn::to_string(nn::Tensor{}) == "Tensor(undefined)");
  NN_CHECK(has(nn::to_string(nn::Tensor::zeros({0, 3})), "(empty)"));
  NN_CHECK(body(nn::Tensor::scalar(3.5f)) == "3.5000");
  NN_CHECK(body(nn::Tensor::from_i32({3, 14, 159, -2})) == "[  3,  14, 159,  -2]");
  NN_CHECK(has(header(nn::Tensor::from_i32({1})), "int32"));
}
