#pragma once

#include <cstdint>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <nn/data/dataset.h>

namespace nn::data {

namespace detail {

// MNIST's IDX format stores every header field as big-endian u32, unlike the
// rest of this library which is little-endian throughout
inline uint32_t read_be32(std::istream& in) {
  unsigned char b[4];
  in.read(reinterpret_cast<char*>(b), 4);
  if (!in) throw std::runtime_error("mnist: unexpected end of file");
  return (uint32_t(b[0]) << 24) | (uint32_t(b[1]) << 16) |
         (uint32_t(b[2]) <<  8) |  uint32_t(b[3]);
}

inline std::ifstream open_or_throw(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("mnist: cannot open " + path);
  return in;
}

} // detail

inline Tensor load_mnist_images(const std::string& path) {
  auto in = detail::open_or_throw(path);
  if (detail::read_be32(in) != 0x00000803u) {  // IDX magic for ubyte images
    throw std::runtime_error("mnist: bad image magic in " + path);
  }
  const uint32_t n    = detail::read_be32(in);
  const uint32_t rows = detail::read_be32(in);
  const uint32_t cols = detail::read_be32(in);
  const uint32_t d    = rows * cols;

  std::vector<uint8_t> raw(size_t(n) * d);
  in.read(reinterpret_cast<char*>(raw.data()), std::streamsize(raw.size()));
  if (!in) throw std::runtime_error("mnist: truncated image data");

  Tensor x(Shape{int(n), int(d)});
  float* out = x.host_data();
  for (size_t i{0}; i < raw.size(); ++i) out[i] = float(raw[i]) * (1.0f / 255.0f);
  return x;
}

inline Tensor load_mnist_labels(const std::string& path) {
  auto in = detail::open_or_throw(path);
  if (detail::read_be32(in) != 0x00000801u) {  // IDX magic for ubyte labels
    throw std::runtime_error("mnist: bad label magic in " + path);
  }
  const uint32_t n = detail::read_be32(in);

  std::vector<uint8_t> raw(n);
  in.read(reinterpret_cast<char*>(raw.data()), std::streamsize(raw.size()));
  if (!in) throw std::runtime_error("mnist: truncated label data");

  Tensor y(Shape{int(n)}, Device::CPU, DType::I32);
  int32_t* out = y.host_data_i32();
  for (uint32_t i = 0; i < n; ++i) out[i] = int32_t(raw[i]);
  return y;
}

inline std::shared_ptr<TensorDataset> load_mnist(const std::string& images,
                                                 const std::string& labels) {
  return std::make_shared<TensorDataset>(load_mnist_images(images),
                                         load_mnist_labels(labels));
}

}