#pragma once

#include <cstddef>
#include <cstdint>

namespace nn {

enum class DType { F32, I32, BF16 };

constexpr size_t dtype_size(DType dtype) {
  switch (dtype) {
    case DType::F32:
      return sizeof(float);
    case DType::I32:
      return sizeof(int32_t);
    case DType::BF16:
      return 2;
    default:
      return 0;
  }
}

constexpr const char* dtype_name(DType dtype) {
  switch (dtype) {
    case DType::F32:
      return "float32";
    case DType::I32:
      return "int32";
    case DType::BF16:
      return "bfloat16";
    default:
      return "unknown";
  }
}

}