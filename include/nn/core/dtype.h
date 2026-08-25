#pragma once

#include <cstddef>

namespace nn {

enum class DType { F32, I32 };

inline size_t dtype_size(DType dtype) {
  switch (dtype) {
    case DType::F32:
      return sizeof(float);
    case DType::I32:
      return sizeof(int32_t);
    default:
      return 0;
  }
}

inline const char* dtype_name(DType dtype) {
  switch (dtype) {
    case DType::F32:
      return "float32";
    case DType::I32:
      return "int32";
    default:
      return "unknown";
  }
}

}