#pragma once

namespace nn {

enum class Device { CPU, CUDA };
inline const char* device_name(Device d) {
  switch (d) {
    case Device::CPU:
      return "CPU";
    case Device::CUDA:
      return "CUDA";
    default:
      return "Unknown";
  }
}

}