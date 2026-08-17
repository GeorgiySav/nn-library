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

struct Stream {
  Device device = Device::CPU;
  void*  handle = nullptr; // cudaStream_t; null == device's default

  // Blocks until work is finished
  // No-op on CPU
  void synchronize() const;
};

const Stream& default_stream(Device d);
// stream ops will submit to right now, on this thread
const Stream& current_stream(Device d);

// redirects current_stream(s.device)
class StreamScope {
public:
  explicit StreamScope(const Stream& s);
  ~StreamScope();

  StreamScope(const StreamScope&) = delete;
  StreamScope& operator=(const StreamScope&) = delete;
private:
  Device        device_;
  const Stream* prev_;
};

}