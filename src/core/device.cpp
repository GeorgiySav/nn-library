#include <nn/core/device.h>

#include <array>
#include <stdexcept>

#include "cuda_common.h"

namespace nn {

// Must not throw: this is called from a function-local static initialiser in
// the test suite, and on a machine with no NVIDIA driver cudaGetDeviceCount
// reports an error rather than zero devices. Either way the answer the caller
// wants is "no CUDA devices".
int cuda_device_count() {
  int n = 0;
  if (cudaGetDeviceCount(&n) != cudaSuccess) return 0;
  return n;
}

namespace {
 
constexpr int kNumDevices = 2;
int index_of(Device d) { return static_cast<int>(d); }  

thread_local std::array<const Stream*, kNumDevices> t_current{nullptr, nullptr};

}

void Stream::synchronize() const {
  if (device == Device::CPU) return;
  NN_CUDA_CHECK(cudaStreamSynchronize(static_cast<cudaStream_t>(handle)));
}

const Stream& default_stream(Device d) {
  static const Stream cpu  {Device::CPU,  nullptr};
  static const Stream cuda {Device::CUDA, nullptr};
  return d == Device::CPU ? cpu : cuda;
}

const Stream& current_stream(Device d) {
  const Stream* s = t_current[index_of(d)];
  return s ? *s : default_stream(d);
}

StreamScope::StreamScope(const Stream& s)
  : device_(s.device), prev_(t_current[index_of(s.device)]) {
  t_current[index_of(s.device)] = &s;
}

StreamScope::~StreamScope() {
  t_current[index_of(device_)] = prev_;
}

} // namespace nn
