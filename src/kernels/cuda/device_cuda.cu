#include <nn/core/device.h>

#include "cuda_common.h"

namespace nn {

// Must not throw: this is called from a function-local static initialiser in
// the test suite, and on a machine with no NVIDIA driver cudaGetDeviceCount
// returns an error rather than reporting zero devices. Either way the answer
// the caller wants is "no CUDA devices".
int cuda_device_count() {
  int n = 0;
  if (cudaGetDeviceCount(&n) != cudaSuccess) return 0;
  return n;
}

}  // namespace nn
