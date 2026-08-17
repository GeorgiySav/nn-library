#pragma once

#include <nn/ops/ops.h>


namespace nn::metrics {

inline int64_t count_correct(const Tensor& logits, const Tensor& labels) {
  Tensor pred = ops::argmax_rows(logits);
  const int32_t* p = pred.host_data_i32();
  const int32_t* y = labels.host_data_i32();
  int64_t n = 0;
  for (int i = 0; i < pred.shape().dim(0); ++i) n += (p[i] == y[i]);
  return n;
}

}