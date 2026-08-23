#include <nn/core/tensor.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <ostream>
#include <string>

namespace nn {

namespace {

struct Style {
  bool is_int = false;
  bool sci    = false;
  int  width  = 1;
};

std::string fmt_f32(float v, bool sci) {
  if (std::isnan(v)) return "nan";
  if (std::isinf(v)) return v > 0.0f ? "inf" : "-inf";
  char buf[32];
  std::snprintf(buf, sizeof buf, sci ? "%.4e" : "%.4f", double(v));
  return buf;
}

std::string fmt_i32(int32_t v) {
  char buf[16];
  std::snprintf(buf, sizeof buf, "%d", v);
  return buf;
}

std::string element(const Tensor& t, int64_t i, const Style& st) {
  return st.is_int ? fmt_i32(t.host_data_i32()[i])
                   : fmt_f32(t.host_data()[i], st.sci);
}

struct Scan {
  float lo = 0.0f, hi = 0.0f;   // most negative, largest magnitude
  float max_abs = 0.0f;
  int64_t nans = 0, infs = 0;
};

Scan scan(const Tensor& t, bool is_int) {
  Scan s;
  const int64_t n = t.numel();
  if (is_int) {
    const int32_t* p = t.host_data_i32();
    for (int64_t i = 0; i < n; ++i) {
      s.lo = std::min(s.lo, float(p[i]));
      s.hi = std::max(s.hi, float(p[i]));
    }
    return s;
  }
  const float* p = t.host_data();
  for (int64_t i = 0; i < n; ++i) {
    const float v = p[i];
    if (std::isnan(v)) { ++s.nans; continue; }
    if (std::isinf(v)) { ++s.infs; continue; }
    s.lo = std::min(s.lo, v);
    s.hi = std::max(s.hi, v);
    s.max_abs = std::max(s.max_abs, std::fabs(v));
  }
  return s;
}

Style style_for(const Tensor& t, bool is_int) {
  Style st;
  st.is_int = is_int;
  const Scan sc = scan(t, is_int);

  if (is_int) {
    st.width = int(std::max(fmt_i32(int32_t(sc.lo)).size(),
                            fmt_i32(int32_t(sc.hi)).size()));
    return st;
  }

  st.sci = sc.max_abs >= 1e5f || (sc.max_abs > 0.0f && sc.max_abs < 1e-3f);
  st.width = int(std::max(fmt_f32(sc.lo, st.sci).size(),
                          fmt_f32(sc.hi, st.sci).size()));
  if (sc.nans) st.width = std::max(st.width, 3);
  if (sc.infs) st.width = std::max(st.width, 4);
  return st;
}

void pad_to(std::string& out, const std::string& s, int width) {
  for (int i = int(s.size()); i < width; ++i) out += ' ';
  out += s;
}

void emit(std::string& out, const Tensor& t, const Style& st,
          const int64_t* cstride, int axis, int64_t base, int edge) {
  const Shape& s = t.shape();
  if (axis == s.rank()) {
    pad_to(out, element(t, base, st), st.width);
    return;
  }

  const int64_t n = s.dim(axis);
  const bool elide = n > 2 * int64_t(edge);
  const int breaks = s.rank() - 1 - axis;

  auto separator = [&] {
    out += ',';
    for (int i = 0; i < breaks; ++i) out += '\n';
    if (breaks == 0) out += ' ';
    else for (int i = 0; i <= axis; ++i) out += ' ';
  };

  out += '[';
  if (!elide) {
    for (int64_t k = 0; k < n; ++k) {
      if (k) separator();
      emit(out, t, st, cstride, axis + 1, base + k * cstride[axis], edge);
    }
  } else {
    for (int64_t k = 0; k < edge; ++k) {
      if (k) separator();
      emit(out, t, st, cstride, axis + 1, base + k * cstride[axis], edge);
    }
    separator();
    if (breaks == 0) pad_to(out, "...", st.width);
    else             out += "...";
    for (int64_t k = n - edge; k < n; ++k) {
      separator();
      emit(out, t, st, cstride, axis + 1, base + k * cstride[axis], edge);
    }
  }
  out += ']';
}

}  // namespace

std::string to_string(const Tensor& t, int edge) {
  if (!t.defined()) return "Tensor(undefined)";
  edge = std::max(edge, 1);

  std::string out = "Tensor" + t.shape().str() + " " +
                    dtype_name(t.dtype()) + " " + device_name(t.device());

  if (!t.is_contiguous() || t.offset() != 0) {
    out += " strides=(";
    for (int i = 0; i < t.strides().rank(); ++i) {
      if (i) out += ", ";
      out += std::to_string(t.stride(i));
    }
    out += ")";
    if (t.offset() != 0) out += " offset=" + std::to_string(t.offset());
    if (!t.is_contiguous()) out += " noncontiguous";
  }

  if (t.numel() == 0) return out + "  (empty)";

  const Tensor h = t.contiguous().to(Device::CPU);
  const bool is_int = (t.dtype() == DType::I32);

  const Scan sc = scan(h, is_int);
  if (sc.nans) out += "  " + std::to_string(sc.nans) + " nan";
  if (sc.infs) out += "  " + std::to_string(sc.infs) + " inf";

  const Style st = style_for(h, is_int);

  const Shape& s = h.shape();
  int64_t cstride[kMaxShapeRank] = {0};
  int64_t acc = 1;
  for (int i = s.rank() - 1; i >= 0; --i) { cstride[i] = acc; acc *= s.dim(i); }

  constexpr int64_t kMaxPrinted = 4096;
  for (;;) {
    int64_t printed = 1;
    for (int i = 0; i < s.rank(); ++i) {
      printed *= std::min<int64_t>(s.dim(i), 2 * edge + 1);
      if (printed > kMaxPrinted) break;
    }
    if (printed <= kMaxPrinted || edge == 1) break;
    --edge;
  }

  out += '\n';
  emit(out, h, st, cstride, 0, 0, edge);
  return out;
}

std::string Tensor::str() const { return to_string(*this); }

std::ostream& operator<<(std::ostream& os, const Tensor& t) {
  return os << to_string(t);
}

}  // namespace nn
