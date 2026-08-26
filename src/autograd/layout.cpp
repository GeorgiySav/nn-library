#include <nn/autograd/functions.h>

#include <initializer_list>
#include <stdexcept>
#include <string>
#include <utility>

#include <nn/core/small_vec.h>
#include <nn/ops/ops.h>

#include "autograd_common.h"

namespace nn::autograd {

Tensor contiguous(const Tensor& x) {
  if (x.is_contiguous()) return x;

  Tensor out = x.pack();
  record_op(out, "contiguous",
    [](const Tensor& g, std::span<Tensor> g_in) {
      g_in[0] = g;
    }, x);

  return out;
}

Tensor permute(const Tensor& x, std::span<const int> order) {
  const int r = x.shape().rank();
  if (int(order.size()) != r) {
    throw std::invalid_argument("permute: " + std::to_string(order.size()) +
                                " axes given for rank " + std::to_string(r));
  }

  SmallVec<int, kMaxShapeRank> axes(order.size()), inv(order.size());
  bool seen[kMaxShapeRank] = {false};
  for (int i{0}; i < r; ++i) {
    const int src = x.shape().resolve_dim(order[i], "permute");
    if (seen[src]) {
      throw std::invalid_argument("permute: axis " + std::to_string(src) +
                                  " appears twice in the order");
    }
    seen[src] = true;
    axes[i] = src;
    inv[src] = i;
  }

  Tensor out = x.permute_view(axes.span());

  record_op(out, "permute",
    [inv](const Tensor& g, std::span<Tensor> g_in) mutable {
      g_in[0] = g.permute_view(inv.span());
    }, x);

  return out;
}

Tensor transpose(const Tensor& x, int a, int b) {
  const int r = x.shape().rank();
  const int ia = x.shape().resolve_dim(a, "transpose");
  const int ib = x.shape().resolve_dim(b, "transpose");

  int order[kMaxShapeRank];
  for (int i = 0; i < r; ++i) order[i] = i;
  order[ia] = ib;
  order[ib] = ia;
  return permute(x, std::span<const int>(order, r));
}

Tensor reshape(const Tensor& x, const Shape& shape) {
  if (shape.numel() != x.numel()) {
    throw std::invalid_argument("reshape: " + x.shape().str() + " -> " + shape.str() +
                                " changes the element count");
  }

  Tensor out = x.reshape_view(shape);

  record_op(out, "reshape",
    [sx = x.shape()](const Tensor& g, std::span<Tensor> g_in) {
      g_in[0] = g.is_contiguous() ? g.reshape_view(sx) : g.pack().reshape_view(sx);
    }, x);

  return out;
}

Tensor slice(const Tensor& x, int axis, int64_t start, int64_t len) {
  const int a = x.shape().resolve_dim(axis, "slice");
  Tensor out = x.slice_view(a, start, len);

  record_op(out, "slice",
    [sx = x.shape(), dev = x.device(), dt = x.dtype(), a, start, len]
    (const Tensor& g_out, std::span<Tensor> g_in) {
      // Zero everywhere the slice did not reach, g_out inside the window.
      // The window is a view sharing storage with g, so unpack writes
      // through its strides into g.
      Tensor g = Tensor::zeros(sx, dev, dt);
      Tensor window = g.slice_view(a, start, len);
      ops::unpack(window, g_out);
      g_in[0] = std::move(g);
    }, x);

  return out;
}

Tensor expand(const Tensor& x, const Shape& to) {
  Tensor out = x.expand_view(to);

  record_op(out, "expand",
    [sx = x.shape()](const Tensor& g, std::span<Tensor> g_in) {
      g_in[0] = ops::sum_to(g, sx);
    }, x);

  return out;
}

// cat does not go through record_op either: it has a variable number of
// inputs, and its backward hands each one a different window of the incoming
// gradient, so it builds the input list itself.
Tensor cat(std::span<const Tensor> parts, int dim) {
  if (parts.empty()) throw std::invalid_argument("cat: nothing to concatenate");
  if (int(parts.size()) > kMaxNodeInputs) {
    throw std::invalid_argument("cat: at most " + std::to_string(kMaxNodeInputs) +
                                " tensors at a time (the tape records a node's "
                                "inputs inline)");
  }

  const Shape& first = parts[0].shape();
  const int d = first.resolve_dim(dim, "cat");

  int total = 0;
  for (const Tensor& p : parts) {
    if (p.shape().rank() != first.rank()) {
      throw std::invalid_argument("cat: " + p.shape().str() + " and " + first.str() +
                                  " have different ranks");
    }
    for (int i = 0; i < first.rank(); ++i) {
      if (i != d && p.shape().dim(i) != first.dim(i)) {
        throw std::invalid_argument("cat: " + p.shape().str() + " and " + first.str() +
                                    " differ away from axis " + std::to_string(d));
      }
    }
    if (p.device() != parts[0].device() || p.dtype() != parts[0].dtype()) {
      throw std::invalid_argument("cat: operands differ in device or dtype");
    }
    total += p.shape().dim(d);
  }

  Shape out_shape = first;
  out_shape.set_dim(d, total);
  Tensor out(out_shape, parts[0].device(), parts[0].dtype());

  SmallVec<int64_t, kMaxNodeInputs> starts;
  int64_t offset = 0;
  for (const Tensor& p : parts) {
    const int64_t len = p.shape().dim(d);
    Tensor window = out.slice_view(d, offset, len);
    ops::unpack(window, p);
    starts.push_back(offset);
    offset += len;
  }

  Tape* tape = active_tape();
  if (!tape) return out;

  NodeInputs ids;
  bool any_tracked = false;
  for (const Tensor& p : parts) {
    const int id = tape->node_for(p);
    ids.push_back(id);
    any_tracked = any_tracked || (id >= 0);
  }
  if (!any_tracked) return out;

  SmallVec<int64_t, kMaxNodeInputs> lens;
  for (const Tensor& p : parts) lens.push_back(p.shape().dim(d));

  tape->set_producer(out, tape->record(
      [d, starts, lens](const Tensor& g, std::span<Tensor> g_in) mutable {
        for (int i = 0; i < starts.size(); ++i) {
          g_in[i] = g.slice_view(d, starts[i], lens[i]);
        }
      },
      ids, "cat"));

  return out;
}

Tensor cat(std::initializer_list<Tensor> parts, int dim) {
  return cat(std::span<const Tensor>(parts.begin(), parts.size()), dim);
}

}  // namespace nn::autograd
