#include <nn/io/checkpoint.h>

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <unordered_map>

#include <nn/core/allocator.h>
#include <nn/core/rng.h>

namespace nn {

double scalar_value(std::span<const NamedScalar> scalars, const std::string& name) {
  for (const NamedScalar& s : scalars) {
    if (s.name == name) return s.value;
  }
  throw std::runtime_error("checkpoint: no scalar named \"" + name + "\"");
}

}  // namespace nn

namespace nn::io {

namespace {

constexpr char kMagic[4] = {'N', 'N', 'C', 'K'};

int64_t align8(int64_t n) { return (n + 7) & ~int64_t(7); }

// Every write and read goes through these, so a truncated file is caught at the
// point it runs out rather than producing plausible garbage further on.
template <class T>
void put(std::ostream& os, const T& v) {
  static_assert(std::is_trivially_copyable_v<T>);
  os.write(reinterpret_cast<const char*>(&v), sizeof(T));
}

template <class T>
T get(std::istream& is, const char* what) {
  T v{};
  is.read(reinterpret_cast<char*>(&v), sizeof(T));
  if (!is) throw std::runtime_error(std::string("checkpoint: file ends inside ") + what);
  return v;
}

void put_name(std::ostream& os, const std::string& name) {
  if (name.size() > 0xFFFF) {
    throw std::invalid_argument("checkpoint: name is too long: " + name);
  }
  put<uint16_t>(os, uint16_t(name.size()));
  os.write(name.data(), std::streamsize(name.size()));
}

std::string get_name(std::istream& is) {
  const uint16_t len = get<uint16_t>(is, "a name length");
  std::string name(len, '\0');
  is.read(name.data(), len);
  if (!is) throw std::runtime_error("checkpoint: file ends inside a name");
  return name;
}

int64_t header_bytes(std::span<const NamedTensor> tensors,
                     std::span<const NamedScalar> scalars) {
  int64_t n = 4 + 4 + 4 + 4;                       // magic, version, two counts
  for (const NamedTensor& e : tensors) {
    n += 2 + int64_t(e.name.size()) + 1 + 1;       // name, dtype, rank
    n += 4 * int64_t(e.t->shape().rank());         // dims
    n += 8 + 8;                                    // nbytes, offset
  }
  for (const NamedScalar& s : scalars) {
    n += 2 + int64_t(s.name.size()) + 8;
  }
  return n;
}

struct Entry {
  DType dtype = DType::F32;
  Shape shape;
  uint64_t nbytes = 0;
  uint64_t offset = 0;
};

std::vector<std::pair<std::string, Entry>> read_header(std::istream& is,
                                                       const std::string& path) {
  char magic[4] = {};
  is.read(magic, 4);
  if (!is || std::memcmp(magic, kMagic, 4) != 0) {
    throw std::runtime_error("checkpoint: " + path + " is not a checkpoint file");
  }
  const uint32_t version = get<uint32_t>(is, "the version");
  if (version != kCheckpointVersion) {
    throw std::runtime_error("checkpoint: " + path + " is version " +
                             std::to_string(version) + ", this build reads " +
                             std::to_string(kCheckpointVersion));
  }

  const uint32_t n_tensors = get<uint32_t>(is, "the tensor count");
  const uint32_t n_scalars = get<uint32_t>(is, "the scalar count");

  std::vector<std::pair<std::string, Entry>> out;
  out.reserve(n_tensors);
  for (uint32_t i = 0; i < n_tensors; ++i) {
    std::string name = get_name(is);
    Entry e;
    e.dtype = DType(get<uint8_t>(is, "a dtype"));
    const int rank = get<uint8_t>(is, "a rank");
    if (rank > kMaxShapeRank) {
      throw std::runtime_error("checkpoint: \"" + name + "\" has rank " +
                               std::to_string(rank) + ", past the maximum");
    }
    int dims[kMaxShapeRank] = {};
    for (int a = 0; a < rank; ++a) dims[a] = get<int32_t>(is, "a shape");
    e.shape = Shape(std::span<const int>(dims, rank));
    e.nbytes = get<uint64_t>(is, "a byte count");
    e.offset = get<uint64_t>(is, "an offset");
    out.emplace_back(std::move(name), e);
  }

  for (uint32_t i = 0; i < n_scalars; ++i) {
    std::string name = get_name(is);
    const double v = get<double>(is, "a scalar");
    out.emplace_back(std::move(name), Entry{DType::F32, Shape{}, 0, 0});
    out.back().second.nbytes = 0;
    std::memcpy(&out.back().second.offset, &v, sizeof(double));
  }
  return out;
}

}  // namespace

void save(const std::string& path,
          std::span<const NamedTensor> tensors,
          std::span<const NamedScalar> scalars) {
  std::unordered_map<std::string, int> seen;
  for (const NamedTensor& e : tensors) {
    if (!e.t || !e.t->defined()) {
      throw std::invalid_argument("checkpoint: \"" + e.name + "\" is not a tensor");
    }
    if (++seen[e.name] > 1) {
      throw std::invalid_argument("checkpoint: duplicate name \"" + e.name + "\"");
    }
  }
  for (const NamedScalar& s : scalars) {
    if (++seen[s.name] > 1) {
      throw std::invalid_argument("checkpoint: duplicate name \"" + s.name + "\"");
    }
  }

  std::vector<Tensor> host(tensors.size());
  std::vector<uint64_t> offset(tensors.size());

  int64_t at = align8(header_bytes(tensors, scalars));
  for (size_t i = 0; i < tensors.size(); ++i) {
    host[i] = tensors[i].t->pack().to(Device::CPU);
    offset[i] = uint64_t(at);
    at = align8(at + int64_t(host[i].numel()) * int64_t(dtype_size(host[i].dtype())));
  }

  const std::filesystem::path target(path);
  const std::filesystem::path tmp = std::filesystem::path(path + ".tmp");

  {
    std::ofstream os(tmp, std::ios::binary | std::ios::trunc);
    if (!os) throw std::runtime_error("checkpoint: cannot write " + tmp.string());

    os.write(kMagic, 4);
    put<uint32_t>(os, kCheckpointVersion);
    put<uint32_t>(os, uint32_t(tensors.size()));
    put<uint32_t>(os, uint32_t(scalars.size()));

    for (size_t i = 0; i < tensors.size(); ++i) {
      const Tensor& t = host[i];
      put_name(os, tensors[i].name);
      put<uint8_t>(os, uint8_t(t.dtype()));
      put<uint8_t>(os, uint8_t(t.shape().rank()));
      for (int a = 0; a < t.shape().rank(); ++a) put<int32_t>(os, t.shape().dim(a));
      put<uint64_t>(os, uint64_t(t.numel()) * dtype_size(t.dtype()));
      put<uint64_t>(os, offset[i]);
    }
    for (const NamedScalar& s : scalars) {
      put_name(os, s.name);
      put<double>(os, s.value);
    }

    for (size_t i = 0; i < tensors.size(); ++i) {
      const int64_t pad = int64_t(offset[i]) - os.tellp();
      if (pad < 0) throw std::logic_error("checkpoint: header size miscomputed");
      static const char zeros[8] = {};
      os.write(zeros, pad);
      os.write(static_cast<const char*>(host[i].raw()),
               std::streamsize(uint64_t(host[i].numel()) * dtype_size(host[i].dtype())));
    }

    os.flush();
    if (!os) throw std::runtime_error("checkpoint: write failed for " + tmp.string());
  }

  // Atomic on NTFS and POSIX both, and it replaces an existing file.
  std::filesystem::rename(tmp, target);
}

void load(const std::string& path,
          std::span<const NamedTensor> tensors,
          std::span<NamedScalar> scalars) {
  std::ifstream is(path, std::ios::binary);
  if (!is) throw std::runtime_error("checkpoint: cannot read " + path);

  const auto entries = read_header(is, path);
  std::unordered_map<std::string, const Entry*> by_name;
  for (const auto& [name, e] : entries) by_name.emplace(name, &e);

  for (const NamedTensor& want : tensors) {
    const auto it = by_name.find(want.name);
    if (it == by_name.end()) {
      throw std::runtime_error("checkpoint: " + path + " has no tensor named \"" +
                               want.name + "\"");
    }
    const Entry& e = *it->second;
    Tensor& dst = *want.t;

    if (e.dtype != dst.dtype()) {
      throw std::runtime_error("checkpoint: \"" + want.name + "\" is " +
                               dtype_name(e.dtype) + " in the file but " +
                               dtype_name(dst.dtype()) + " in the model");
    }
    if (e.shape != dst.shape()) {
      throw std::runtime_error("checkpoint: \"" + want.name + "\" is " +
                               e.shape.str() + " in the file but " +
                               dst.shape().str() + " in the model");
    }

    // Staged on the host, then copied to wherever this tensor lives, so the
    // file says nothing about devices.
    Tensor staging(e.shape, Device::CPU, e.dtype);
    is.seekg(std::streamoff(e.offset));
    is.read(static_cast<char*>(staging.raw()), std::streamsize(e.nbytes));
    if (!is) {
      throw std::runtime_error("checkpoint: " + path + " ends inside \"" +
                               want.name + "\"");
    }

    if (dst.is_contiguous()) {
      copy_bytes(dst.raw(), dst.device(), staging.raw(), Device::CPU, size_t(e.nbytes));
    } else {
      Tensor on_device = staging.to(dst.device());
      ops::unpack(dst, on_device);
    }
  }

  for (NamedScalar& want : scalars) {
    const auto it = by_name.find(want.name);
    if (it == by_name.end()) {
      throw std::runtime_error("checkpoint: " + path + " has no scalar named \"" +
                               want.name + "\"");
    }
    std::memcpy(&want.value, &it->second->offset, sizeof(double));
  }
}

std::vector<std::string> tensor_names(const std::string& path) {
  std::ifstream is(path, std::ios::binary);
  if (!is) throw std::runtime_error("checkpoint: cannot read " + path);

  std::vector<std::string> names;
  for (auto& [name, e] : read_header(is, path)) names.push_back(name);
  return names;
}

namespace {

struct RunState {
  std::vector<NamedTensor> tensors;
  std::vector<NamedScalar> scalars;
};

RunState run_state(Module& model, optim::Optimizer& opt) {
  RunState s;
  s.tensors = model.named_parameters();
  opt.collect_state("opt.", s.tensors, s.scalars);
  return s;
}

}  // namespace

void save_checkpoint(const std::string& path, Module& model,
                     optim::Optimizer& opt, int64_t step) {
  RunState s = run_state(model, opt);
  s.scalars.push_back({"step", double(step)});
  s.scalars.push_back({"rng.seed", double(random_seed())});
  s.scalars.push_back({"rng.counter", double(random_counter())});
  save(path, s.tensors, s.scalars);
}

int64_t load_checkpoint(const std::string& path, Module& model,
                        optim::Optimizer& opt) {
  RunState s = run_state(model, opt);
  s.scalars.push_back({"step", 0.0});
  s.scalars.push_back({"rng.seed", 0.0});
  s.scalars.push_back({"rng.counter", 0.0});

  load(path, s.tensors, s.scalars);

  opt.apply_state("opt.", s.scalars);
  set_random_state(uint64_t(scalar_value(s.scalars, "rng.seed")),
                   uint64_t(scalar_value(s.scalars, "rng.counter")));
  return int64_t(scalar_value(s.scalars, "step"));
}

void save_weights(const std::string& path, Module& model) {
  save(path, model.named_parameters());
}

void load_weights(const std::string& path, Module& model) {
  const std::vector<NamedTensor> params = model.named_parameters();
  load(path, params);
}

}  // namespace nn::io
