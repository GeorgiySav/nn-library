#include <nn/data/token_dataset.h>

#include <fstream>
#include <system_error>

#include <mio/mio.hpp>

namespace nn::data {

std::vector<int32_t> load_tokens(const std::string& path) {
  std::ifstream in(path, std::ios::binary | std::ios::ate);
  if (!in) throw std::runtime_error("load_tokens: cannot open " + path);

  const std::streamoff bytes = in.tellg();
  if (bytes < 0 || bytes % 4 != 0) {
    // tokens are stored as raw int32_t, so a size that isn't a multiple of 4
    // means the file is truncated or not a token file at all
    throw std::runtime_error("load_tokens: file size is not a multiple of 4: " + path);
  }

  in.seekg(0);
  std::vector<int32_t> ids(size_t(bytes) / 4);
  in.read(reinterpret_cast<char*>(ids.data()), bytes);
  if (!in) throw std::runtime_error("load_tokens: truncated read: " + path);

  return ids;
}

// mio::mmap_source is kept behind a pimpl so <mio/mio.hpp> stays out of
// token_dataset.h and out of every translation unit that just wants to read
// tokens.
struct MappedTokens::Impl {
  mio::mmap_source file;
};

MappedTokens::MappedTokens(const std::string& path)
  : impl_(std::make_unique<Impl>()) {
  std::error_code ec;
  impl_->file.map(path, ec);
  if (ec) throw std::system_error(ec, "MappedTokens: " + path);
  if (impl_->file.size() % 4 != 0) {
    throw std::runtime_error("MappedTokens: file size is not a multiple of 4: " + path);
  }
}

MappedTokens::~MappedTokens() = default;
MappedTokens::MappedTokens(MappedTokens&&) noexcept = default;
MappedTokens& MappedTokens::operator=(MappedTokens&&) noexcept = default;

// the mapping is never copied into RAM here, so this reinterprets the raw
// mapped bytes directly as int32_t tokens (the file must already be
// little-endian, native-alignment int32 data)
std::span<const int32_t> MappedTokens::tokens() const {
  return {reinterpret_cast<const int32_t*>(impl_->file.data()), impl_->file.size() / 4};
}

}  // namespace nn::data
