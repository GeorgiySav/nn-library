#include <array>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#ifndef NN_PROJECT_ROOT
#define NN_PROJECT_ROOT "."
#endif

#include <nn/nn.h>
#include <nn/module.h>
#include <nn/data/token_dataset.h>
#include <nn/data/dataloader.h>

using namespace nn;

struct LMConfig {
  int vocab_size = 0;   // set from the corpus once it's read
  int embed_dim = 32;
  float dropout_rate = 0.1f;
  int window = 16;
  int stride = 1;
  int batch_size = 8;
};

class CharTokenizer {
public:
  explicit CharTokenizer(const std::string& text) {
    id_of_.fill(-1);
    for (unsigned char c : text) {
      if (id_of_[c] < 0) {
        id_of_[c] = int(chars_.size());
        chars_.push_back(char(c));
      }
    }
  }

  int vocab_size() const { return int(chars_.size()); }

  std::vector<int32_t> encode(const std::string& text) const {
    std::vector<int32_t> ids;
    ids.reserve(text.size());
    for (unsigned char c : text) ids.push_back(id_of_[c]);
    return ids;
  }

  std::string decode(std::span<const int32_t> ids) const {
    std::string s;
    s.reserve(ids.size());
    for (int32_t id : ids) s.push_back(chars_[size_t(id)]);
    return s;
  }

private:
  std::array<int, 256> id_of_;
  std::vector<char> chars_;
};

static std::string read_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("could not open " + path);
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

static Tensor* linear_weight(Linear& lin) {
  std::vector<NamedTensor> params;
  lin.collect_named("", params);
  for (NamedTensor& p : params) {
    if (p.name == "w") return p.t;
  }
  throw std::logic_error("linear_weight: no tensor named \"w\"");
}

class LM : public Module {
public:
  LM(const LMConfig& config, Pcg32& rng)
    : config_(config),
      token_embed_(Embedding(config.vocab_size, config.embed_dim, rng)),
      layers_(Sequential(
        Dropout(config.dropout_rate),
        Linear(config.embed_dim, config.embed_dim*4, rng),
        ReLu(),
        Linear(4*config.embed_dim, config.embed_dim, rng),
        ReLu()
      )),
      mix_drop_(Dropout(config.dropout_rate)),
      mix1_(Linear(config.window, config.window, rng)),
      mix2_(Linear(config.window, config.window, rng)),
      head_(Linear(config.embed_dim, config.vocab_size, rng)),
      keep_mask_(causal_keep_mask(config.window)) {
    enforce_causal_mask();
  }

  Tensor forward(const Tensor& x) override {
    // x: [b, t]
    Tensor t_embed = token_embed_.forward(x); // [b, t, e]
    Tensor l = layers_.forward(t_embed); // [b, t, e]

    Tensor m = l.transpose(-2, -1).contiguous();   // [b, e, t]
    m = mix_drop_.forward(m);
    m = mix1_.forward(m).relu();
    m = mix2_.forward(m).relu();
    m = m.transpose(-2, -1).contiguous();          // [b, t, e]

    Tensor logits = head_.forward(m); // [b, t, v]
    return logits;
  }

  void collect_named(const std::string& prefix, std::vector<NamedTensor>& out) override {
    token_embed_.collect_named(prefix + "embed.", out);
    layers_.collect_named(prefix + "mlp.", out);
    mix1_.collect_named(prefix + "mix1.", out);
    mix2_.collect_named(prefix + "mix2.", out);
    head_.collect_named(prefix + "head.", out);
  }

  void to(Device d) {
    Module::to(d);
    keep_mask_ = keep_mask_.to(d);
  }

  void enforce_causal_mask() {
    mask_(*linear_weight(mix1_));
    mask_(*linear_weight(mix2_));
  }

private:
  static Tensor causal_keep_mask(int window) {
    return tril_mask(window).transpose_view(0, 1);
  }

  void mask_(Tensor& w) {
    const bool rg = w.requires_grad();
    w = ops::binary(ops::BinaryOp::Mul, w, keep_mask_);
    w.set_requires_grad(rg);
  }

  LMConfig config_;

  Embedding token_embed_;
  Sequential layers_;
  Dropout mix_drop_;
  Linear mix1_;
  Linear mix2_;
  Linear head_;
  Tensor keep_mask_;
};

int main() {
  const std::string text = read_file(NN_PROJECT_ROOT "/data/shakespeare/text.txt");

  CharTokenizer tok(text);
  const std::vector<int32_t> ids = tok.encode(text);

  LMConfig config{};
  config.vocab_size = tok.vocab_size();

  // setup
  Pcg32 rng(42);
  Device device = cuda_device_count() > 0 ? Device::CUDA : Device::CPU;

  std::printf("corpus: %zu characters, vocab size %d\n", text.size(), config.vocab_size);

  auto dataset = std::make_shared<data::TokenDataset>(
    std::span<const int32_t>(ids), config.window, config.stride);
  data::DataLoader<2> loader(
    dataset,
    config.batch_size,
    rng,
    /*shuffle=*/false, /*drop_last=*/false,
    device);

  // model
  LM model(config, rng);
  model.to(device);

  optim::Adam opt(model.parameters(), 0.001f);

  // training
  for (int epoch = 0; epoch < 10; ++epoch) {
    loader.reset();
    double running = 0.0;
    int steps = 0;

    while (loader.has_next()) {
        auto [xb, yb] = loader.next();
        opt.zero_grad();

        autograd::GradScope grad;
        Tensor logits = model.forward(xb);   // [b, t, v]

        // cross_entropy wants rank-2 logits and rank-1 labels
        const int B = logits.extent(0),
                  T = logits.extent(1),
                  V = logits.extent(2);
        Tensor loss = nn::cross_entropy(logits.reshape(Shape{B * T, V}),
                                        yb.reshape(Shape{B * T}));
        loss.backward();
        opt.step();
        model.enforce_causal_mask();

        running += loss.item();
        ++steps;
    }

    std::printf("epoch %2d: loss %.4f\n", epoch, running / steps);
  }
}
