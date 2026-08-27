#include "test_harness.h"
#include "devices.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nn/autograd/functions.h>
#include <nn/autograd/tape.h>
#include <nn/core/rng.h>
#include <nn/core/tensor.h>
#include <nn/io/checkpoint.h>
#include <nn/module.h>
#include <nn/optim/optim.h>

namespace {

std::vector<float> host_of(const nn::Tensor& t) {
  const nn::Tensor h = t.pack().to(nn::Device::CPU);
  return std::vector<float>(h.host_data(), h.host_data() + h.numel());
}

// Every test writes under the build's temp directory and cleans up after
// itself, so a failing run leaves nothing behind for the next one to read.
struct TempFile {
  std::filesystem::path path;
  explicit TempFile(const char* name)
    : path(std::filesystem::temp_directory_path() / name) {
    std::filesystem::remove(path);
  }
  ~TempFile() {
    std::filesystem::remove(path);
    std::filesystem::remove(std::filesystem::path(path.string() + ".tmp"));
  }
  std::string str() const { return path.string(); }
};

nn::Sequential make_model(nn::Device d, uint64_t seed = 4) {
  nn::Pcg32 rng(seed);
  nn::Sequential m(nn::Linear(6, 5, rng), nn::ReLu(),
                   nn::Dropout(0.25f), nn::Linear(5, 3, rng));
  m.to(d);
  return m;
}

// One training step, so the parameters and the moments are both non-trivial.
void train_steps(nn::Module& model, nn::optim::Optimizer& opt,
                 const nn::Tensor& x, const nn::Tensor& labels, int steps) {
  for (int i = 0; i < steps; ++i) {
    opt.zero_grad();
    nn::autograd::GradScope grad;
    nn::Tensor loss = nn::cross_entropy(model.forward(x), labels);
    loss.backward();
    opt.step();
  }
}

}  // namespace

NN_TEST(named_parameters_carry_their_path) {
  nn::Pcg32 rng(1);
  nn::Sequential model(nn::Linear(4, 8, rng), nn::ReLu(),
                       nn::LayerNorm(8), nn::Linear(8, 2, rng));

  const std::vector<nn::NamedTensor> named = model.named_parameters();
  const char* want[] = {"0.w", "0.b", "2.w", "2.b", "3.w", "3.b"};
  NN_CHECK(named.size() == 6);
  for (size_t i = 0; i < named.size(); ++i) NN_CHECK(named[i].name == want[i]);

  NN_CHECK(named[2].name == "2.w");

  const std::vector<nn::Tensor*> flat = model.parameters();
  NN_CHECK(flat.size() == named.size());
  for (size_t i = 0; i < flat.size(); ++i) NN_CHECK(flat[i] == named[i].t);
}

NN_TEST(a_round_trip_is_exact) {
  TempFile file("nn_ck_roundtrip.bin");

  NN_TEST_FOR_EACH_DEVICE(dev) {
    nn::Sequential a = make_model(dev, 7);
    nn::Sequential b = make_model(dev, 99);      // different init

    const std::vector<float> before = host_of(*a.parameters()[0]);
    NN_CHECK(before != host_of(*b.parameters()[0]));

    nn::io::save_weights(file.str(), a);
    nn::io::load_weights(file.str(), b);

    const std::vector<nn::Tensor*> pa = a.parameters(), pb = b.parameters();
    for (size_t i = 0; i < pa.size(); ++i) {
      const std::vector<float> va = host_of(*pa[i]), vb = host_of(*pb[i]);
      NN_CHECK(va.size() == vb.size());
      for (size_t j = 0; j < va.size(); ++j) NN_CHECK_CLOSE(vb[j], va[j], 0.0f);
    }
  }
}

NN_TEST(a_checkpoint_crosses_devices) {
  if (!nn::test::have_cuda()) return;

  TempFile file("nn_ck_devices.bin");

  nn::Sequential on_cpu = make_model(nn::Device::CPU, 11);
  nn::Sequential on_gpu = make_model(nn::Device::CUDA, 12);

  nn::io::save_weights(file.str(), on_cpu);
  nn::io::load_weights(file.str(), on_gpu);

  const std::vector<nn::Tensor*> pc = on_cpu.parameters(), pg = on_gpu.parameters();
  for (size_t i = 0; i < pc.size(); ++i) {
    NN_CHECK(pg[i]->device() == nn::Device::CUDA);   // still where it was
    const std::vector<float> a = host_of(*pc[i]), b = host_of(*pg[i]);
    for (size_t j = 0; j < a.size(); ++j) NN_CHECK_CLOSE(b[j], a[j], 0.0f);
  }
}

NN_TEST(a_resumed_run_continues_identically) {
  TempFile file("nn_ck_resume.bin");

  NN_TEST_FOR_EACH_DEVICE(dev) {
    const nn::Tensor x = nn::Tensor::from({{0.4f, -0.7f, 1.1f, 0.2f, 0.9f, -0.3f},
                                           {-0.3f, 0.9f, 0.1f, -1.2f, 0.5f, 0.8f}}, dev);
    const nn::Tensor labels = nn::Tensor::from_i32({0, 2}, dev);

    auto run = [&](bool interrupt) {
      nn::manual_seed(2024);
      nn::Sequential model = make_model(dev);   // with dropout live
      nn::optim::AdamW opt(model.parameters(), 0.05f, 0.01f);
      nn::optim::Schedule sched(0.05f, 10, 2);

      for (int64_t s = 0; s < 5; ++s) {
        opt.set_lr(sched.at(s));
        train_steps(model, opt, x, labels, 1);
      }

      if (interrupt) {
        nn::io::save_checkpoint(file.str(), model, opt, 5);

        nn::Sequential fresh = make_model(dev, 12345);
        nn::optim::AdamW fopt(fresh.parameters(), 0.0f, 0.01f);
        const int64_t step = nn::io::load_checkpoint(file.str(), fresh, fopt);
        NN_CHECK(step == 5);
        NN_CHECK_CLOSE(fopt.lr(), sched.at(4), 0.0f);   // lr came back too

        for (int64_t s = step; s < 10; ++s) {
          fopt.set_lr(sched.at(s));
          train_steps(fresh, fopt, x, labels, 1);
        }
        return host_of(*fresh.parameters()[0]);
      }

      for (int64_t s = 5; s < 10; ++s) {
        opt.set_lr(sched.at(s));
        train_steps(model, opt, x, labels, 1);
      }
      return host_of(*model.parameters()[0]);
    };

    const std::vector<float> straight = run(false), resumed = run(true);
    NN_CHECK(straight.size() == resumed.size());
    for (size_t i = 0; i < straight.size(); ++i) {
      NN_CHECK_CLOSE(resumed[i], straight[i], 1e-6f);
    }
  }
}

NN_TEST(the_optimizer_step_count_survives_and_matters) {
  TempFile file("nn_ck_step.bin");

  NN_TEST_FOR_EACH_DEVICE(dev) {
    const nn::Tensor x = nn::Tensor::full({2, 6}, 0.5f, dev);
    const nn::Tensor labels = nn::Tensor::from_i32({0, 2}, dev);

    nn::Sequential model = make_model(dev);
    model.eval();
    nn::optim::AdamW opt(model.parameters(), 0.02f, 0.0f);
    train_steps(model, opt, x, labels, 30);
    nn::io::save_checkpoint(file.str(), model, opt, 30);

    // step is in the file, and it is the value the optimiser had
    std::vector<nn::NamedScalar> want{{"opt.step", 0.0}};
    nn::io::load(file.str(), {}, want);
    NN_CHECK_CLOSE(float(want[0].value), 30.0f, 0.0f);

    // restored, one more step lands where continuing would have
    nn::Sequential a = make_model(dev), b = make_model(dev, 555);
    a.eval(); b.eval();
    nn::optim::AdamW aopt(a.parameters(), 0.02f, 0.0f);
    nn::optim::AdamW bopt(b.parameters(), 0.02f, 0.0f);

    nn::io::load_checkpoint(file.str(), a, aopt);
    nn::io::load_checkpoint(file.str(), b, bopt);

    train_steps(model, opt, x, labels, 1);   // the original carries straight on
    train_steps(a, aopt, x, labels, 1);      // the resumed one

    const std::vector<float> went_on = host_of(*model.parameters()[0]);
    const std::vector<float> resumed = host_of(*a.parameters()[0]);
    for (size_t i = 0; i < went_on.size(); ++i) {
      NN_CHECK_CLOSE(resumed[i], went_on[i], 1e-6f);
    }
  }
}

NN_TEST(the_rng_position_survives) {
  TempFile file("nn_ck_rng.bin");

  nn::Sequential model = make_model(nn::Device::CPU);
  nn::optim::SGD opt(model.parameters(), 0.1f);

  nn::manual_seed(777);
  const nn::Tensor x = nn::Tensor::full({64}, 1.0f);
  const std::vector<float> first = host_of(nn::dropout(x, 0.5f));

  const uint64_t counter = nn::random_counter();
  NN_CHECK(counter == 64);                    // one call reserved 64 values
  nn::io::save_checkpoint(file.str(), model, opt, 3);

  // Somebody else's run moves the generator a long way on
  nn::manual_seed(1);
  for (int i = 0; i < 5; ++i) nn::dropout(x, 0.5f);

  nn::io::load_checkpoint(file.str(), model, opt);
  NN_CHECK(nn::random_seed() == 777);
  NN_CHECK(nn::random_counter() == counter);

  // the next mask is the one the interrupted run would have drawn, not `first`
  const std::vector<float> next = host_of(nn::dropout(x, 0.5f));
  int same = 0;
  for (size_t i = 0; i < next.size(); ++i) same += (next[i] == first[i]);
  NN_CHECK(same < int(next.size()));

  nn::manual_seed(777);
  nn::dropout(x, 0.5f);                       // burn the same 64 values
  const std::vector<float> expect = host_of(nn::dropout(x, 0.5f));
  for (size_t i = 0; i < next.size(); ++i) NN_CHECK_CLOSE(next[i], expect[i], 0.0f);
}

NN_TEST(weights_only_reads_out_of_a_full_checkpoint) {
  TempFile full("nn_ck_full.bin"), thin("nn_ck_thin.bin");

  nn::Sequential model = make_model(nn::Device::CPU, 31);
  nn::optim::AdamW opt(model.parameters(), 0.01f);

  nn::io::save_checkpoint(full.str(), model, opt, 12);
  nn::io::save_weights(thin.str(), model);

  // the optimiser state is the bulk of it: two moments per parameter
  const auto bytes = [](const TempFile& f) {
    return int64_t(std::filesystem::file_size(f.path));
  };
  NN_CHECK(bytes(full) > 2 * bytes(thin));

  const std::vector<std::string> names = nn::io::tensor_names(full.str());
  NN_CHECK(std::find(names.begin(), names.end(), "0.w") != names.end());
  NN_CHECK(std::find(names.begin(), names.end(), "opt.m.0") != names.end());
  NN_CHECK(std::find(names.begin(), names.end(), "opt.step") != names.end());
  NN_CHECK(std::find(names.begin(), names.end(), "opt.m.0") ==
           std::find(names.begin(), names.end(), "opt.m.0"));

  // weights-only load out of the full file
  nn::Sequential other = make_model(nn::Device::CPU, 32);
  nn::io::load_weights(full.str(), other);
  const std::vector<float> a = host_of(*model.parameters()[0]);
  const std::vector<float> b = host_of(*other.parameters()[0]);
  for (size_t i = 0; i < a.size(); ++i) NN_CHECK_CLOSE(b[i], a[i], 0.0f);

  // and the thin file has no optimiser state to offer
  nn::optim::AdamW fresh_opt(other.parameters(), 0.01f);
  NN_CHECK_THROWS(nn::io::load_checkpoint(thin.str(), other, fresh_opt),
                  std::runtime_error);
}

// The reason names are worth the trouble: a shape mismatch says which tensor.
NN_TEST(a_mismatched_model_is_rejected_by_name) {
  TempFile file("nn_ck_mismatch.bin");

  nn::Pcg32 rng(3);
  nn::Sequential saved(nn::Linear(6, 5, rng), nn::ReLu(), nn::Linear(5, 3, rng));
  nn::io::save_weights(file.str(), saved);

  {   // a different width
    nn::Pcg32 r2(3);
    nn::Sequential wider(nn::Linear(6, 8, r2), nn::ReLu(), nn::Linear(8, 3, r2));
    NN_CHECK_THROWS(nn::io::load_weights(file.str(), wider), std::runtime_error);
  }
  {   // an extra layer, so the names shift and "4.w" is not in the file
    nn::Pcg32 r3(3);
    nn::Sequential deeper(nn::Linear(6, 5, r3), nn::ReLu(),
                          nn::Linear(5, 5, r3), nn::ReLu(),
                          nn::Linear(5, 3, r3));
    NN_CHECK_THROWS(nn::io::load_weights(file.str(), deeper), std::runtime_error);
  }
  {   // a LayerNorm where the ReLu was: same parameter count is not enough
    nn::Pcg32 r4(3);
    nn::Sequential swapped(nn::Linear(6, 5, r4), nn::LayerNorm(5),
                           nn::Linear(5, 3, r4));
    NN_CHECK_THROWS(nn::io::load_weights(file.str(), swapped), std::runtime_error);
  }
}

NN_TEST(a_damaged_or_foreign_file_is_rejected) {
  TempFile good("nn_ck_good.bin"), junk("nn_ck_junk.bin"), cut("nn_ck_cut.bin");

  nn::Sequential model = make_model(nn::Device::CPU);
  nn::io::save_weights(good.str(), model);

  {
    std::ofstream os(junk.path, std::ios::binary);
    os << "this is not a checkpoint, it is a text file";
  }
  NN_CHECK_THROWS(nn::io::load_weights(junk.str(), model), std::runtime_error);
  NN_CHECK_THROWS(nn::io::tensor_names(junk.str()), std::runtime_error);

  {
    const auto n = std::filesystem::file_size(good.path);
    std::vector<char> bytes(n);
    std::ifstream(good.path, std::ios::binary).read(bytes.data(), std::streamsize(n));
    std::ofstream os(cut.path, std::ios::binary);
    os.write(bytes.data(), std::streamsize(n - 8));
  }
  NN_CHECK_THROWS(nn::io::load_weights(cut.str(), model), std::runtime_error);

  NN_CHECK_THROWS(nn::io::load_weights("nn_ck_does_not_exist.bin", model),
                  std::runtime_error);
}

NN_TEST(save_rejects_duplicate_names_and_leaves_no_file) {
  TempFile file("nn_ck_dup.bin");

  nn::Tensor a = nn::Tensor::zeros({2});
  nn::Tensor b = nn::Tensor::zeros({2});
  const std::vector<nn::NamedTensor> both{{"w", &a}, {"w", &b}};

  NN_CHECK_THROWS(nn::io::save(file.str(), both), std::invalid_argument);
  NN_CHECK(!std::filesystem::exists(file.path));

  // a name that collides with a scalar is caught too
  const std::vector<nn::NamedTensor> one{{"step", &a}};
  const std::vector<nn::NamedScalar> s{{"step", 1.0}};
  NN_CHECK_THROWS(nn::io::save(file.str(), one, s), std::invalid_argument);
}

NN_TEST(saving_over_an_existing_checkpoint_replaces_it_atomically) {
  TempFile file("nn_ck_atomic.bin");

  nn::Sequential first = make_model(nn::Device::CPU, 41);
  nn::io::save_weights(file.str(), first);
  const auto size_one = std::filesystem::file_size(file.path);

  nn::Sequential second = make_model(nn::Device::CPU, 42);
  nn::io::save_weights(file.str(), second);
  NN_CHECK(std::filesystem::file_size(file.path) == size_one);

  // the temp file is gone, not left beside the checkpoint
  NN_CHECK(!std::filesystem::exists(std::filesystem::path(file.str() + ".tmp")));

  nn::Sequential reader = make_model(nn::Device::CPU, 43);
  nn::io::load_weights(file.str(), reader);
  const std::vector<float> want = host_of(*second.parameters()[0]);
  const std::vector<float> got = host_of(*reader.parameters()[0]);
  for (size_t i = 0; i < want.size(); ++i) NN_CHECK_CLOSE(got[i], want[i], 0.0f);
}

NN_TEST(dtypes_and_odd_shapes_round_trip) {
  TempFile file("nn_ck_dtypes.bin");

  nn::Tensor ints = nn::Tensor::from_i32({7, -3, 0, 100000});
  nn::Tensor scalar = nn::Tensor::scalar(2.5f);
  nn::Tensor rank4 = nn::Tensor::full({2, 1, 3, 2}, 1.75f);

  nn::io::save(file.str(), std::vector<nn::NamedTensor>{
      {"i", &ints}, {"s", &scalar}, {"r4", &rank4}});

  nn::Tensor ints2 = nn::Tensor::zeros({4}, nn::Device::CPU, nn::DType::I32);
  nn::Tensor scalar2 = nn::Tensor::scalar(0.0f);
  nn::Tensor rank42 = nn::Tensor::zeros({2, 1, 3, 2});
  nn::io::load(file.str(), std::vector<nn::NamedTensor>{
      {"i", &ints2}, {"s", &scalar2}, {"r4", &rank42}});

  NN_CHECK(ints2.host_data_i32()[0] == 7);
  NN_CHECK(ints2.host_data_i32()[1] == -3);
  NN_CHECK(ints2.host_data_i32()[3] == 100000);
  NN_CHECK_CLOSE(scalar2.item(), 2.5f, 0.0f);
  NN_CHECK(rank42.shape() == nn::Shape({2, 1, 3, 2}));
  for (float v : host_of(rank42)) NN_CHECK_CLOSE(v, 1.75f, 0.0f);

  nn::Tensor wrong = nn::Tensor::zeros({4});
  NN_CHECK_THROWS(nn::io::load(file.str(), std::vector<nn::NamedTensor>{{"i", &wrong}}),
                  std::runtime_error);
}
