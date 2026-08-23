#pragma once

#include <span>
#include <string>
#include <vector>

#include <nn/core/state.h>
#include <nn/nn/module.h>
#include <nn/optim/optim.h>

namespace nn::io {

// A checkpoint is a name -> tensor map plus a name -> double map, in one file.
//
//   magic "NNCK" | version u32 | n_tensors u32 | n_scalars u32
//   n_tensors x { name_len u16, name, dtype u8, rank u8, dims i32[rank],
//                 nbytes u64, offset u64 }
//   n_scalars x { name_len u16, name, value f64 }
//   padding to 8 bytes, then the tensor data, each entry 8-byte aligned
//
// Little-endian and IEEE-754, i.e. whatever the machine already is.
inline constexpr uint32_t kCheckpointVersion = 1;

// Writes to path + ".tmp" and renames over path when the last byte is down.
void save(const std::string& path,
          std::span<const NamedTensor> tensors,
          std::span<const NamedScalar> scalars = {});

// Fills every tensor through its pointer and every scalar's value in place,
// looking each up by name.
void load(const std::string& path,
          std::span<const NamedTensor> tensors,
          std::span<NamedScalar> scalars = {});

// Every name in the file, in file order. For inspecting a checkpoint without
// having the model that wrote it.
std::vector<std::string> tensor_names(const std::string& path);

// Model weights, optimiser moments and step count, the RNG position, and the
// caller's step number: everything needed to resume as if nothing happened.
void save_checkpoint(const std::string& path, Module& model,
                     optim::Optimizer& opt, int64_t step);

// Returns the step number the checkpoint was written at.
int64_t load_checkpoint(const std::string& path, Module& model,
                        optim::Optimizer& opt);

// Weights only
void save_weights(const std::string& path, Module& model);
void load_weights(const std::string& path, Module& model);

}  // namespace nn::io
