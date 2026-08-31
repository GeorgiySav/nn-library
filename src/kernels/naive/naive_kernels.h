#pragma once

#include <kernels/kernel_api.h>

namespace nn::kernels {

// generated from the one kernel list. `ReluFn naive_relu;` declares a function
// with exactly the signature the table slot expects, because the Fn aliases are
// function types rather than pointers to them. a body whose signature drifts
// fails to link rather than silently registering the wrong thing.
#define NN_KERNEL(name, Type) Type naive_##name;
#include <kernels/kernel_list.def>
#undef NN_KERNEL

}
