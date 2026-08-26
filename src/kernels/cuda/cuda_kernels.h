#pragma once

#include <kernels/kernel_api.h>

namespace nn::kernels {

// Generated from the one kernel list; see naive_kernels.h for why this works.
#define NN_KERNEL(name, Type) Type cuda_##name;
#include <kernels/kernel_list.def>
#undef NN_KERNEL

// Not in the list: cuBLAS provides an alternative gemm rather than a slot of
// its own, and register_cublas_kernels overwrites t.gemm with it.
GemmFn cublas_gemm;

}
