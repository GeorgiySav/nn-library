#pragma once

// Every layer, for callers who want the model-building vocabulary and nothing
// else -- their own data pipeline, their own training loop. <nn/nn.h> pulls
// this in along with the rest of the library.
//
// Each header below is usable on its own. nn/module/module.h in particular is
// just the base class, with no dependency on autograd, so a translation unit
// that only needs to name Module does not pay for the layers.

#include <nn/module/module.h>

#include <nn/module/activation.h>
#include <nn/module/container.h>
#include <nn/module/dropout.h>
#include <nn/module/embedding.h>
#include <nn/module/linear.h>
#include <nn/module/norm.h>
