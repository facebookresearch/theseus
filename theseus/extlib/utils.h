// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <torch/extension.h>


// Various checks for tensor dimensions, dtype and devices
#define THESEUS_BASE_TENSOR_CHECK(tensor, d, d1, dt)             \
  do {                                                          \
    TORCH_CHECK(tensor.dim() == d);                             \
    TORCH_CHECK(tensor.size(0) == d1);                          \
    TORCH_CHECK(tensor.dtype() == dt);                          \
  } while (0)

#define THESEUS_TENSOR_CHECK_CPU(tensor, d, d1, dt)         \
  do {                                                          \
    THESEUS_BASE_TENSOR_CHECK(tensor, d, d1, dt);                \
    TORCH_CHECK(tensor.device().is_cpu());                      \
  } while (0)

#define THESEUS_TENSOR_CHECK_CUDA(tensor, d, d1, dt)        \
  do {                                                          \
    THESEUS_BASE_TENSOR_CHECK(tensor, d, d1, dt);                \
    TORCH_CHECK(tensor.device().is_cuda());                     \
  } while (0)

