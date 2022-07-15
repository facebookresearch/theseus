# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch

TEST_EPS = 5e-7
EPS = 1e-10
PI = math.pi

_SO2_NORMALIZATION_EPS = {
    torch.float32: 1e-12,
    torch.float64: 1e-12,
}

_SO2_MATRIX_EPS = {
    torch.float32: 1e-5,
    torch.float64: 4e-7,
}

_SE2_NEAR_ZERO_EPS = {torch.float32: 3e-2, torch.float64: 1e-6}

_SO3_NEAR_PI_EPS = {
    torch.float32: 1e-2,
    torch.float64: 1e-7,
}

_SO3_NEAR_ZERO_EPS = {
    torch.float32: 1e-2,
    torch.float64: 5e-3,
}

_SO3_MATRIX_EPS = {
    torch.float32: 4e-4,
    torch.float64: 1e-6,
}

_SO3_QUATERNION_EPS = {
    torch.float32: 2e-4,
    torch.float64: 5e-7,
}

_SO3_HAT_EPS = {
    torch.float32: 5e-6,
    torch.float64: 5e-7,
}

_SE3_NEAR_PI_EPS = {
    torch.float32: 1e-2,
    torch.float64: 1e-7,
}

_SE3_NEAR_ZERO_EPS = {
    torch.float32: 1e-2,
    torch.float64: 5e-3,
}

_SE3_HAT_EPS = {
    torch.float32: 5e-6,
    torch.float64: 5e-7,
}
