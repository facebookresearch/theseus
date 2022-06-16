# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch

EPS = 1e-10
PI = math.pi

_SE2_EPS = {torch.float32: 3e-4, torch.float64: 5e-7}

_SE3_NEAR_PI_EPS = {
    torch.float32: 1e-2,
    torch.float64: 1e-7,
}

_SE3_NEAR_ZERO_EPS = {
    torch.float32: 1e-2,
    torch.float64: 5e-3,
}
