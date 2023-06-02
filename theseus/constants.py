# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Optional, Union
import math

import torch

TEST_EPS = 5e-7
EPS = 1e-10
PI = math.pi

__FROM_THESEUS_LAYER_TOKEN__ = "__FROM_THESEUS_LAYER_TOKEN__"


def _CHECK_DTYPE_SUPPORTED(dtype):
    if dtype not in [torch.float32, torch.float64]:
        raise ValueError(
            f"Unsupported data type {dtype}. "
            "Theseus only supports 32- and 64-bit tensors."
        )


class EPSDict:
    def __init__(self, float32_eps, float64_eps):
        self.float32_eps = float32_eps
        self.float64_eps = float64_eps

    def __getitem__(self, dtype):
        _CHECK_DTYPE_SUPPORTED(dtype)
        if dtype is torch.float32:
            return self.float32_eps
        else:
            return self.float64_eps


_SO2_NORMALIZATION_EPS = EPSDict(float32_eps=1e-12, float64_eps=1e-12)

_SO2_MATRIX_EPS = EPSDict(float32_eps=1e-5, float64_eps=4e-7)

_SE2_NEAR_ZERO_EPS = EPSDict(float32_eps=3e-2, float64_eps=1e-6)

DeviceType = Optional[Union[str, torch.device]]
