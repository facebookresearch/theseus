# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import math
from typing import Optional, Union

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


DeviceType = Optional[Union[str, torch.device]]
