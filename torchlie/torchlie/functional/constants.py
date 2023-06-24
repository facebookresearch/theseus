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


_NON_ZERO = 1.0

_INF = torch.inf

_NEAR_ZERO_D_ONE_MINUS_COSINE_BY_THETA2 = -1 / 12.0

_NEAR_ZERO_D_THETA_MINUS_SINE_BY_THETA3 = -1 / 60.0

DeviceType = Optional[Union[str, torch.device]]
