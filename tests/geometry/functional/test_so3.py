# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest
from .common import check_lie_group_function
from theseus.geometry.functional.constants import TEST_EPS
from theseus.geometry.functional import SO3

import torch


@pytest.mark.parametrize("batch_size", [1, 20, 100])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_exp_map(batch_size: int, dtype: torch.dtype):
    tangent_vector = torch.rand(batch_size, 3, dtype=dtype)
    check_lie_group_function(SO3, "exp_map", TEST_EPS, tangent_vector)
