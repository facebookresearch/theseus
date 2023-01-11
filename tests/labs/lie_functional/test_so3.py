# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest
from .common import check_lie_group_function
from theseus.labs.lie_functional.constants import TEST_EPS
import theseus.labs.lie_functional.so3 as so3

import torch


@pytest.mark.parametrize("batch_size", [1, 20, 100])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_exp(batch_size: int, dtype: torch.dtype):
    rng = torch.Generator()
    rng.manual_seed(0)
    tangent_vector = torch.rand(batch_size, 3, dtype=dtype, generator=rng)

    # check analytic backward for the operator
    check_lie_group_function(so3, "exp", TEST_EPS, tangent_vector)


@pytest.mark.parametrize("batch_size", [1, 20, 100])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_adjoint(batch_size: int, dtype: torch.dtype):
    rng = torch.Generator()
    rng.manual_seed(0)
    tangent_vector = torch.rand(batch_size, 3, dtype=dtype, generator=rng)
    group = so3.exp(tangent_vector)

    # check analytic backward for the operator
    check_lie_group_function(so3, "adjoint", TEST_EPS, group)


@pytest.mark.parametrize("batch_size", [1, 20, 100])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_inverse(batch_size: int, dtype: torch.dtype):
    rng = torch.Generator()
    rng.manual_seed(0)
    tangent_vector = torch.rand(batch_size, 3, dtype=dtype, generator=rng)
    group = so3.exp(tangent_vector)

    # check analytic backward for the operator
    check_lie_group_function(so3, "inverse", TEST_EPS, group)


@pytest.mark.parametrize("batch_size", [1, 20, 100])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_hat(batch_size: int, dtype: torch.dtype):
    rng = torch.Generator()
    rng.manual_seed(0)
    tangent_vector = torch.rand(batch_size, 3, dtype=dtype, generator=rng)

    # check analytic backward for the operator
    check_lie_group_function(so3, "hat", TEST_EPS, tangent_vector)


@pytest.mark.parametrize("batch_size", [1, 20, 100])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_vee(batch_size: int, dtype: torch.dtype):
    rng = torch.Generator()
    rng.manual_seed(0)
    tangent_vector = torch.rand(batch_size, 3, dtype=dtype, generator=rng)
    matrix = so3.hat(tangent_vector)

    # check analytic backward for the operator
    check_lie_group_function(so3, "vee", TEST_EPS, matrix)

    # check the correctness of hat and vee
    actual_tangent_vector = so3.vee(matrix)
    assert torch.allclose(actual_tangent_vector, tangent_vector, atol=TEST_EPS)
