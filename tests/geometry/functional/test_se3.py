# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest
from .common import check_lie_group_function
from theseus.geometry.functional.constants import TEST_EPS
import theseus.geometry.functional.se3 as se3

import torch


@pytest.mark.parametrize("batch_size", [1, 20, 100])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_exp(batch_size: int, dtype: torch.dtype):
    rng = torch.Generator()
    rng.manual_seed(0)
    tangent_vector = torch.rand(batch_size, 6, dtype=dtype, generator=rng)

    # check analytic backward for the operator
    check_lie_group_function(se3, "exp", 1e-6, (tangent_vector,))


@pytest.mark.parametrize("batch_size", [1, 20, 100])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_adjoint(batch_size: int, dtype: torch.dtype):
    rng = torch.Generator()
    rng.manual_seed(0)
    group = se3.rand(batch_size, generator=rng, dtype=dtype)

    # check analytic backward for the operator
    check_lie_group_function(se3, "adjoint", TEST_EPS, (group,))


@pytest.mark.parametrize("batch_size", [1, 20, 100])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_inverse(batch_size: int, dtype: torch.dtype):
    rng = torch.Generator()
    rng.manual_seed(0)
    group = se3.rand(batch_size, generator=rng, dtype=dtype)

    # check analytic backward for the operator
    check_lie_group_function(se3, "inverse", TEST_EPS, (group,))


@pytest.mark.parametrize("batch_size", [1, 20, 100])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_hat(batch_size: int, dtype: torch.dtype):
    rng = torch.Generator()
    rng.manual_seed(0)
    tangent_vector = torch.rand(batch_size, 6, dtype=dtype, generator=rng)

    # check analytic backward for the operator
    check_lie_group_function(se3, "hat", TEST_EPS, (tangent_vector,))


@pytest.mark.parametrize("batch_size", [1, 20, 100])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_vee(batch_size: int, dtype: torch.dtype):
    rng = torch.Generator()
    rng.manual_seed(0)
    tangent_vector = torch.rand(batch_size, 6, dtype=dtype, generator=rng)
    matrix = se3.hat(tangent_vector)

    # check analytic backward for the operator
    check_lie_group_function(se3, "vee", TEST_EPS, (matrix,))

    # check the correctness of hat and vee
    actual_tangent_vector = se3.vee(matrix)
    assert torch.allclose(actual_tangent_vector, tangent_vector, atol=TEST_EPS)
