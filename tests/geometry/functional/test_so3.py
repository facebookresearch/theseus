# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest
from .common import check_lie_group_function
from theseus.geometry.functional.constants import TEST_EPS
import theseus.geometry.functional.so3 as so3

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
    group = so3.rand(batch_size, generator=rng, dtype=dtype)

    # check analytic backward for the operator
    check_lie_group_function(so3, "adjoint", TEST_EPS, group)


@pytest.mark.parametrize("batch_size", [1, 20, 100])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_inverse(batch_size: int, dtype: torch.dtype):
    rng = torch.Generator()
    rng.manual_seed(0)
    group = so3.rand(batch_size, generator=rng, dtype=dtype)

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


@pytest.mark.parametrize("batch_size", [1, 20, 100])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_compose(batch_size: int, dtype: torch.dtype):
    rng = torch.Generator()
    rng.manual_seed(0)
    group0 = so3.rand(batch_size, generator=rng, dtype=dtype)
    group1 = so3.rand(batch_size, generator=rng, dtype=dtype)

    # check analytic backward for the operator
    check_lie_group_function(so3, "compose", TEST_EPS, group0, group1)


@pytest.mark.parametrize("batch_size", [1, 20, 100])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_quaternion_to_rotation(batch_size: int, dtype: torch.dtype):
    rng = torch.Generator()
    rng.manual_seed(0)
    quaternion = torch.rand(batch_size, 4, dtype=dtype, generator=rng)
    quaternion = quaternion / torch.norm(quaternion, dim=1, keepdim=True)

    # check analytic backward for the operator
    check_lie_group_function(so3, "quaternion_to_rotation", TEST_EPS, quaternion)


@pytest.mark.parametrize("batch_size", [1, 20, 100])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_lift(batch_size: int, dtype: torch.dtype):
    rng = torch.Generator()
    matrix = torch.rand(
        batch_size,
        int(torch.randint(1, 20, (1,), generator=rng)),
        3,
        dtype=dtype,
        generator=rng,
    )

    # check analytic backward for the operator
    check_lie_group_function(so3, "lift", TEST_EPS, matrix)


@pytest.mark.parametrize("batch_size", [1, 20, 100])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_project(batch_size: int, dtype: torch.dtype):
    rng = torch.Generator()
    matrix = torch.rand(
        batch_size,
        int(torch.randint(1, 20, (1,), generator=rng)),
        3,
        3,
        dtype=dtype,
        generator=rng,
    )

    # check analytic backward for the operator
    check_lie_group_function(so3, "project", TEST_EPS, matrix)
