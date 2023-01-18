# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest

import torch

from tests.decorators import run_if_labs
from .common import TEST_EPS, check_lie_group_function


@run_if_labs()
@pytest.mark.parametrize("batch_size", [1, 20, 100])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_exp(batch_size: int, dtype: torch.dtype):
    import theseus.labs.lie_functional.so3 as so3

    rng = torch.Generator()
    rng.manual_seed(0)
    tangent_vector = torch.rand(batch_size, 3, dtype=dtype, generator=rng)

    # check analytic backward for the operator
    check_lie_group_function(so3, "exp", TEST_EPS, tangent_vector)


@run_if_labs()
@pytest.mark.parametrize("batch_size", [1, 20, 100])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_adjoint(batch_size: int, dtype: torch.dtype):
    import theseus.labs.lie_functional.so3 as so3

    rng = torch.Generator()
    rng.manual_seed(0)
    tangent_vector = torch.rand(batch_size, 3, dtype=dtype, generator=rng)
    group = so3.exp(tangent_vector)

    # check analytic backward for the operator
    check_lie_group_function(so3, "adjoint", TEST_EPS, group)


@run_if_labs()
@pytest.mark.parametrize("batch_size", [1, 20, 100])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_inverse(batch_size: int, dtype: torch.dtype):
    import theseus.labs.lie_functional.so3 as so3

    rng = torch.Generator()
    rng.manual_seed(0)
    tangent_vector = torch.rand(batch_size, 3, dtype=dtype, generator=rng)
    group = so3.exp(tangent_vector)

    # check analytic backward for the operator
    check_lie_group_function(so3, "inverse", TEST_EPS, group)


@run_if_labs()
@pytest.mark.parametrize("batch_size", [1, 20, 100])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_hat(batch_size: int, dtype: torch.dtype):
    import theseus.labs.lie_functional.so3 as so3

    rng = torch.Generator()
    rng.manual_seed(0)
    tangent_vector = torch.rand(batch_size, 3, dtype=dtype, generator=rng)

    # check analytic backward for the operator
    check_lie_group_function(so3, "hat", TEST_EPS, tangent_vector)


@run_if_labs()
@pytest.mark.parametrize("batch_size", [1, 20, 100])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_vee(batch_size: int, dtype: torch.dtype):
    import theseus.labs.lie_functional.so3 as so3

    rng = torch.Generator()
    rng.manual_seed(0)
    tangent_vector = torch.rand(batch_size, 3, dtype=dtype, generator=rng)
    matrix = so3.hat(tangent_vector)

    # check analytic backward for the operator
    check_lie_group_function(so3, "vee", TEST_EPS, matrix)

    # check the correctness of hat and vee
    actual_tangent_vector = so3.vee(matrix)
    assert torch.allclose(actual_tangent_vector, tangent_vector, atol=TEST_EPS)
