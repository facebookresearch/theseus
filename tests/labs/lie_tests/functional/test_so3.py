# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Sequence, Union

import pytest
import torch

from tests.decorators import run_if_labs
from .common import (
    BATCH_SIZES_TO_TEST,
    TEST_EPS,
    check_binary_op_broadcasting,
    check_lie_group_function,
    check_jacrev_binary,
    check_jacrev_unary,
    run_test_op,
)


@run_if_labs()
@pytest.mark.parametrize(
    "op_name",
    [
        "exp",
        "log",
        "adjoint",
        "inverse",
        "hat",
        "compose",
        "transform_from",
        "quaternion_to_rotation",
        "lift",
        "project",
        "left_act",
        "left_project",
        "normalize",
    ],
)
@pytest.mark.parametrize("batch_size", BATCH_SIZES_TO_TEST)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_op(op_name, batch_size, dtype):
    import lie.functional.so3_impl as so3

    rng = torch.Generator()
    rng.manual_seed(0)
    run_test_op(op_name, batch_size, dtype, rng, 3, (3, 3), so3)


@run_if_labs()
@pytest.mark.parametrize("batch_size", BATCH_SIZES_TO_TEST)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_vee(batch_size: Union[int, Sequence[int]], dtype: torch.dtype):
    import lie.functional.so3_impl as so3

    if isinstance(batch_size, int):
        batch_size = (batch_size,)
    rng = torch.Generator()
    rng.manual_seed(0)
    tangent_vector = torch.rand(*batch_size, 3, dtype=dtype, generator=rng)
    matrix = so3._hat_autograd_fn(tangent_vector)

    # check analytic backward for the operator
    check_lie_group_function(so3, "vee", TEST_EPS, (matrix,))

    # check the correctness of hat and vee
    actual_tangent_vector = so3._vee_autograd_fn(matrix)
    torch.testing.assert_close(
        actual_tangent_vector, tangent_vector, atol=TEST_EPS, rtol=TEST_EPS
    )


@run_if_labs()
@pytest.mark.parametrize("batch_size", [1, 10, 100])
@pytest.mark.parametrize("name", ["exp", "inv"])
def test_jacrev_unary(batch_size, name):
    import lie.functional as lieF

    check_jacrev_unary(lieF.SO3, 3, batch_size, name)


@run_if_labs()
@pytest.mark.parametrize("batch_size", [1, 10, 100])
@pytest.mark.parametrize("name", ["compose", "transform_from"])
def test_jacrev_binary(batch_size, name):
    if not hasattr(torch, "vmap"):
        return

    import lie.functional as lieF

    check_jacrev_binary(lieF.SO3, batch_size, name)


@run_if_labs()
@pytest.mark.parametrize("name", ["compose", "transform_from"])
def test_binary_op_broadcasting(name):
    import lie.functional as lieF

    print(lieF.__file__)
    from lie.functional import SO3

    rng = torch.Generator()
    rng.manual_seed(0)
    batch_sizes = [(1,), (2,), (1, 2), (2, 1), (2, 2), (2, 2, 2), tuple()]
    for bs1 in batch_sizes:
        for bs2 in batch_sizes:
            check_binary_op_broadcasting(
                SO3, name, (3, 3), bs1, bs2, torch.float64, rng
            )
