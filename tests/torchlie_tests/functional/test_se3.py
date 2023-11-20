# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Sequence, Union

import pytest
import torch

import torchlie.functional.se3_impl as se3_impl
from torchlie.functional import SE3

from .common import (
    BATCH_SIZES_TO_TEST,
    TEST_EPS,
    check_binary_op_broadcasting,
    check_left_project_broadcasting,
    check_lie_group_function,
    check_jacrev_binary,
    check_jacrev_unary,
    check_log_map_passt,
    run_test_op,
)


@pytest.mark.parametrize(
    "op_name",
    [
        "exp",
        "log",
        "adjoint",
        "inverse",
        "hat",
        "compose",
        "transform",
        "untransform",
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
    rng = torch.Generator(device="cuda:0" if torch.cuda.is_available() else "cpu")
    rng.manual_seed(0)
    run_test_op(op_name, batch_size, dtype, rng, 6, (3, 4), se3_impl)


@pytest.mark.parametrize("batch_size", BATCH_SIZES_TO_TEST)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_vee(batch_size: Union[int, Sequence[int]], dtype: torch.dtype):
    if isinstance(batch_size, int):
        batch_size = (batch_size,)

    rng = torch.Generator()
    rng.manual_seed(0)
    tangent_vector = torch.rand(*batch_size, 6, dtype=dtype, generator=rng)
    matrix = se3_impl._hat_autograd_fn(tangent_vector)

    # check analytic backward for the operator
    check_lie_group_function(se3_impl, "vee", TEST_EPS, (matrix,))

    # check the correctness of hat and vee
    actual_tangent_vector = se3_impl._vee_autograd_fn(matrix)
    torch.testing.assert_close(
        actual_tangent_vector, tangent_vector, atol=TEST_EPS, rtol=TEST_EPS
    )


@pytest.mark.parametrize("batch_size", [1, 10, 100])
@pytest.mark.parametrize("name", ["exp", "inv"])
def test_jacrev_unary(batch_size, name):
    check_jacrev_unary(SE3, 6, batch_size, name)


@pytest.mark.parametrize("batch_size", [1, 10, 100])
@pytest.mark.parametrize("name", ["compose", "transform", "untransform"])
def test_jacrev_binary(batch_size, name):
    if not hasattr(torch, "vmap"):
        return

    check_jacrev_binary(SE3, batch_size, name)


@pytest.mark.parametrize("name", ["compose", "transform", "untransform"])
def test_binary_op_broadcasting(name):
    rng = torch.Generator()
    rng.manual_seed(0)
    batch_sizes = [(1,), (2,), (1, 2), (2, 1), (2, 2), (2, 2, 2), tuple()]
    for bs1 in batch_sizes:
        for bs2 in batch_sizes:
            check_binary_op_broadcasting(
                SE3, name, (3, 4), bs1, bs2, torch.float64, rng
            )


def test_left_project_broadcasting():
    rng = torch.Generator()
    rng.manual_seed(0)
    batch_sizes = [tuple(), (1, 2), (1, 1, 2), (2, 1), (2, 2), (2, 2, 2)]
    check_left_project_broadcasting(SE3, batch_sizes, [0, 1, 2], (3, 4), rng)


def test_log_map_passt():
    check_log_map_passt(SE3, se3_impl)
