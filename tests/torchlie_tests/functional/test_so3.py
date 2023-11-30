# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from typing import Sequence, Union

import pytest
import torch

import torchlie.functional.so3_impl as so3_impl
from torchlie.functional import SO3
from torchlie.global_params import set_global_params

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
    set_global_params({"_faster_log_maps": True})
    rng = torch.Generator(device="cuda:0" if torch.cuda.is_available() else "cpu")
    rng.manual_seed(0)
    run_test_op(op_name, batch_size, dtype, rng, 3, (3, 3), so3_impl)


@pytest.mark.parametrize("batch_size", BATCH_SIZES_TO_TEST)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_vee(batch_size: Union[int, Sequence[int]], dtype: torch.dtype):
    if isinstance(batch_size, int):
        batch_size = (batch_size,)
    rng = torch.Generator()
    rng.manual_seed(0)
    tangent_vector = torch.rand(*batch_size, 3, dtype=dtype, generator=rng)
    matrix = so3_impl._hat_autograd_fn(tangent_vector)

    # check analytic backward for the operator
    check_lie_group_function(so3_impl, "vee", TEST_EPS, (matrix,))

    # check the correctness of hat and vee
    actual_tangent_vector = so3_impl._vee_autograd_fn(matrix)
    torch.testing.assert_close(
        actual_tangent_vector, tangent_vector, atol=TEST_EPS, rtol=TEST_EPS
    )


@pytest.mark.parametrize("batch_size", [1, 10, 100])
@pytest.mark.parametrize("name", ["exp", "inv"])
def test_jacrev_unary(batch_size, name):
    check_jacrev_unary(SO3, 3, batch_size, name)


@pytest.mark.parametrize("batch_size", [1, 10, 100])
@pytest.mark.parametrize("name", ["compose", "transform", "untransform"])
def test_jacrev_binary(batch_size, name):
    if not hasattr(torch, "vmap"):
        return

    check_jacrev_binary(SO3, batch_size, name)


@pytest.mark.parametrize("name", ["compose", "transform", "untransform"])
def test_binary_op_broadcasting(name):
    rng = torch.Generator()
    rng.manual_seed(0)
    batch_sizes = [(1,), (2,), (1, 2), (2, 1), (2, 2), (2, 2, 2), tuple()]
    for bs1 in batch_sizes:
        for bs2 in batch_sizes:
            check_binary_op_broadcasting(
                SO3, name, (3, 3), bs1, bs2, torch.float64, rng
            )


def test_left_project_broadcasting():
    rng = torch.Generator()
    rng.manual_seed(0)
    batch_sizes = [tuple(), (1, 2), (1, 1, 2), (2, 1), (2, 2), (2, 2, 2)]
    check_left_project_broadcasting(SO3, batch_sizes, [0, 1, 2], (3, 3), rng)


def test_log_map_passt():
    check_log_map_passt(SO3, so3_impl)


# This tests that the CUDA implementation of sine axis returns the same result
# as the CPU implementation
@pytest.mark.parametrize("batch_size", [[1], [10], [2, 10]])
def test_sine_axis(batch_size):
    set_global_params({"_faster_log_maps": True})
    if not torch.cuda.is_available():
        return
    for _ in range(10):
        g = so3_impl.rand(*batch_size)
        g_cuda = g.to("cuda:0")
        sa_1 = so3_impl._sine_axis_fn(g, g.shape[:-2])
        sa_2 = so3_impl._sine_axis_fn(g_cuda, g.shape[:-2])
        torch.testing.assert_close(sa_1, sa_2.cpu())
