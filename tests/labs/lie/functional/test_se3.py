# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest

import torch

from tests.decorators import run_if_labs
from .common import BATCH_SIZES_TO_TEST, TEST_EPS, check_lie_group_function, run_test_op


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
        "lift",
        "project",
        "left_act",
        "left_project",
    ],
)
@pytest.mark.parametrize("batch_size", BATCH_SIZES_TO_TEST)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_op(op_name, batch_size, dtype):
    import theseus.labs.lie.functional.se3_impl as se3

    rng = torch.Generator()
    rng.manual_seed(0)
    run_test_op(op_name, batch_size, dtype, rng, 6, (3, 4), se3)


@run_if_labs()
@pytest.mark.parametrize("batch_size", BATCH_SIZES_TO_TEST)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_vee(batch_size: int, dtype: torch.dtype):
    import theseus.labs.lie.functional.se3_impl as se3

    rng = torch.Generator()
    rng.manual_seed(0)
    tangent_vector = torch.rand(batch_size, 6, dtype=dtype, generator=rng)
    matrix = se3._hat_autograd_fn(tangent_vector)

    # check analytic backward for the operator
    check_lie_group_function(se3, "vee", TEST_EPS, (matrix,))

    # check the correctness of hat and vee
    actual_tangent_vector = se3._vee_autograd_fn(matrix)
    torch.testing.assert_close(
        actual_tangent_vector, tangent_vector, atol=TEST_EPS, rtol=TEST_EPS
    )
