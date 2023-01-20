# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest

import torch

from tests.decorators import run_if_labs
from .common import (
    check_lie_group_function,
    get_test_cfg,
    left_project_func,
    sample_inputs,
    TEST_EPS,
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
        "lift",
        "project",
        "left_act",
        "left_project",
    ],
)
@pytest.mark.parametrize("batch_size", [1, 20, 100])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_op(op_name, batch_size, dtype):
    import theseus.labs.lie_functional.se3 as se3

    rng = torch.Generator()
    rng.manual_seed(0)

    all_input_types, atol = get_test_cfg(op_name, dtype, 6, (3, 4), module=se3)
    for input_types in all_input_types:
        inputs = sample_inputs(input_types, batch_size, dtype, rng)
        funcs = (
            tuple(left_project_func(se3, x) for x in inputs)
            if op_name == "log"
            else None
        )

        # check analytic backward for the operator
        check_lie_group_function(se3, op_name, atol, inputs, funcs=funcs)


@run_if_labs()
@pytest.mark.parametrize("batch_size", [1, 20, 100])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_vee(batch_size: int, dtype: torch.dtype):
    import theseus.labs.lie_functional.se3 as se3

    rng = torch.Generator()
    rng.manual_seed(0)
    tangent_vector = torch.rand(batch_size, 6, dtype=dtype, generator=rng)
    matrix = se3.hat(tangent_vector)

    # check analytic backward for the operator
    check_lie_group_function(se3, "vee", TEST_EPS, (matrix,))

    # check the correctness of hat and vee
    actual_tangent_vector = se3.vee(matrix)
    assert torch.allclose(actual_tangent_vector, tangent_vector, atol=TEST_EPS)
