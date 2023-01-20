# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest

import torch

from tests.decorators import run_if_labs
from .common import TEST_EPS, check_lie_group_function, left_project_func


def _get_inputs(input_types, batch_size, dtype, rng, module=None):
    def _sample(input_type):
        type_str, param = input_type

        def _quat_sample():
            q = torch.rand(batch_size, param, dtype=dtype, generator=rng)
            return q / torch.norm(q, dim=1, keepdim=True)

        sample_fns = {
            "tangent": lambda: torch.rand(
                batch_size, param, dtype=dtype, generator=rng
            ),
            "group": lambda: param.rand(batch_size, generator=rng, dtype=dtype),
            "quat": lambda: _quat_sample(),
            "matrix": lambda: torch.rand(
                (batch_size,) + param, generator=rng, dtype=dtype
            ),
        }
        return sample_fns[type_str]()

    return tuple(_sample(type_str) for type_str in input_types)


def _get_test_cfg(op_name, dtype, module=None):
    atol = TEST_EPS
    # input_type --> tuple[str, param]
    # input_types --> a tuple of type info for a given function
    # all_input_types --> a list of input types, if more than one check isn needed
    all_input_types = []
    if op_name == "exp":
        all_input_types.append((("tangent", 3),))
        atol = 1e-6
    if op_name == "log":
        all_input_types.append((("group", module),))
        atol = 5e-6 if dtype == torch.float32 else TEST_EPS
    if op_name in ["adjoint", "inverse"]:
        all_input_types.append((("group", module),))
    if op_name in ["hat"]:
        all_input_types.append((("tangent", 3),))
    if op_name == "compose":
        all_input_types.append((("group", module),) * 2)
    if op_name == "quaternion_to_rotation":
        all_input_types.append((("quat", 4),))
    if op_name == "lift":
        matrix_shape = (torch.randint(1, 20, ()).item(), 3)
        all_input_types.append((("matrix", matrix_shape),))
    if op_name == "project":
        matrix_shape = (torch.randint(1, 20, ()).item(), 3, 3)
        all_input_types.append((("matrix", matrix_shape),))
    if op_name == "left_act":
        for shape in [
            (3, torch.randint(1, 5, ()).item()),
            (2, 4, 3, torch.randint(1, 5, ()).item()),
        ]:
            all_input_types.append((("group", module), ("matrix", shape)))
    if op_name == "left_project":
        for shape in [
            (3, 3),
            (torch.randint(1, 5, ()).item(), 3, 3),
        ]:
            all_input_types.append((("group", module), ("matrix", shape)))
    return all_input_types, atol


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
        "quaternion_to_rotation",
        "lift",
        "project",
        "left_act",
        "left_project",
    ],
)
@pytest.mark.parametrize("batch_size", [1, 20, 100])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_op(op_name, batch_size, dtype):
    import theseus.labs.lie_functional.so3 as so3

    rng = torch.Generator()
    rng.manual_seed(0)

    all_input_types, atol = _get_test_cfg(op_name, dtype, module=so3)
    for input_types in all_input_types:
        inputs = _get_inputs(input_types, batch_size, dtype, rng)
        funcs = (
            tuple(left_project_func(so3, x) for x in inputs)
            if op_name == "log"
            else None
        )

        # check analytic backward for the operator
        check_lie_group_function(so3, op_name, atol, inputs, funcs=funcs)


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
    check_lie_group_function(so3, "vee", TEST_EPS, (matrix,))

    # check the correctness of hat and vee
    actual_tangent_vector = so3.vee(matrix)
    assert torch.allclose(actual_tangent_vector, tangent_vector, atol=TEST_EPS)
