# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch


BATCH_SIZES_TO_TEST = [1, 20]
TEST_EPS = 5e-7


def get_test_cfg(op_name, dtype, dim, data_shape, module=None):
    atol = TEST_EPS
    # input_type --> tuple[str, param]
    # input_types --> a tuple of type info for a given function
    # all_input_types --> a list of input types, if more than one check isn needed
    all_input_types = []
    if op_name in ["exp", "hat"]:
        all_input_types.append((("tangent", dim),))
        atol = 1e-6 if op_name == "exp" else TEST_EPS
    if op_name == "log":
        all_input_types.append((("group", module),))
        atol = 5e-6 if dtype == torch.float32 else TEST_EPS
    if op_name in ["adjoint", "inverse"]:
        all_input_types.append((("group", module),))
    if op_name == "compose":
        all_input_types.append((("group", module),) * 2)
    if op_name == "lift":
        matrix_shape = (torch.randint(1, 20, ()).item(), dim)
        all_input_types.append((("matrix", matrix_shape),))
    if op_name == "project":
        matrix_shape = (torch.randint(1, 20, ()).item(),) + data_shape
        all_input_types.append((("matrix", matrix_shape),))
    if op_name == "left_act":
        for shape in [
            (3, torch.randint(1, 5, ()).item()),
            (2, 4, 3, torch.randint(1, 5, ()).item()),
        ]:
            all_input_types.append((("group", module), ("matrix", shape)))
    if op_name == "left_project":
        for shape in [
            data_shape,
            ((torch.randint(1, 5, ()).item(),) + data_shape),
        ]:
            all_input_types.append((("group", module), ("matrix", shape)))
    return all_input_types, atol


# Sample inputs with the desired types.
def sample_inputs(input_types, batch_size, dtype, rng, module=None):
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


# Run the test for a Lie group operator
def run_test_op(op_name, batch_size, dtype, rng, dim, data_shape, module):
    all_input_types, atol = get_test_cfg(op_name, dtype, dim, data_shape, module=module)
    for input_types in all_input_types:
        inputs = sample_inputs(input_types, batch_size, dtype, rng)
        funcs = (
            tuple(left_project_func(module, x) for x in inputs)
            if op_name == "log"
            else None
        )

        # check analytic backward for the operator
        check_lie_group_function(module, op_name, atol, inputs, funcs=funcs)


# Checks if the jacobian computed by default torch autograd is close to the one
# provided with custom backward
# funcs is a list of callable that modifiies the jacobian. If provided we also
# check that func(jac_autograd) is close to func(jac_custom), for each func in
# the list
def check_lie_group_function(module, op_name: str, atol: float, inputs, funcs=None):
    op_impl = getattr(module, f"_{op_name}_impl")
    op = getattr(module, f"_{op_name}_autograd_fn")

    jacs_impl = torch.autograd.functional.jacobian(op_impl, inputs)
    jacs = torch.autograd.functional.jacobian(op, inputs)

    if funcs is None:
        for jac_impl, jac in zip(jacs_impl, jacs):
            assert torch.allclose(jac_impl, jac, atol=atol)
    else:
        for jac_impl, jac, func in zip(jacs_impl, jacs, funcs):
            if func is None:
                assert torch.allclose(jac_impl, jac, atol=atol)
            else:
                assert torch.allclose(func(jac_impl), func(jac), atol=atol)


def left_project_func(module, group):
    sels = range(group.shape[0])

    def func(matrix: torch.Tensor):
        return module._left_project_autograd_fn(group, matrix[sels, ..., sels, :, :])

    return func
