# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch


def check_lie_group_function(module, op_name: str, atol: float, args, funcs=None):
    op_impl = getattr(module, "_" + op_name + "_impl")
    op = getattr(module, op_name)

    jacs_impl = torch.autograd.functional.jacobian(op_impl, args)
    jacs = torch.autograd.functional.jacobian(op, args)

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
        return module.left_project(group, matrix[sels, ..., sels, :, :])

    return func
