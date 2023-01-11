# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch


TEST_EPS = 5e-7


def check_lie_group_function(module, op_name: str, atol: float, *args):
    op_impl = getattr(module, "_" + op_name + "_impl")
    op = getattr(module, op_name)

    jacs_impl = torch.autograd.functional.jacobian(op_impl, args)
    jacs = torch.autograd.functional.jacobian(op, args)

    for jac_impl, jac in zip(jacs_impl, jacs):
        assert torch.allclose(jac_impl, jac, atol=atol)
